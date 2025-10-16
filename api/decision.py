from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_OrderUpTo_Pro"
VERSION = "v2.0"

ROLES = ["retailer", "wholesaler", "distributor", "factory"]
ROLE_LEADTIME = {  # effective lead-time in weeks per role (tune if needed)
    "retailer": 1.5,
    "wholesaler": 2.5,
    "distributor": 3.0,
    "factory": 3.5,
}

DEFAULT_INV_COST = 1.0
DEFAULT_BACKLOG_COST = 2.0


# ---------- helpers ----------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ema(series: List[float], alpha: float, default: float = 10.0) -> float:
    if not series:
        return default
    v = series[0]
    for x in series[1:]:
        v = alpha * x + (1 - alpha) * v
    return v


def vec(weeks: List[Dict[str, Any]], role: str, key: str) -> List[float]:
    out = []
    for w in weeks:
        out.append(float(w.get("roles", {}).get(role, {}).get(key, 0)))
    return out


def compute_metrics(weeks: List[Dict[str, Any]], inv_cost=DEFAULT_INV_COST, backlog_cost=DEFAULT_BACKLOG_COST):
    inv_sum = 0.0
    back_sum = 0.0
    peak_inv = 0
    peak_back = 0
    for w in weeks:
        for r in ROLES:
            st = w["roles"].get(r, {})
            inv = int(st.get("inventory", 0))
            back = int(st.get("backlog", 0))
            inv_sum += inv
            back_sum += back
            peak_inv = max(peak_inv, inv)
            peak_back = max(peak_back, back)
    inv_cost_total = inv_sum * inv_cost
    back_cost_total = back_sum * backlog_cost
    total_cost = inv_cost_total + back_cost_total
    return {
        "inventory_cost": int(round(inv_cost_total)),
        "backlog_cost": int(round(back_cost_total)),
        "total_cost": int(round(total_cost)),
        "peak_inventory": int(peak_inv),
        "peak_backlog": int(peak_back),
    }


# ---------- policy ----------
def order_for(role: str, weeks: List[Dict[str, Any]]) -> int:
    if not weeks:
        return 10

    L = ROLE_LEADTIME.get(role, 3.0)
    inv = vec(weeks, role, "inventory")[-1]
    back = vec(weeks, role, "backlog")[-1]
    in_orders = vec(weeks, role, "incoming_orders")

    # Demand forecast (short EMA reacts, long EMA stabilizes)
    f_short = ema(in_orders[-6:], alpha=0.5, default=10.0)
    f_long  = ema(in_orders[-12:], alpha=0.25, default=f_short)
    forecast = 0.6 * f_short + 0.4 * f_long

    # Target inventory position: demand over lead time + safety
    safety = 0.35 * forecast
    target_pos = L * forecast + safety

    # Current (approximate) inventory position (no pipeline info available)
    inv_pos = inv - back

    # Base order from order-up-to logic
    base = forecast + (target_pos - inv_pos) / max(1.0, L)

    # Backlog clearance spread across a horizon (avoid spikes)
    horizon = max(2.0, L)
    clear_term = back / horizon

    raw = base + 0.5 * clear_term  # 0.5 weight to avoid overshoot

    # Damping vs last observed incoming orders (proxy for last order)
    prev = in_orders[-1] if in_orders else forecast
    max_step = 0.25 * max(6.0, forecast)  # limit per-week change
    damped = prev + clamp(raw - prev, -max_step, max_step)

    # Dynamic bounds relative to forecast to reduce bullwhip
    lo = max(4.0, 0.6 * forecast)
    hi = max(lo + 2.0, 1.6 * forecast + 0.6 * clear_term)
    return int(clamp(damped, lo, hi))


def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return {r: order_for(r, weeks) for r in ROLES}


def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return decide_blackbox(weeks)


# ---------- API ----------
@app.post("/api/decision")
async def decision(req: Request):
    body = await req.json()
    if body.get("handshake"):
        return JSONResponse({
            "ok": True,
            "student_email": STUDENT_EMAIL,
            "algorithm_name": ALGO_NAME,
            "version": VERSION,
            "supports": {"blackbox": True, "glassbox": True},
            "message": "Ready",
        })

    weeks = body.get("weeks", [])
    if not weeks:
        return JSONResponse({"orders": {r: 10 for r in ROLES}})

    mode = body.get("mode", "blackbox").lower()
    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)

    costs = body.get("costs", {}) or {}
    inv_c = float(costs.get("inventory_per_unit", DEFAULT_INV_COST))
    back_c = float(costs.get("backlog_per_unit", DEFAULT_BACKLOG_COST))
    metrics = compute_metrics(weeks, inv_cost=inv_c, backlog_cost=back_c)

    return JSONResponse({"orders": orders, "metrics": metrics})
