from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import math

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_OrderUpTo_Elite"
VERSION = "v2.3"

ROLES = ["retailer", "wholesaler", "distributor", "factory"]
ROLE_LEADTIME = {"retailer": 1.5, "wholesaler": 2.3, "distributor": 2.8, "factory": 3.2}

DEFAULT_INV_COST = 1.0
DEFAULT_BACKLOG_COST = 2.0


# ---------- helpers ----------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ema(xs: List[float], alpha: float, default: float = 10.0) -> float:
    if not xs:
        return default
    v = xs[0]
    for x in xs[1:]:
        v = alpha * x + (1 - alpha) * v
    return v

def slope(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(xs) / n
    num = sum((i - mean_x) * (xs[i] - mean_y) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n)) or 1.0
    return num / den

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


# ---------- core policy ----------
def order_for(role: str, weeks: List[Dict[str, Any]], inv_unit: float, back_unit: float, weeks_total: int | None) -> int:
    if not weeks:
        return 10

    # role data
    L = ROLE_LEADTIME.get(role, 3.0)
    inv = vec(weeks, role, "inventory")[-1]
    back = vec(weeks, role, "backlog")[-1]
    in_orders = vec(weeks, role, "incoming_orders")

    # demand forecast with dual EMA
    f_short = ema(in_orders[-6:], alpha=0.55, default=10.0)
    f_long  = ema(in_orders[-14:], alpha=0.25, default=f_short)
    forecast = 0.6 * f_short + 0.4 * f_long
    forecast = max(4.0, forecast)

    # trend control (if demand trend is down, reduce target)
    tr = slope(in_orders[-8:])
    trend_factor = 1.0 + clamp(tr / max(1.0, forecast), -0.25, 0.25)

    # cost-aware safety
    cost_ratio = back_unit / max(1e-6, inv_unit)  # >1 means backlog is more expensive
    base_safety = 0.30 * forecast
    safety = base_safety * clamp(cost_ratio, 0.7, 2.0)

    # end-of-horizon drain
    if weeks_total:
        w_now = len(weeks)
        left = max(0, weeks_total - w_now)
        if left < L + 2:
            # shrink target smoothly as horizon ends
            shrink = clamp((L + 2 - left) / (L + 2), 0.0, 1.0)
            trend_factor *= (1.0 - 0.5 * shrink)
            safety *= (1.0 - 0.7 * shrink)

    target_pos = trend_factor * (L * forecast) + safety

    # approximate inventory position (no pipeline visibility)
    inv_pos = inv - back

    # base order from order-up-to
    base = forecast + (target_pos - inv_pos) / max(1.0, L)

    # controlled backlog clearance
    horizon = max(2.0, L + 1.0)
    clear_term = back / horizon
    raw = base + 0.45 * clear_term

    # damping against last observed incoming orders
    prev = in_orders[-1] if in_orders else forecast
    max_step = 0.22 * max(6.0, forecast)
    damped = prev + clamp(raw - prev, -max_step, max_step)

    # dynamic bounds
    lo = max(4.0, 0.55 * forecast)
    hi = max(lo + 2.0, 1.55 * forecast + 0.5 * clear_term)
    return int(clamp(damped, lo, hi))


def decide(weeks: List[Dict[str, Any]], costs: Dict[str, float], mode: str) -> Dict[str, int]:
    inv_c = float(costs.get("inventory_per_unit", DEFAULT_INV_COST))
    back_c = float(costs.get("backlog_per_unit", DEFAULT_BACKLOG_COST))
    weeks_total = int(costs.get("weeks_total", 0)) or None  # allow passing via costs or body root

    return {r: order_for(r, weeks, inv_c, back_c, weeks_total) for r in ROLES}


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

    costs = body.get("costs", {}) or {}
    # accept weeks_total either in body root or in costs
    if "weeks_total" not in costs and body.get("weeks_total"):
        costs["weeks_total"] = body["weeks_total"]

    mode = (body.get("mode") or "blackbox").lower()
    orders = decide(weeks, costs, mode)

    inv_c = float(costs.get("inventory_per_unit", DEFAULT_INV_COST))
    back_c = float(costs.get("backlog_per_unit", DEFAULT_BACKLOG_COST))
    metrics = compute_metrics(weeks, inv_cost=inv_c, backlog_cost=back_c)

    return JSONResponse({"orders": orders, "metrics": metrics})
