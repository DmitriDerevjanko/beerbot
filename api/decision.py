from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import math

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Ultra_v3_3_MaxDet"
VERSION = "v3.3"

ROLES = ["retailer", "wholesaler", "distributor", "factory"]
ROLE_LEADTIME = {"retailer": 1.4, "wholesaler": 2.2, "distributor": 2.7, "factory": 3.1}

DEFAULT_INV_COST = 1.0
DEFAULT_BACKLOG_COST = 2.0


# ---------- helpers (deterministic) ----------
def fround(x: float, nd: int = 6) -> float:
    return float(f"{x:.{nd}f}")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ema(xs: List[float], alpha: float, default: float = 10.0) -> float:
    if not xs:
        return default
    v = xs[0]
    for x in xs[1:]:
        v = alpha * x + (1 - alpha) * v
    return v

def holt_linear(xs: List[float], alpha: float, gamma: float, default: float = 10.0) -> float:
    """Holt's linear (no seasonality). Returns 1-step-ahead forecast."""
    if not xs:
        return default
    l = xs[0]
    b = xs[1] - xs[0] if len(xs) > 1 else 0.0
    for x in xs:
        l_new = alpha * x + (1 - alpha) * (l + b)
        b_new = gamma * (l_new - l) + (1 - gamma) * b
        l, b = l_new, b_new
    return l + b

def rolling_std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(0.0, var))

def slope(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(xs) / n
    num = sum((i - mx) * (xs[i] - my) for i in range(n))
    den = sum((i - mx) ** 2 for i in range(n)) or 1.0
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


# ---------- core policy (no global state) ----------
def demand_anchor(weeks: List[Dict[str, Any]]) -> float:
    r_in = vec(weeks, "retailer", "incoming_orders")
    f_s = ema(r_in[-6:], 0.6, 10.0)
    f_l = ema(r_in[-16:], 0.22, f_s)
    return 0.65 * f_s + 0.35 * f_l

def pipeline_estimate(weeks: List[Dict[str, Any]], role: str, L: float) -> float:
    arr = vec(weeks, role, "arriving_shipments")
    k = max(1, int(math.ceil(L)))
    weight = 0.9 if role == "factory" else 0.8
    return sum(arr[-k:]) * weight

def integral_from_history(back_hist: List[float], decay: float = 0.92) -> float:
    s = 0.0
    for b in back_hist:
        s = decay * s + b
    return s

def order_for(role: str,
              weeks: List[Dict[str, Any]],
              inv_unit: float,
              back_unit: float,
              weeks_total: Optional[int]) -> int:
    if not weeks:
        return 10

    L = ROLE_LEADTIME.get(role, 3.0)
    inv = vec(weeks, role, "inventory")[-1]
    back_hist = vec(weeks, role, "backlog")
    back = back_hist[-1]
    in_orders = vec(weeks, role, "incoming_orders")

    # Forecast: strong global anchor + Holt's linear for local series
    anchor = demand_anchor(weeks)
    holt = holt_linear(in_orders[-12:], alpha=0.55, gamma=0.15, default=anchor)
    f_local_s = ema(in_orders[-6:], 0.55, holt)
    f_local_l = ema(in_orders[-14:], 0.25, f_local_s)
    local_mix = 0.6 * f_local_s + 0.4 * f_local_l
    forecast = 0.75 * anchor + 0.25 * local_mix
    forecast = max(4.0, forecast)

    # Safety: variance- and cost-aware
    win = in_orders[-10:]
    sig = rolling_std(win)
    cost_ratio = back_unit / max(1e-6, inv_unit)
    safety = (0.18 * forecast + 0.55 * sig) * clamp(cost_ratio, 0.7, 2.2)

    # Trend moderation
    tr = slope(in_orders[-8:])
    trend_factor = 1.0 + clamp(tr / max(1.0, forecast), -0.25, 0.25)

    # Stronger end-of-horizon drain (last 6 weeks)
    if weeks_total:
        left = max(0, weeks_total - len(weeks))
        if left < L + 3:
            shrink = clamp((L + 3 - left) / (L + 3), 0.0, 1.0)
            trend_factor *= (1.0 - 0.55 * shrink)
            safety *= (1.0 - 0.80 * shrink)
        if left < 6:
            shrink2 = (6 - left) / 6.0
            trend_factor *= (1.0 - 0.70 * shrink2)
            safety *= (1.0 - 0.88 * shrink2)

    # Target position with pipeline
    pipe = pipeline_estimate(weeks, role, L)
    target_pos = trend_factor * (L * forecast) + safety
    inv_pos = inv + pipe - back

    base = forecast + (target_pos - inv_pos) / max(1.0, L)

    # Adaptive PI backlog controller (deterministic)
    # P is capped, Kp grows with backlog via smooth factor; Ki from history
    gain_boost = 1 / (1 + math.exp(- (back - 40) / 12))  # 0..1
    kp = 0.38 + 0.08 * gain_boost
    ki = 0.055
    integ = integral_from_history(back_hist)
    p_term = kp * min(back, 55.0)
    ctrl = p_term + ki * integ

    raw = base + ctrl

    # Phase-aware step cap (tighter mid-horizon)
    prev = in_orders[-1] if in_orders else forecast
    vol = max(6.0, forecast + 0.8 * sig)
    phase = len(weeks) / max(1.0, (weeks_total or 36))
    base_cap = 0.17 if 0.35 < phase < 0.80 else 0.20
    step_cap = (base_cap - 0.05 * clamp(abs(tr) / max(1.0, forecast), 0, 1)) * vol
    damped = prev + clamp(raw - prev, -step_cap, step_cap)

    # Tighter dynamic bounds (anti-bullwhip)
    lo = max(4.0, 0.60 * anchor)
    hi = max(lo + 2.0, 1.36 * anchor + 0.28 * sig + 0.24 * back / max(1.0, L))

    order = int(clamp(fround(damped), fround(lo), fround(hi)))
    return order


def decide(weeks: List[Dict[str, Any]], costs: Dict[str, float]) -> Dict[str, int]:
    inv_c = float(costs.get("inventory_per_unit", DEFAULT_INV_COST))
    back_c = float(costs.get("backlog_per_unit", DEFAULT_BACKLOG_COST))
    weeks_total = int(costs.get("weeks_total", 0)) or None
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
    if "weeks_total" not in costs and body.get("weeks_total"):
        costs["weeks_total"] = body["weeks_total"]

    orders = decide(weeks, costs)

    inv_c = float(costs.get("inventory_per_unit", DEFAULT_INV_COST))
    back_c = float(costs.get("backlog_per_unit", DEFAULT_BACKLOG_COST))
    metrics = compute_metrics(weeks, inv_cost=inv_c, backlog_cost=back_c)

    return JSONResponse({"orders": orders, "metrics": metrics})
