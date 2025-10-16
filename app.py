from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Ultra_CostAware"
VERSION = "v4.0.0"

def moving_avg(values: List[int], window=3):
    vals = values[-window:] if len(values) >= window else values
    return sum(vals) / len(vals) if vals else 0

def linear_trend(values: List[int]) -> float:
    """Approximate linear trend (slope) for last 4 points."""
    if len(values) < 4:
        return 0.0
    n = 4
    x = list(range(n))
    y = values[-n:]
    avg_x, avg_y = sum(x)/n, sum(y)/n
    num = sum((xi-avg_x)*(yi-avg_y) for xi, yi in zip(x,y))
    den = sum((xi-avg_x)**2 for xi in x)
    return num / den if den else 0.0

def clamp(x, lo, hi): return max(lo, min(hi, x))
def as_int(x): return int(round(x)) if x > 0 else 0

def get_state(week, role): return week["roles"][role]
def incoming_series(weeks, role): return [get_state(w, role)["incoming_orders"] for w in weeks]
def prev_order(weeks, role): 
    if len(weeks) >= 2: return weeks[-2]["orders"][role]
    return 10

def decide_ultra(weeks, role):
    last = get_state(weeks[-1], role)
    inv, back = last["inventory"], last["backlog"]
    inc = incoming_series(weeks, role)
    avg_demand = moving_avg(inc, 3)
    prev = prev_order(weeks, role)

    # --- Trend forecasting ---
    slope = linear_trend(inc)
    forecast = avg_demand + slope * 0.5
    forecast = max(0, forecast)

    # --- Dynamic safety stock ---
    volatility = sum(abs(inc[-i] - avg_demand) for i in range(1, min(3, len(inc))) )
    stability_factor = 1.0 - clamp(volatility / (avg_demand + 1e-6), 0, 0.5)
    safety_stock = avg_demand * (0.9 + 0.2 * stability_factor)

    # --- Cost-aware correction ---
    backlog_weight = 1.0 if back > 5 else 0.6
    inventory_penalty = 0.3 if inv > 1.2 * safety_stock else 0.5
    correction = (safety_stock - inv) * inventory_penalty + back * backlog_weight

    # --- Base order ---
    base_order = forecast + correction
    base_order = max(base_order, avg_demand * 0.5)

    # --- Anti-overshoot & damping ---
    if inv > 1.5 * safety_stock:
        base_order *= 0.7
    damping = 0.35 if inv > safety_stock else 0.55
    order = prev + damping * (base_order - prev)

    # --- Clamp weekly change ---
    order = clamp(order, prev - 2, prev + 2)
    return as_int(order)

def decide_blackbox(weeks):
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    return {r: decide_ultra(weeks, r) for r in roles}

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
            "message": "BeerBot ready"
        })
    weeks = body.get("weeks", [])
    if not weeks:
        return JSONResponse({"orders": {r: 10 for r in ["retailer","wholesaler","distributor","factory"]}})
    return JSONResponse({"orders": decide_blackbox(weeks)})
