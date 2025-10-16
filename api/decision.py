from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import List, Dict
import numpy as np

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Adaptive_v5"
VERSION = "v5.0.0"

# === HELPERS ===
def clamp(x, lo, hi): return max(lo, min(hi, x))
def as_int(x): return max(0, int(round(x)))

def moving_avg(values: List[int], window=3):
    vals = values[-window:] if len(values) >= window else values
    return np.mean(vals) if vals else 0

def linear_trend(values: List[int]) -> float:
    if len(values) < 4:
        return 0.0
    n = 4
    x = np.arange(n)
    y = np.array(values[-n:])
    slope = np.polyfit(x, y, 1)[0]
    return slope

def get_state(week, role): return week["roles"][role]
def incoming_series(weeks, role): return [get_state(w, role)["incoming_orders"] for w in weeks]
def prev_order(weeks, role): return weeks[-2]["orders"][role] if len(weeks) >= 2 else 10

# === ROLE ADAPTATION ===
ROLE_PARAMS = {
    "retailer": dict(alpha=0.9, safety=1.3),
    "wholesaler": dict(alpha=0.7, safety=1.2),
    "distributor": dict(alpha=0.6, safety=1.1),
    "factory": dict(alpha=0.4, safety=1.0)
}

# === DECISION FUNCTION ===
def decide_adaptive(weeks, role):
    params = ROLE_PARAMS[role]
    last = get_state(weeks[-1], role)
    inv, back = last["inventory"], last["backlog"]
    in_transit = sum(last.get("in_transit", []))
    total_stock = inv + in_transit

    inc = incoming_series(weeks, role)
    avg_demand = moving_avg(inc, 4)
    slope = linear_trend(inc)

    # --- Adaptive forecast ---
    forecast = avg_demand + slope * 0.7
    forecast = max(forecast, avg_demand * 0.5)

    # --- Dynamic safety stock ---
    volatility = np.std(inc[-4:]) if len(inc) >= 4 else 0
    safety_stock = (avg_demand + volatility) * params["safety"]

    # --- Target inventory based on backlog pressure ---
    backlog_factor = 0.8 if back > 10 else 0.4
    demand_factor = 1.0 + min(0.3, slope / (avg_demand + 1e-6))
    target_inventory = avg_demand * demand_factor + back * backlog_factor

    # --- Correction ---
    gap = target_inventory - total_stock
    correction = gap * 0.8 + back * 0.6

    base_order = forecast + correction
    base_order = max(base_order, avg_demand * 0.7)

    # --- Smooth reaction ---
    prev = prev_order(weeks, role)
    alpha = params["alpha"]
    order = prev + alpha * (base_order - prev)

    # --- Clamp within reasonable range ---
    change_limit = clamp(3 + back / 30, 3, 10)
    order = clamp(order, prev - change_limit, prev + change_limit)

    return as_int(order)

# === MAIN LOOP ===
def decide_blackbox(weeks):
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    return {r: decide_adaptive(weeks, r) for r in roles}

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


