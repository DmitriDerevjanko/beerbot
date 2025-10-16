from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Elite_AdaptivePredictive"
VERSION = "v3.0.0"

# === Utilities ===
def moving_avg(values: List[int], window=3):
    vals = values[-window:] if len(values) >= window else values
    return sum(vals) / len(vals) if vals else 0

def trend(values: List[int]) -> float:
    if len(values) < 3:
        return 0
    a, b, c = values[-3:]
    return ((b + c) / 2 - a) / max(a, 1)

def volatility(values: List[int]) -> float:
    if len(values) < 3:
        return 0
    avg = moving_avg(values, 3)
    return sum(abs(v - avg) for v in values[-3:]) / (avg + 1e-6)

def clamp(x, lo, hi): return max(lo, min(hi, x))
def as_int(x): return int(round(x)) if x > 0 else 0

def get_state(w, r): return w["roles"][r]
def incoming_series(weeks, role): return [get_state(w, role)["incoming_orders"] for w in weeks]
def prev_order(weeks, role): 
    if len(weeks) >= 2: return weeks[-2]["orders"][role]
    return 10

# === Decision core ===
def decide_elite(weeks, role):
    last = get_state(weeks[-1], role)
    inv, back = last["inventory"], last["backlog"]
    inc = incoming_series(weeks, role)
    avg_demand = moving_avg(inc, 3)
    tr = trend(inc)
    vol = volatility(inc)
    prev = prev_order(weeks, role)

    # Forecast next demand
    forecast = avg_demand * (1 + clamp(tr * 0.7, -0.3, 0.4))

    # Dynamic safety stock (reduces when demand stable)
    safety = (1.0 + min(vol, 0.5)) * forecast

    # Adaptive gain: stronger when backlog high
    gain = 0.4 + clamp(back / 15, 0, 0.3)

    # Error control
    error = safety - inv
    base_order = forecast + back * 0.8 + gain * error

    # Anti-overshoot: slow decrease if inventory >> target
    if inv > 1.5 * safety:
        base_order *= 0.7

    # Smooth momentum
    damping = 0.5 if back > 5 else 0.35
    new_order = prev + damping * (base_order - prev)
    new_order = clamp(new_order, prev - 3, prev + 3)
    return as_int(new_order)

def decide_blackbox(weeks):
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    return {r: decide_elite(weeks, r) for r in roles}

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
