from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import numpy as np
import copy

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Ultra_CostAware_v5.3"
VERSION = "v5.3.0"

# === HELPERS ===
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def as_int(x):
    return int(round(x)) if x > 0 else 0

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

def get_state(week, role): 
    return week["roles"][role]

def incoming_series(weeks, role): 
    return [get_state(w, role)["incoming_orders"] for w in weeks]

def prev_order(weeks, role): 
    if len(weeks) >= 2:
        return weeks[-2]["orders"][role]
    return 10


# === MAIN DECISION LOGIC (v5.3) ===
def decide_ultra(weeks, role):
    last = get_state(weeks[-1], role)
    inv, back = last["inventory"], last["backlog"]
    inc = incoming_series(weeks, role)
    avg_demand = moving_avg(inc, 4)
    prev = prev_order(weeks, role)
    slope = linear_trend(inc)

    # --- Base forecast ---
    forecast = max(0, avg_demand + slope * 1.1)
    forecast *= 1.05 if slope > 0 else 0.95

    # --- Adaptive safety stock ---
    volatility = np.std(inc[-4:]) if len(inc) >= 4 else 0
    safety_stock = avg_demand * (1.2 + 0.5 * volatility / (avg_demand + 1e-6))
    if back > 10:
        safety_stock *= 1.3

    # --- Backlog correction ---
    backlog_weight = 1.0 + clamp(back / 25, 0, 2.5)
    inventory_penalty = 0.4 if inv > 1.3 * safety_stock else 0.6
    correction = (safety_stock - inv) * inventory_penalty + back * 0.9 * backlog_weight

    # --- Predictive boost for rising demand ---
    if len(inc) >= 3 and inc[-1] > inc[-2] * 1.2 and inc[-2] > inc[-3] * 1.2:
        forecast *= 1.25
        correction += back * 0.4

    # --- Combine forecast and correction ---
    base_order = forecast + correction
    base_order = max(base_order, avg_demand * 0.9)

    # --- Dynamic damping (adaptive smoothing) ---
    ratio = back / (back + inv + 1e-6)
    damping = 0.45 + 0.4 * ratio
    damping = clamp(damping, 0.45, 0.85)

    # --- Shipping delay simulation (1-week buffer) ---
    shipping_delay = 1.0 + 0.2 * ratio  # backlog увеличивает лаг
    delayed_order = base_order * shipping_delay

    # --- Overshoot control ---
    if inv > 1.5 * safety_stock and back < 5:
        delayed_order *= 0.75

    # --- Smooth order update ---
    order = prev + damping * (delayed_order - prev)

    # --- Adaptive clamp range ---
    max_change = clamp(6 + back / 6, 5, 20)
    order = clamp(order, prev - max_change, prev + max_change)

    # --- Emergency boost if inventory is empty ---
    if inv == 0 and back > 0:
        order += min(back * 0.5, 25)

    # --- Cooldown when demand drops ---
    if len(inc) >= 3 and inc[-1] < inc[-2] * 0.8:
        order *= 0.9

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


# === LOCAL TEST SECTION ===
if __name__ == "__main__":
    import json

    with open("test.json", "r") as f:
        data = json.load(f)
    old_weeks = data["weeklyData"]

    backlog_penalty = 2
    inventory_penalty = 1

    def calc_cost(weeks):
        inv_cost = sum(w["roles"][r]["inventory"] * inventory_penalty for w in weeks for r in ["retailer","wholesaler","distributor","factory"])
        back_cost = sum(w["roles"][r]["backlog"] * backlog_penalty for w in weeks for r in ["retailer","wholesaler","distributor","factory"])
        return inv_cost, back_cost, inv_cost + back_cost

    old_inv, old_back, old_total = calc_cost(old_weeks)
    print("\n=== OLD COST (from test.json) ===")
    print(f"Inventory cost: {old_inv:.0f}")
    print(f"Backlog cost:   {old_back:.0f}")
    print(f"Total cost:     {old_total:.0f}")

    # Simple re-simulation for estimation
    new_weeks = [copy.deepcopy(old_weeks[0])]
    for i in range(1, len(old_weeks)):
        subset = new_weeks[-5:] if len(new_weeks) > 5 else new_weeks
        new_orders = decide_blackbox(subset)
        last_state = copy.deepcopy(new_weeks[-1])
        next_state = {"roles": {}, "orders": new_orders}
        for r in ["retailer", "wholesaler", "distributor", "factory"]:
            prev = last_state["roles"][r]
            demand = old_weeks[i]["roles"][r]["incoming_orders"]
            inventory = prev["inventory"]
            shipped = min(inventory, demand)
            inv_new = max(0, inventory - shipped + new_orders[r])
            back_new = max(0, demand - inventory)
            next_state["roles"][r] = {
                "inventory": inv_new,
                "backlog": back_new,
                "incoming_orders": demand,
            }
        new_weeks.append(next_state)

    new_inv, new_back, new_total = calc_cost(new_weeks)
    improvement = (old_total - new_total) / old_total * 100

    print("\n=== NEW SIMULATION (BeerBot_Ultra_CostAware_v5.3) ===")
    print(f"Inventory cost: {new_inv:.0f}")
    print(f"Backlog cost:   {new_back:.0f}")
    print(f"Total cost:     {new_total:.0f}")
    print(f"Improvement: {improvement:+.1f}% ✅")
