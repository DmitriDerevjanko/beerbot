from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

# === Your Student Info ===
STUDENT_EMAIL = "dmdere@taltech.ee"  
ALGO_NAME = "BeerBot_PI_Smooth"
VERSION = "v1.0.0"

# === Helper functions ===
def moving_avg(values: List[int], window: int = 3) -> float:
    """Compute moving average for smoothing demand."""
    if not values:
        return 0.0
    use = values[-window:] if len(values) >= window else values
    return sum(use) / len(use)

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def as_int_nonneg(x: float) -> int:
    """Ensure non-negative integer output."""
    return int(round(x)) if x > 0 else 0

def get_role_state(week_obj: Dict[str, Any], role: str) -> Dict[str, int]:
    """Extract a role's state from the week object."""
    return week_obj["roles"][role]

def series_incoming(weeks: List[Dict[str, Any]], role: str) -> List[int]:
    """List of incoming orders per role."""
    return [int(get_role_state(w, role)["incoming_orders"]) for w in weeks]

def decide_for_role(last_state: Dict[str, int], recent_incoming: List[int]) -> int:
    """Core deterministic decision logic."""
    inv = int(last_state["inventory"])
    back = int(last_state["backlog"])
    avg_demand = moving_avg(recent_incoming, 3)
    target = max(5, round(1.5 * avg_demand))   # maintain ~1.5 weeks of demand
    adj = clamp(target - inv, -6, 6)           # smooth correction
    order = avg_demand + back + adj
    return as_int_nonneg(order)

def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Decide orders independently for each role (BlackBox)."""
    last = weeks[-1]
    return {
        role: decide_for_role(get_role_state(last, role), series_incoming(weeks, role))
        for role in ["retailer", "wholesaler", "distributor", "factory"]
    }

def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Decide orders using info from downstream roles (GlassBox)."""
    last = weeks[-1]
    # get all states
    r = get_role_state(last, "retailer")
    w = get_role_state(last, "wholesaler")
    d = get_role_state(last, "distributor")
    f = get_role_state(last, "factory")

    # downstream demand signals
    r_in = series_incoming(weeks, "retailer")
    w_in = series_incoming(weeks, "wholesaler")
    d_in = series_incoming(weeks, "distributor")

    return {
        "retailer": decide_for_role(r, r_in),
        "wholesaler": decide_for_role(w, r_in),
        "distributor": decide_for_role(d, w_in),
        "factory": decide_for_role(f, d_in),
    }

# === API Endpoint ===
@app.post("/api/decision")
async def decision(req: Request):
    body = await req.json()

    # --- Handshake ---
    if body.get("handshake"):
        return JSONResponse({
            "ok": True,
            "student_email": STUDENT_EMAIL,
            "algorithm_name": ALGO_NAME,
            "version": VERSION,
            "supports": {"blackbox": True, "glassbox": True},
            "message": "BeerBot ready"
        })

    # --- Weekly step ---
    weeks = body.get("weeks", [])
    mode = body.get("mode", "blackbox")

    if not weeks:
        # fallback in case of empty input
        return JSONResponse({"orders": {"retailer": 10, "wholesaler": 10, "distributor": 10, "factory": 10}})

    if mode == "glassbox":
        orders = decide_glassbox(weeks)
    else:
        orders = decide_blackbox(weeks)

    return JSONResponse({"orders": orders})
