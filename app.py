from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import math

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_PRO_Predictive"
VERSION = "v2.0.0"


# === UTILITIES ===
def moving_avg(values: List[int], window: int = 3) -> float:
    if not values:
        return 0.0
    use = values[-window:] if len(values) >= window else values
    return sum(use) / len(use)


def trend_factor(values: List[int]) -> float:
    """Detect demand trend: + if growing, - if shrinking."""
    if len(values) < 3:
        return 0.0
    a, b, c = values[-3:]
    if a == 0:
        return 0.0
    return ((b + c) / 2 - a) / max(a, 1)  # relative trend (% change)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def as_int_nonneg(x: float) -> int:
    return int(round(x)) if x > 0 else 0


def get_state(week_obj: Dict[str, Any], role: str) -> Dict[str, int]:
    return week_obj["roles"][role]


def get_incoming(weeks: List[Dict[str, Any]], role: str) -> List[int]:
    return [int(get_state(w, role)["incoming_orders"]) for w in weeks]


def get_prev_order(weeks: List[Dict[str, Any]], role: str) -> int:
    if len(weeks) >= 2:
        return int(weeks[-2]["orders"][role])
    return 10


# === CORE DECISION ===
def decide_predictive(weeks: List[Dict[str, Any]], role: str) -> int:
    last = get_state(weeks[-1], role)
    inv, back = int(last["inventory"]), int(last["backlog"])
    incoming = get_incoming(weeks, role)
    avg_demand = moving_avg(incoming, 3)
    prev_order = get_prev_order(weeks, role)

    # --- Trend analysis ---
    tr = trend_factor(incoming)
    trend_adj = clamp(tr * 0.5, -0.3, 0.3)  # -30%...+30%

    # --- Adaptive target ---
    target_stock = max(3, avg_demand * (1.1 + trend_adj))
    error = target_stock - inv

    # --- Base PI control ---
    base_order = avg_demand + (back * 0.8) + (0.5 * error)

    # --- Damping (momentum smoothing) ---
    damping = 0.4 if abs(error) < 5 else 0.6
    smoothed = prev_order + damping * (base_order - prev_order)

    # --- Clamp change per week (prevent shocks) ---
    limited = clamp(smoothed, prev_order - 5, prev_order + 5)

    return as_int_nonneg(limited)


# === ROLES ===
def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    return {r: decide_predictive(weeks, r) for r in roles}


def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    r_in, w_in, d_in = (
        get_incoming(weeks, x) for x in ["retailer", "wholesaler", "distributor"]
    )
    last = weeks[-1]
    r, w, d, f = (get_state(last, x) for x in ["retailer", "wholesaler", "distributor", "factory"])

    return {
        "retailer": decide_predictive(weeks, "retailer"),
        "wholesaler": decide_predictive(weeks, "wholesaler"),
        "distributor": decide_predictive(weeks, "distributor"),
        "factory": decide_predictive(weeks, "factory"),
    }


# === API ===
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
    mode = body.get("mode", "blackbox")
    if not weeks:
        return JSONResponse({"orders": {"retailer": 10, "wholesaler": 10, "distributor": 10, "factory": 10}})

    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)
    return JSONResponse({"orders": orders})
