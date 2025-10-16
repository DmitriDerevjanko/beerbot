from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Ultra_CostAware"
VERSION = "v6.0"

# --------------------- Helper Functions ---------------------
def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a value between lower and upper bounds."""
    return max(lo, min(hi, x))

def ema(series: List[float], alpha: float) -> float:
    """Compute exponential moving average (EMA) for a time series."""
    if not series:
        return 0.0
    v = series[0]
    for x in series[1:]:
        v = alpha * x + (1 - alpha) * v
    return v

def get_state(week: Dict[str, Any], role: str) -> Dict[str, int]:
    """Return the state dictionary for a specific role in a given week."""
    return week["roles"][role]

def incoming_series(weeks: List[Dict[str, Any]], role: str) -> List[int]:
    """Extract the list of incoming orders for a given role across all weeks."""
    return [int(get_state(w, role)["incoming_orders"]) for w in weeks]

def prev_order(weeks: List[Dict[str, Any]], role: str) -> int:
    """Return the previous week's order for a given role, or 10 if not available."""
    if len(weeks) >= 2:
        return int(weeks[-2]["orders"][role])
    return 10

# --------------------- APIOBPCS+ Ultra Logic ---------------------
def decide_beerbot(weeks: List[Dict[str, Any]], role: str) -> int:
    """
    BeerBot Ultra Cost-Aware v6.0
    Advanced APIOBPCS+ with momentum damping, adaptive backlog control, and demand momentum.
    """
    last = weeks[-1]["roles"][role]
    inv  = int(last["inventory"])
    back = int(last["backlog"])
    inc  = incoming_series(weeks, role)
    prev = prev_order(weeks, role)

    # --- Typical logistics delay (2 weeks in the MIT Beer Game)
    L = 2

    # --- Base parameters
    b_base = 0.80
    k_I    = 0.35      # slightly reduced inventory correction
    k_P    = 0.22
    k_B    = 0.12
    up_step_max   = 6
    down_step_max = 4

    # --- Volatility detection
    if len(inc) >= 3:
        avg3 = sum(inc[-3:]) / 3
        vol = sum(abs(x - avg3) for x in inc[-3:]) / (avg3 + 1e-6)
    else:
        vol = 0.0

    # --- Demand trend detection
    if len(inc) >= 6:
        last3 = sum(inc[-3:]) / 3
        prev3 = sum(inc[-6:-3]) / 3
        spike_up = last3 > prev3 * 1.20
        drop_dn  = last3 < prev3 * 0.85
    else:
        spike_up = drop_dn = False

    # --- Adaptive EMA (faster response under low volatility)
    alpha = clamp(0.25 + 0.3 * (1.0 - clamp(vol, 0.0, 1.0)), 0.20, 0.60)
    D_hat = ema(inc, alpha)

    # --- Demand momentum predictor
    if len(inc) >= 3:
        trend = (inc[-1] - inc[-3]) / max(inc[-3], 1)
    else:
        trend = 0.0
    D_hat *= (1 + 0.25 * trend)

    # --- Estimate the pipeline (sum of last L orders)
    pipe_est = 0
    for lag in range(1, L + 1):
        idx = len(weeks) - 1 - lag
        if idx >= 0:
            pipe_est += int(weeks[idx]["orders"][role])

    # --- Targets
    S_pipe   = L * D_hat
    safety_r = b_base * (1.0 + 0.5 * clamp(vol, 0.0, 1.0))
    S_invpos = max(0.0, safety_r * D_hat)

    # --- Event-based adaptation
    if spike_up:
        k_P *= 1.25
        k_B *= 1.25
        S_pipe *= 1.15
        up_step_max = 7
    if drop_dn:
        S_invpos *= 0.80
        S_pipe   *= 0.85
        k_B      *= 0.70
        down_step_max = 5

    # --- Dynamic backlog adaptation
    if back > D_hat * 1.2:
        k_B *= 1.5
        k_P *= 0.9
    elif back < D_hat * 0.5:
        k_B *= 0.8

    # --- Inventory position
    inv_pos = inv - back + pipe_est

    # --- Core APIOBPCS controller
    order_raw = (
        D_hat
        + k_I * (S_invpos - inv_pos)
        + k_P * (S_pipe   - pipe_est)
        + k_B * back
    )

    # --- Anti-oversupply protection
    if inv_pos > (S_invpos + 0.30 * S_pipe) * 1.5:
        order_raw *= 0.75

    # --- Step limits
    delta = order_raw - prev
    if   delta >  up_step_max:   order_limited = prev + up_step_max
    elif delta < -down_step_max: order_limited = prev - down_step_max
    else:                        order_limited = order_raw

    # --- Momentum damping (smooth transition)
    smooth_factor = 0.35
    order_final = (1 - smooth_factor) * order_limited + smooth_factor * prev

    # --- Backlog safeguard
    if back > 0 and order_final < D_hat:
        order_final = D_hat

    # Return non-negative integer order quantity
    return int(round(order_final)) if order_final > 0 else 0

# --------------------- Mode Wrappers ---------------------
def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Independent control for each role (BlackBox mode)."""
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    return {r: decide_beerbot(weeks, r) for r in roles}

def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Shared logic for all roles (GlassBox mode)."""
    return decide_blackbox(weeks)

# --------------------- API Endpoint ---------------------
@app.post("/api/decision")
async def decision(req: Request):
    """Main BeerBot API handler for both handshake and weekly decision requests."""
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

    # --- Weekly decision step ---
    weeks = body.get("weeks", [])
    mode = body.get("mode", "blackbox")

    if not weeks:
        return JSONResponse({"orders": {r: 10 for r in ["retailer", "wholesaler", "distributor", "factory"]}})

    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)
    return JSONResponse({"orders": orders})
