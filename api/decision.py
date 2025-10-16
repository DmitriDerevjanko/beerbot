from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot"
VERSION = "main"

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

# --------------------- APIOBPCS+ Core Logic ---------------------
def decide_beerbot(weeks: List[Dict[str, Any]], role: str) -> int:
    """
    APIOBPCS+ (Adaptive Pipeline & Inventory Order-Based Production Control System)

    Formula:
        order = D_hat
                + k_I * (S_invpos - inv_position)
                + k_P * (S_pipe   - pipe_est)
                + k_B * backlog

    where:
        D_hat        — exponentially smoothed demand (EMA)
        inv_position — inventory position = inventory - backlog + pipeline_est
        pipe_est     — pipeline estimate (sum of last L orders)
        S_invpos     — target inventory position
        S_pipe       — target pipeline (≈ L * D_hat)
        k_I, k_P, k_B — control coefficients (adaptive)
    """
    last = weeks[-1]["roles"][role]
    inv  = int(last["inventory"])
    back = int(last["backlog"])
    inc  = incoming_series(weeks, role)
    prev = prev_order(weeks, role)

    # --- Typical logistics delay (2 weeks in the MIT Beer Game)
    L = 2

    # --- Base parameters (adapt dynamically later)
    b_base = 0.80          # Base safety factor (fraction of demand)
    k_I    = 0.40          # Inventory correction
    k_P    = 0.22          # Pipeline correction
    k_B    = 0.12          # Backlog correction (soft)
    up_step_max   = 6      # Max upward step per week
    down_step_max = 4      # Max downward step per week

    # --- Volatility detection (based on last 3 demand points)
    if len(inc) >= 3:
        avg3 = sum(inc[-3:]) / 3
        vol = sum(abs(x - avg3) for x in inc[-3:]) / (avg3 + 1e-6)
    else:
        vol = 0.0

    # --- Event detection: demand spikes or drops
    spike_up = False
    drop_dn  = False
    if len(inc) >= 6:
        last3 = sum(inc[-3:]) / 3
        prev3 = sum(inc[-6:-3]) / 3
        if prev3 > 0:
            if last3 > prev3 * 1.20:   # demand increased >20% in 3 weeks
                spike_up = True
            if last3 < prev3 * 0.85:   # demand dropped >15% in 3 weeks
                drop_dn = True

    # --- Adaptive EMA factor: lower volatility = faster response
    alpha = clamp(0.25 + 0.25 * (1.0 - clamp(vol, 0.0, 1.0)), 0.20, 0.50)
    D_hat = ema(inc, alpha)

    # --- Estimate the pipeline (sum of last L orders)
    pipe_est = 0
    for lag in range(1, L + 1):
        idx = len(weeks) - 1 - lag
        if idx >= 0:
            pipe_est += int(weeks[idx]["orders"][role])

    # --- Targets
    S_pipe   = L * D_hat
    safety_r = b_base * (1.0 + 0.5 * clamp(vol, 0.0, 1.0))  # higher volatility → larger safety
    S_invpos = max(0.0, safety_r * D_hat)

    # --- Event-based adaptation
    if spike_up:
        # Aggressively push pipeline and backlog to avoid shortage
        k_P *= 1.25
        k_B *= 1.25
        S_pipe *= 1.15
        up_step_max = 7
    if drop_dn:
        # Faster de-stocking when demand drops
        S_invpos *= 0.80
        S_pipe   *= 0.85
        k_B      *= 0.70
        down_step_max = 5

    # --- Inventory position
    inv_pos = inv - back + pipe_est

    # --- Core APIOBPCS controller
    order_raw = (
        D_hat
        + k_I * (S_invpos - inv_pos)
        + k_P * (S_pipe   - pipe_est)
        + k_B * back
    )

    # --- Anti-oversupply protection (if stock + pipeline far above target)
    if inv_pos > (S_invpos + 0.30 * S_pipe) * 1.5:
        order_raw *= 0.75

    # --- Step limits (anti-oscillation smoothing)
    delta = order_raw - prev
    if   delta >  up_step_max:   order_limited = prev + up_step_max
    elif delta < -down_step_max: order_limited = prev - down_step_max
    else:                        order_limited = order_raw

    # --- Backlog safeguard: never go below demand when backlog exists
    if back > 0 and order_limited < D_hat:
        order_limited = D_hat

    # Return non-negative integer order quantity
    return int(round(order_limited)) if order_limited > 0 else 0

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