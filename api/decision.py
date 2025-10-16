from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"        
ALGO_NAME = "BeerBot_Ultra_CostAware"       
VERSION = "v7.5"                            

# --------------------- Helpers ---------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ema(series: List[float], alpha: float) -> float:
    if not series:
        return 0.0
    v = float(series[0])
    for x in series[1:]:
        v = alpha * float(x) + (1.0 - alpha) * v
    return v

def double_ema(series: List[float], alpha1: float = 0.35, alpha2: float = 0.35) -> float:
    if not series:
        return 0.0
    first = []
    v = float(series[0])
    for x in series:
        v = alpha1 * float(x) + (1.0 - alpha1) * v
        first.append(v)
    return ema(first, alpha2)

def get_state(week: Dict[str, Any], role: str) -> Dict[str, int]:
    return week["roles"][role]

def incoming_series(weeks: List[Dict[str, Any]], role: str) -> List[int]:
    return [int(get_state(w, role)["incoming_orders"]) for w in weeks]

def backlog_series(weeks: List[Dict[str, Any]], role: str) -> List[int]:
    return [int(get_state(w, role)["backlog"]) for w in weeks]

def prev_order(weeks: List[Dict[str, Any]], role: str) -> int:
    return int(weeks[-2]["orders"][role]) if len(weeks) >= 2 else 10

def linreg_slope_last(series: List[int], window: int = 5) -> float:

    n = min(len(series), window)
    if n < 3:
        return 0.0
    y = [float(v) for v in series[-n:]]
    x = list(range(n))
    sx = sum(x)
    sy = sum(y)
    sxx = sum(i * i for i in x)
    sxy = sum(x[i] * y[i] for i in range(n))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    a = (n * sxy - sx * sy) / denom   # slope
    mean_y = sy / n
    return a / (mean_y + 1e-6)

# --------------------- Decision Logic v7.5 ---------------------
def decide_beerbot(weeks: List[Dict[str, Any]], role: str) -> int:

    last = weeks[-1]["roles"][role]
    inv: int = int(last["inventory"])
    back: int = int(last["backlog"])
    inc = incoming_series(weeks, role)
    backs = backlog_series(weeks, role)
    prev = prev_order(weeks, role)
    week_now = int(weeks[-1]["week"]) if "week" in weeks[-1] else len(weeks)
    phase = clamp(week_now / max(int(weeks[0].get("weeks_total", 36)), 36), 0.0, 1.0)

    D_smooth = double_ema(inc, 0.35, 0.35)
    slope = linreg_slope_last(inc, 5)  
    if slope > 0.18:
        D_hat = D_smooth * 1.15
    elif slope < -0.12:
        D_hat = D_smooth * 0.90
    else:
        D_hat = D_smooth

    if len(inc) >= 4:
        recent = inc[-4:]
        m = sum(recent) / 4.0
        vol = (sum(abs(x - m) for x in recent) / 4.0) / (m + 1e-6)  
        vol = clamp(vol, 0.0, 1.5)
    else:
        vol = 0.0

    L = 2
    if len(backs) >= 3 and backs[-1] > backs[-2] >= backs[-3]:  # бэклог ухудшается
        L = 3

    pipe_est = 0
    for i in range(max(0, len(weeks) - L), len(weeks)):
        pipe_est += int(weeks[i]["orders"][role]) if "orders" in weeks[i] else 0

    base_kI, base_kP, base_kB = 0.25, 0.15, 0.10

    kI = base_kI * (1.0 + 0.40 * clamp(vol, 0.0, 1.0)) * (1.0 - 0.25 * phase)
    kP = base_kP * (1.0 + 0.35 * clamp(abs(slope), 0.0, 1.0)) * (1.0 - 0.20 * phase)
    kB = base_kB * (1.0 + 0.60 * (1 if back > D_hat else 0)) * (1.0 - 0.15 * phase)

    safety = 0.85 + 0.25 * clamp(vol, 0.0, 1.0)           
    target_inv = safety * D_hat
    target_pipe = L * D_hat

    inv_pos = inv - back + pipe_est

    order_raw = (
        D_hat
        + kI * (target_inv - inv_pos)
        + kP * (target_pipe - pipe_est)
        + kB * back
    )

    if len(backs) >= 3 and backs[-1] > backs[-2] > backs[-3]:
        order_raw *= 1.12

    if back == 0 and inv > 2.0 * D_hat:
        order_raw *= 0.82

    up_base = 5 if phase < 0.5 else 4
    dn_base = 4 if phase < 0.5 else 3
    step_up = int(round(up_base + 2 * clamp(vol, 0.0, 1.0)))  
    step_dn = int(round(dn_base + 1 * clamp(vol, 0.0, 1.0)))

    delta = order_raw - prev
    if   delta >  step_up: order_raw = prev + step_up
    elif delta < -step_dn: order_raw = prev - step_dn

    damping = clamp(0.30 + 0.35 * phase + 0.10 * clamp(vol, 0.0, 1.0), 0.25, 0.75)
    order_final = (1.0 - damping) * order_raw + damping * prev

    min_order = 0.85 * D_hat
    if order_final < min_order:
        order_final = min_order

    return int(round(clamp(order_final, 0.0, 999999)))

# --------------------- Mode Wrappers ---------------------
def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    return {r: decide_beerbot(weeks, r) for r in roles}

def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return decide_blackbox(weeks)

# --------------------- API Endpoint ---------------------
@app.post("/api/decision")
async def decision(req: Request):
    body = await req.json()

    # --- Handshake ---
    if body.get("handshake") is True:
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
        default_orders = {r: 10 for r in ["retailer", "wholesaler", "distributor", "factory"]}
        return JSONResponse({"orders": default_orders})

    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)
    orders = {k: max(0, int(v)) for k, v in orders.items()}
    return JSONResponse({"orders": orders})
