from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Tuple
import math

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_v8_Cooperative"
VERSION = "v8.0.0"

ROLES = ["retailer", "wholesaler", "distributor", "factory"]
ROLE_UP = {"retailer": None, "wholesaler": "retailer", "distributor": "wholesaler", "factory": "distributor"}
LEAD = 2

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ema(y: List[float], alpha: float) -> float:
    if not y: return 0.0
    v = float(y[0])
    for x in y[1:]:
        v = alpha * float(x) + (1 - alpha) * v
    return v

def holt(y: List[float], a: float = 0.45, b: float = 0.2) -> Tuple[float, float]:
    if not y: return 0.0, 0.0
    L, T = float(y[0]), 0.0
    for x in y[1:]:
        prev_L = L
        L = a * float(x) + (1 - a) * (L + T)
        T = b * (L - prev_L) + (1 - b) * T
    return L, T

def mean_std(y: List[float], w: int = 6) -> Tuple[float, float]:
    if not y: return 0.0, 0.0
    t = y[-w:] if len(y) >= w else y
    m = sum(t) / len(t)
    s = math.sqrt(sum((v - m) ** 2 for v in t) / len(t))
    return m, s

def cusum(y: List[int], k: float = 0.22) -> Tuple[bool, bool]:
    if len(y) < 6: return False, False
    l3 = sum(y[-3:]) / 3.0
    p3 = sum(y[-6:-3]) / 3.0
    if p3 <= 0: return False, False
    return l3 > p3 * (1 + k), l3 < p3 * (1 - k)

def incoming_series(weeks: List[Dict[str, Any]], role: str) -> List[int]:
    return [int(weeks[i]["roles"][role]["incoming_orders"]) for i in range(len(weeks))]

def prev_order(weeks: List[Dict[str, Any]], role: str) -> int:
    if len(weeks) >= 2: return int(weeks[-2]["orders"][role])
    return 10

def global_totals(last_roles: Dict[str, Dict[str, int]]) -> Tuple[int, int]:
    inv = sum(int(last_roles[r]["inventory"]) for r in ROLES)
    back = sum(int(last_roles[r]["backlog"]) for r in ROLES)
    return inv, back

def supply_line(weeks: List[Dict[str, Any]], role: str, L: int = LEAD) -> int:
    s = 0
    for lag in range(1, L + 1):
        idx = len(weeks) - 1 - lag
        if idx >= 0:
            s += int(weeks[idx]["orders"][role])
    return s

def choose_policy(vol: float, spike_up: bool, drop_dn: bool) -> Dict[str, float]:
    if spike_up:
        return dict(b=0.40, kI=0.60, kP=0.30, kB=0.60, up=10, down=6, coop=0.10)
    if drop_dn:
        return dict(b=0.35, kI=0.55, kP=0.25, kB=0.25, up=8,  down=7, coop=0.08)
    if vol > 0.35:
        return dict(b=0.42, kI=0.58, kP=0.28, kB=0.45, up=9,  down=6, coop=0.09)
    return dict(b=0.38, kI=0.52, kP=0.24, kB=0.35, up=8,  down=5, coop=0.07)

def decide_one(weeks: List[Dict[str, Any]], role: str, mode: str) -> int:
    last_roles = weeks[-1]["roles"]
    inv = int(last_roles[role]["inventory"])
    back = int(last_roles[role]["backlog"])
    prev = prev_order(weeks, role)

    src_role = role
    if mode == "glassbox" and ROLE_UP[role] is not None:
        src_role = ROLE_UP[role]

    inc = incoming_series(weeks, src_role)
    Ls, Ts = holt([float(x) for x in inc], 0.5, 0.22)
    D_hat = max(0.0, Ls + Ts)

    mean6, std6 = mean_std([float(x) for x in inc], 6)
    vol = clamp(std6 / (mean6 + 1e-6), 0.0, 1.5)
    spike_up, drop_dn = cusum(inc, 0.25)

    params = choose_policy(vol, spike_up, drop_dn)
    b, kI, kP, kB = params["b"], params["kI"], params["kP"], params["kB"]
    up_cap, down_cap, kCoop = params["up"], params["down"], params["coop"]

    pipe_now = supply_line(weeks, role, LEAD)
    inv_pos = inv - back + pipe_now

    S_pipe = LEAD * D_hat + 0.6 * min(back, D_hat * LEAD)
    S_invpos = b * D_hat

    g_inv, g_back = global_totals(last_roles)
    coop_term = kCoop * (g_back - 0.6 * g_inv) / max(1.0, len(ROLES))

    order_raw = (
        D_hat
        + kI * (S_invpos - inv_pos)
        + kP * (S_pipe - pipe_now)
        + kB * back
        + coop_term
    )

    if inv_pos > (S_invpos + 0.35 * S_pipe): order_raw *= 0.80
    if back > D_hat * 0.8: order_raw *= 1.12

    delta = order_raw - prev
    if   delta >  up_cap:   order = prev + up_cap
    elif delta < -down_cap: order = prev - down_cap
    else:                   order = order_raw

    if back > 0 and order < D_hat: order = D_hat
    return int(max(0, round(order)))

def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return {r: decide_one(weeks, r, mode="blackbox") for r in ROLES}

def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return {r: decide_one(weeks, r, mode="glassbox") for r in ROLES}

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
        return JSONResponse({"orders": {r: 10 for r in ROLES}})
    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)
    return JSONResponse({"orders": orders})
