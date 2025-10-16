from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import math

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Adaptive_Harmony"
VERSION = "v8.6"

# ===== Helpers =====
def ema(series: List[float], alpha: float = 0.2) -> float:
    if not series: return 0.0
    v = float(series[0])
    for x in series[1:]:
        v = alpha * float(x) + (1 - alpha) * v
    return v

def ewvar(series: List[float], alpha: float = 0.25) -> float:
    if not series: return 0.0
    m = ema(series, alpha)
    v = 0.0
    for x in series:
        v = alpha * (float(x) - m) ** 2 + (1 - alpha) * v
    return max(v, 0.0)

def get_state(week: Dict[str, Any], role: str) -> Dict[str, int]:
    return week["roles"][role]

def prev_order(weeks: List[Dict[str, Any]], role: str) -> int:
    if len(weeks) >= 2:
        return int(weeks[-2]["orders"][role])
    return 10

def soft_step(prev: float, target: float, ratio: float) -> float:
    diff = target - prev
    limit = max(1.0, abs(prev) * ratio)  # чтобы при prev=0 не залипать
    if diff >  limit: target = prev + limit
    if diff < -limit: target = prev - limit
    return target

# ===== Policy params =====
COVER_WEEKS = {
    "retailer": 1.0,
    "wholesaler": 1.3,
    "distributor": 1.5,
    "factory":   1.7,
}
ROLE_GAIN = {"retailer": 1.00, "wholesaler": 0.85, "distributor": 0.75, "factory": 0.75}
FEED_FORWARD = {"factory": 0.18, "distributor": 0.10, "wholesaler": 0.06}

ROLES = ["retailer", "wholesaler", "distributor", "factory"]

# ===== Costs (можно переопределить в запросе) =====
DEFAULT_INV_COST = 1.0
DEFAULT_BACKLOG_COST = 2.0

def compute_metrics(weeks: List[Dict[str, Any]], inv_cost=DEFAULT_INV_COST, backlog_cost=DEFAULT_BACKLOG_COST):
    inv_sum = 0.0
    back_sum = 0.0
    peak_inv = 0
    peak_back = 0

    for w in weeks:
        for r in ROLES:
            st = w["roles"].get(r, {})
            inv = int(st.get("inventory", 0))
            back = int(st.get("backlog", 0))
            inv_sum += inv
            back_sum += back
            peak_inv = max(peak_inv, inv)
            peak_back = max(peak_back, back)

    inv_cost_total = inv_sum * inv_cost
    back_cost_total = back_sum * backlog_cost
    total_cost = inv_cost_total + back_cost_total

    return {
        "inventory_cost": int(round(inv_cost_total)),
        "backlog_cost": int(round(back_cost_total)),
        "total_cost": int(round(total_cost)),
        "peak_inventory": int(peak_inv),
        "peak_backlog": int(peak_back),
    }

# ===== Core decision =====
def decide_one_role(weeks: List[Dict[str, Any]], role: str, glassbox: bool) -> int:
    last = weeks[-1]["roles"][role]
    inv  = float(last["inventory"])
    back = float(last["backlog"])
    incoming = [float(get_state(w, role)["incoming_orders"]) for w in weeks] or [10.0]

    # Умеренный прогноз
    D_fast = ema(incoming, 0.30)
    D_slow = ema(incoming, 0.15)
    trend  = 0.20 * (incoming[-1] - incoming[-2]) if len(incoming) >= 2 else 0.0
    D_hat  = max(0.0, 0.6 * D_fast + 0.4 * D_slow + trend)

    std_d  = math.sqrt(ewvar(incoming, 0.28))
    week_i = weeks[-1]["week"]
    total  = max(weeks[-1].get("weeks_total", 36), 36)
    phase  = max(0.0, min(1.0, week_i / total))

    cover = COVER_WEEKS.get(role, 1.3)
    z = 1.10 - 0.50 * phase
    safety_stock = z * std_d * math.sqrt(max(cover, 0.1))
    target_pos = cover * D_hat + safety_stock

    pos = inv - back
    g = ROLE_GAIN.get(role, 0.8)
    target_order = g * (target_pos - pos)

    # Cost-aware
    if role == "factory":
        target_order += 0.35 * back
    else:
        target_order += 0.25 * back
    target_order -= 0.06 * inv

    # Feed-forward
    if glassbox and role != "retailer" and "retailer" in weeks[-1]["roles"]:
        r_incoming = float(get_state(weeks[-1], "retailer")["incoming_orders"])
        ff = FEED_FORWARD.get(role, 0.0)
        target_order = (1 - ff) * target_order + ff * r_incoming

    # Поздняя фаза ближе к спросу
    target_order = (1 - 0.30 * phase) * target_order + 0.30 * phase * D_hat

    prev = float(prev_order(weeks, role))
    stress = (back > 0.8 * max(1.0, D_hat)) or (pos < 0)
    step_ratio = (0.30 if role == "factory" else 0.25) if stress else 0.12
    order = soft_step(prev, target_order, ratio=step_ratio)

    # Service floor
    service_floor = 0.65 * D_hat + 0.08 * back
    order = max(order, service_floor)

    # Adaptive cap
    cap = 1.8 * D_hat + 0.30 * back
    order = min(order, cap)

    # Сглаживание
    order = 0.30 * prev + 0.70 * order

    return max(0, int(round(order)))

def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return {r: decide_one_role(weeks, r, glassbox=False) for r in ROLES}

def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return {r: decide_one_role(weeks, r, glassbox=True) for r in ROLES}

# ===== API =====
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
        default = {r: 10 for r in ROLES}
        return JSONResponse({"orders": default})

    mode = body.get("mode", "blackbox").lower()
    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)

    # Метрики (можно переопределить ставки в body.costs)
    costs = body.get("costs", {}) or {}
    inv_c = float(costs.get("inventory_per_unit", DEFAULT_INV_COST))
    back_c = float(costs.get("backlog_per_unit", DEFAULT_BACKLOG_COST))
    metrics = compute_metrics(weeks, inv_cost=inv_c, backlog_cost=back_c)

    return JSONResponse({"orders": orders, "metrics": metrics})
