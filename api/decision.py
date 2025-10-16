from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import math

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Adaptive_Harmony"
VERSION = "v8.3"   # обновил версию, чтобы отличать сборки

# --------------------- Helpers ---------------------
def ema(series: List[float], alpha: float = 0.2) -> float:
    if not series:
        return 0.0
    v = float(series[0])
    for x in series[1:]:
        v = alpha * float(x) + (1 - alpha) * v
    return v

def ewvar(series: List[float], alpha: float = 0.25) -> float:
    if not series:
        return 0.0
    m = ema(series, alpha)
    v = 0.0
    for x in series:
        v = alpha * (float(x) - m) ** 2 + (1 - alpha) * v
    return max(v, 0.0)

def get_state(week: Dict[str, Any], role: str) -> Dict[str, int]:
    return week["roles"][role]

def prev_order(weeks: List[Dict[str, Any]], role: str) -> int:
    # если есть предыдущая неделя — берём её заказ, иначе 10
    if len(weeks) >= 2:
        return int(weeks[-2]["orders"][role])
    return 10

def soft_step(prev: float, target: float, ratio: float = 0.15) -> float:
    """мягко ограничивает изменение относительно прошлого заказа"""
    diff = target - prev
    limit = max(1.0, abs(prev) * ratio)  # фикс: при prev=0 не залипаем
    if diff >  limit: target = prev + limit
    if diff < -limit: target = prev - limit
    return target

# --------------------- Policy params ---------------------
COVER_WEEKS = {
    "retailer": 1.2,
    "wholesaler": 1.6,
    "distributor": 1.9,
    "factory": 2.2,
}
ROLE_GAIN = {"retailer": 1.05, "wholesaler": 0.9, "distributor": 0.8, "factory": 0.7}
FEED_FORWARD = {"factory": 0.12, "distributor": 0.07, "wholesaler": 0.04}

# ---------- Core ----------
def decide_one_role(weeks: List[Dict[str, Any]], role: str, glassbox: bool) -> int:
    last = weeks[-1]["roles"][role]
    inv  = float(last["inventory"])
    back = float(last["backlog"])

    incoming = [float(get_state(w, role)["incoming_orders"]) for w in weeks] or [10.0]

    # гибридный прогноз + тренд
    D_fast = ema(incoming, 0.35)
    D_slow = ema(incoming, 0.15)
    trend  = 0.35 * (incoming[-1] - incoming[-2]) if len(incoming) >= 2 else 0.0
    D_hat  = max(0.0, 0.6 * D_fast + 0.4 * D_slow + trend)

    # волатильность и фаза игры
    std_d  = math.sqrt(ewvar(incoming, 0.3))
    week_i = weeks[-1]["week"]
    total  = max(weeks[-1].get("weeks_total", 36), 36)
    phase  = max(0.0, min(1.0, week_i / total))

    # целевая позиция = покрытие спроса + safety stock (ослабляем к концу)
    cover = COVER_WEEKS.get(role, 1.4)
    z = 1.25 - 0.55 * phase
    safety_stock = z * std_d * math.sqrt(max(cover, 0.1))
    target_pos = cover * D_hat + safety_stock

    # текущая позиция (инвентарь - бэклог)
    pos = inv - back

    # подтягиваем позицию к цели
    g = ROLE_GAIN.get(role, 0.85)
    target_order = g * (target_pos - pos)

    # cost-aware: агрессивно гасим бэклог, мягче штрафуем склад
    target_order += 0.50 * back
    target_order -= 0.02 * inv

    # feed-forward от розницы (только в glassbox, не для самого ритейла)
    if glassbox and role != "retailer" and "retailer" in weeks[-1]["roles"]:
        r_incoming = float(get_state(weeks[-1], "retailer")["incoming_orders"])
        ff = FEED_FORWARD.get(role, 0.0)
        target_order = (1 - ff) * target_order + ff * r_incoming

    # поздняя фаза — чуть ближе к чистому спросу
    target_order = (1 - 0.35 * phase) * target_order + 0.35 * phase * D_hat

    # адаптивный лимит шага: если стресс (большой бэклог/дефицит), разрешаем быстрее расти
    prev = float(prev_order(weeks, role))
    stress = (back > 0.8 * max(1.0, D_hat)) or (pos < 0)
    step_ratio = 0.35 if stress else 0.15
    order = soft_step(prev, target_order, ratio=step_ratio)

    # нижний «service floor», чтобы не голодать
    service_floor = 0.7 * D_hat + 0.1 * back
    order = max(order, service_floor)

    # слабее якорь к прошлому — быстрее догоняем спрос
    order = 0.10 * prev + 0.90 * order

    return max(0, int(round(order)))

def decide_beerbot(weeks: List[Dict[str, Any]], role: str, glassbox: bool) -> int:
    return decide_one_role(weeks, role, glassbox)

def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    return {r: decide_one_role(weeks, r, glassbox=False) for r in roles}

def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    roles = ["retailer", "wholesaler", "distributor", "factory"]
    return {r: decide_one_role(weeks, r, glassbox=True) for r in roles}

# ---------- API ----------
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
        default = {r: 10 for r in ["retailer","wholesaler","distributor","factory"]}
        return JSONResponse({"orders": default})

    mode = body.get("mode", "blackbox").lower()
    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)
    return JSONResponse({"orders": orders})
