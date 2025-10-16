from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "NaiveBeerBot_OrderUpTo"
VERSION = "v1.2"

ROLES = ["retailer", "wholesaler", "distributor", "factory"]
DEFAULT_INV_COST = 1.0
DEFAULT_BACKLOG_COST = 2.0

# --- helpers ---
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ema(series: List[float], alpha: float = 0.4, default: float = 10.0) -> float:
    if not series:
        return default
    v = series[0]
    for x in series[1:]:
        v = alpha * x + (1 - alpha) * v
    return v

def last_series(weeks: List[Dict[str, Any]], role: str, key: str) -> List[float]:
    out = []
    for w in weeks:
        st = w.get("roles", {}).get(role, {})
        out.append(float(st.get(key, 0)))
    return out

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

# --- core policy: order-up-to with EMA forecast ---
def smart_order(role: str, weeks: List[Dict[str, Any]]) -> int:
    # Параметры (можно тюнить)
    L = 3.0          # оценка совокупного lead time в неделях
    ALPHA = 0.45     # EMA для прогноза спроса
    BETA = 0.35      # доля safety stock от прогноза
    MIN_O, MAX_O = 4, 28
    DAMP = 0.25      # мягкое демпфирование (ограничивает шаг)

    if not weeks:
        return 10

    inv_hist = last_series(weeks, role, "inventory")
    back_hist = last_series(weeks, role, "backlog")
    inord_hist = last_series(weeks, role, "incoming_orders")

    inv = inv_hist[-1]
    back = back_hist[-1]

    # Прогноз спроса на неделю вперёд
    forecast = ema(inord_hist[-6:], alpha=ALPHA, default=10.0)

    # Целевая позиция и текущая позиция
    safety = BETA * forecast
    target_pos = L * forecast + safety
    inv_pos = inv - back  # approx без pipeline

    # Сырая рекомендация заказа
    raw = forecast + (target_pos - inv_pos) / max(1.0, L)

    # Демпфирование: не даём прыгать > DAMP * forecast за раз
    # В качестве предыдущего заказа берём последнюю входящую как прокси,
    # чтобы не зависеть от наличия истории наших заказов.
    prev_proxy = inord_hist[-1]
    max_step = DAMP * max(6.0, forecast)
    ordered = prev_proxy + clamp(raw - prev_proxy, -max_step, max_step)

    return int(clamp(ordered, MIN_O, MAX_O))

def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return {r: smart_order(r, weeks) for r in ROLES}

def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    return decide_blackbox(weeks)

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
            "message": "OrderUpTo EMA policy ready"
        })

    weeks = body.get("weeks", [])
    if not weeks:
        return JSONResponse({"orders": {r: 10 for r in ROLES}})

    mode = body.get("mode", "blackbox").lower()
    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)

    costs = body.get("costs", {}) or {}
    inv_c = float(costs.get("inventory_per_unit", DEFAULT_INV_COST))
    back_c = float(costs.get("backlog_per_unit", DEFAULT_BACKLOG_COST))
    metrics = compute_metrics(weeks, inv_cost=inv_c, backlog_cost=back_c)

    return JSONResponse({"orders": orders, "metrics": metrics})
