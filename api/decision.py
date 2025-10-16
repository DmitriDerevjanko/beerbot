from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "NaiveBeerBot_Improved"
VERSION = "v1.1"

ROLES = ["retailer", "wholesaler", "distributor", "factory"]
DEFAULT_INV_COST = 1.0
DEFAULT_BACKLOG_COST = 2.0


def clamp(x: float, lo: float, hi: float) -> float:
    """Ограничивает значение между нижним и верхним порогом"""
    return max(lo, min(hi, x))


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


def smart_order(role: str, weeks: List[Dict[str, Any]]) -> int:
    """Улучшённая логика заказа"""
    if len(weeks) < 1:
        return 10

    last = weeks[-1]["roles"][role]
    incoming = last["incoming_orders"]
    backlog = last["backlog"]
    inventory = last["inventory"]

    # Базовый заказ — как у наивного
    base = incoming

    # Добавляем компенсацию backlog
    compensation = 0.4 * backlog  # можно отрегулировать

    # Амортизируем (сглаживаем)
    smooth = (inventory * 0.1)

    # Итог
    order = base + compensation - smooth

    # Ограничиваем
    return int(clamp(order, 4, 30))


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
            "message": "NaiveBeerBot_Improved ready"
        })

    weeks = body.get("weeks", [])
    if not weeks:
        default = {r: 10 for r in ROLES}
        return JSONResponse({"orders": default})

    mode = body.get("mode", "blackbox").lower()
    orders = decide_glassbox(weeks) if mode == "glassbox" else decide_blackbox(weeks)

    costs = body.get("costs", {}) or {}
    inv_c = float(costs.get("inventory_per_unit", DEFAULT_INV_COST))
    back_c = float(costs.get("backlog_per_unit", DEFAULT_BACKLOG_COST))
    metrics = compute_metrics(weeks, inv_cost=inv_c, backlog_cost=back_c)

    return JSONResponse({"orders": orders, "metrics": metrics})
