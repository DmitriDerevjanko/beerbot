from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

app = FastAPI()

STUDENT_EMAIL = "dmdere@taltech.ee"
ALGO_NAME = "BeerBot_Adaptive_Harmony"
VERSION = "v8.0"

def ema(series: List[float], alpha: float = 0.2) -> float:
    if not series: return 0.0
    v = float(series[0])
    for x in series[1:]:
        v = alpha * float(x) + (1 - alpha) * v
    return v

def get_state(week: Dict[str, Any], role: str) -> Dict[str, int]:
    return week["roles"][role]

def prev_order(weeks: List[Dict[str, Any]], role: str) -> int:
    if len(weeks) >= 2:
        return int(weeks[-2]["orders"][role])
    return 10

def soft_step(prev: float, target: float, ratio: float = 0.15) -> float:
    diff = target - prev
    limit = abs(prev) * ratio
    if diff >  limit: target = prev + limit
    if diff < -limit: target = prev - limit
    return target

# ---------- Glass-aware logic ----------
def decide_beerbot(weeks: List[Dict[str, Any]], role: str, glassbox: bool) -> int:
    last = weeks[-1]["roles"][role]
    inv, back = int(last["inventory"]), int(last["backlog"])
    incoming = [get_state(w, role)["incoming_orders"] for w in weeks]
    D_hat = ema(incoming, 0.2)

    prev = prev_order(weeks, role)
    week = weeks[-1]["week"]
    phase = week / max(weeks[-1].get("weeks_total", 36), 36)

    # base gains by role (factory меньше дергается)
    gains = {"retailer":1.0, "wholesaler":0.8, "distributor":0.7, "factory":0.6}
    g = gains.get(role, 0.8)

    # basic target
    target = D_hat + 0.5 * (back - inv*0.2)
    target *= g

    # feed-forward (glassbox only)
    if glassbox and "retailer" in weeks[-1]["roles"]:
        low_demand = get_state(weeks[-1], "retailer")["incoming_orders"]
        if role == "factory":      target = 0.9*target + 0.1*low_demand
        elif role == "distributor": target = 0.95*target + 0.05*low_demand

    # late-phase damping
    target = (1 - 0.4*phase) * target + 0.4*phase * D_hat

    # soft limit ±15 %
    order = soft_step(prev, target, ratio=0.15)
    return max(0, int(round(order)))

def decide_blackbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    roles = ["retailer","wholesaler","distributor","factory"]
    return {r: decide_beerbot(weeks, r, glassbox=False) for r in roles}

def decide_glassbox(weeks: List[Dict[str, Any]]) -> Dict[str, int]:
    roles = ["retailer","wholesaler","distributor","factory"]
    return {r: decide_beerbot(weeks, r, glassbox=True) for r in roles}

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
        default = {r:10 for r in ["retailer","wholesaler","distributor","factory"]}
        return JSONResponse({"orders": default})

    mode = body.get("mode", "blackbox")
    orders = decide_glassbox(weeks) if mode=="glassbox" else decide_blackbox(weeks)
    return JSONResponse({"orders": orders})
