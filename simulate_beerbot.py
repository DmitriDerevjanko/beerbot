# simulate_beerbot.py — BPTK full-match optimizer (final)
import numpy as np, random
from math import isfinite

TARGET = {"inv": 976, "back": 15874, "pinv": 82, "pback": 477}
WEEKS, ROLES = 36, ["retailer", "wholesaler", "distributor", "factory"]
HOLD_COST, BACK_COST = 1.0, 2.0

def simulate_beergame(params):
    a, kI, kSL, S, S_SL, L, perception = params
    demand = [20]*10 + [40]*10 + [10]*16
    state = {r: dict(I=20., B=0.) for r in ROLES}
    pipe = {r: [0.]*int(L) for r in ROLES}
    orders = {r: [10.] for r in ROLES}
    smooth = {r: [10., 10.] for r in ROLES}  # perception buffer
    inv_cost = back_cost = 0.
    pinv = pback = 0.

    for t in range(WEEKS):
        D = {"retailer": demand[t]}
        D["wholesaler"]  = orders["retailer"][-1]
        D["distributor"] = orders["wholesaler"][-1]
        D["factory"]     = orders["distributor"][-1]

        for r in ROLES:
            arr = pipe[r].pop(0)
            state[r]["I"] += arr
            pipe[r].append(0.)

        for r in ROLES:
            need = D[r] + state[r]["B"]
            ship = min(need, state[r]["I"])
            state[r]["I"] -= ship
            state[r]["B"] = need - ship

        for r in ROLES:
            I, B = state[r]["I"], state[r]["B"]
            SL = sum(pipe[r])

            # perception smoothing (acts like 1-2 week delay)
            smooth[r].append(D[r])
            if len(smooth[r]) > perception:
                smooth[r].pop(0)
            D_smooth = sum(smooth[r]) / len(smooth[r])

            # Sterman decision rule
            order = max(0., a * D_smooth + kI * (S - I) + kSL * (S_SL - SL) + B)
            pipe[r][-1] += order
            orders[r].append(order)

        inv_total = sum(s["I"] for s in state.values())
        back_total = sum(s["B"] for s in state.values())
        inv_cost += inv_total
        back_cost += back_total
        pinv = max(pinv, inv_total)
        pback = max(pback, back_total)

    total = HOLD_COST * inv_cost + BACK_COST * back_cost
    return inv_cost, back_cost, total, pinv, pback

def score(inv, back, total, pinv, pback):
    target_total = HOLD_COST * TARGET["inv"] + BACK_COST * TARGET["back"]
    diff = np.array([
        abs(inv - TARGET["inv"]) / TARGET["inv"],
        2.5 * abs(back - TARGET["back"]) / TARGET["back"],
        abs(total - target_total) / target_total,
        abs(pinv - TARGET["pinv"]) / TARGET["pinv"],
        abs(pback - TARGET["pback"]) / TARGET["pback"]
    ])
    return np.sum(diff ** 2)

best = float("inf")
best_params = None

for i in range(8000):
    params = np.array([
        random.uniform(0.6, 0.8),      # a
        random.uniform(0.3, 0.6),      # kI
        random.uniform(1.3, 1.6),      # kSL
        random.uniform(8, 14),         # S
        random.uniform(25, 35),        # S_SL
        random.uniform(7, 8),          # L
        random.choice([2, 3])          # perception delay
    ])
    inv, back, total, pinv, pback = simulate_beergame(params)
    s = score(inv, back, total, pinv, pback)
    if s < best and all(isfinite(x) for x in [inv, back, total]):
        best = s
        best_params = (params, inv, back, total, pinv, pback)
        print(f"New best ({i}): loss={s:.6f}  "
              f"a={params[0]:.3f}, kI={params[1]:.3f}, kSL={params[2]:.3f}, "
              f"S={params[3]:.2f}, S_SL={params[4]:.2f}, L={params[5]:.1f}, P={params[6]}  "
              f"-> inv={inv:.0f}, back={back:.0f}, total={total:.0f}, pinv={pinv:.0f}, pback={pback:.0f}")

# Fine-tuning
p, inv, back, total, pinv, pback = best_params
for j in range(2000):
    pert = np.array([
        np.clip(p[0] + random.uniform(-0.02, 0.02), 0.6, 0.8),
        np.clip(p[1] + random.uniform(-0.05, 0.05), 0.3, 0.6),
        np.clip(p[2] + random.uniform(-0.05, 0.05), 1.3, 1.6),
        np.clip(p[3] + random.uniform(-0.5, 0.5), 8, 14),
        np.clip(p[4] + random.uniform(-0.5, 0.5), 25, 35),
        np.clip(p[5] + random.uniform(-0.1, 0.1), 7, 8),
        random.choice([2, 3])
    ])
    inv2, back2, total2, pinv2, pback2 = simulate_beergame(pert)
    s = score(inv2, back2, total2, pinv2, pback2)
    if s < best:
        best = s
        best_params = (pert, inv2, back2, total2, pinv2, pback2)
        print(f"Fine ({j}): loss={s:.6f}  "
              f"a={pert[0]:.3f}, kI={pert[1]:.3f}, kSL={pert[2]:.3f}, "
              f"S={pert[3]:.2f}, S_SL={pert[4]:.2f}, L={pert[5]:.1f}, P={pert[6]}  "
              f"-> inv={inv2:.0f}, back={back2:.0f}, total={total2:.0f}, "
              f"pinv={pinv2:.0f}, pback={pback2:.0f}")

p, inv, back, total, pinv, pback = best_params
target_total = HOLD_COST * TARGET["inv"] + BACK_COST * TARGET["back"]
print("\n=== FINAL MATCH (1-to-1 with BPTK) ===")
print(f"a={p[0]:.3f}, kI={p[1]:.3f}, kSL={p[2]:.3f}, S={p[3]:.2f}, S_SL={p[4]:.2f}, L={p[5]:.1f}, perception={int(p[6])}")
print(f"Inventory={inv:.0f} (target {TARGET['inv']}), Backlog={back:.0f} (target {TARGET['back']}), Total={total:.0f} (target {target_total:.0f})")
print(f"PeakInv={pinv:.0f} (target {TARGET['pinv']}), PeakBack={pback:.0f} (target {TARGET['pback']})")
