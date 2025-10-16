# beerbot_bptk_style_v3.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WEEKS = 36
ROLES = ["retailer", "wholesaler", "distributor", "factory"]
ALPHA, BETA = 0.7, 0.5
S_target, SL_target = 18, 38
LEAD = 5

HOLD_COST, BACK_COST = 1.0, 2.0

def demand_pattern(name="flat"):
    if name == "flat": return [20]*WEEKS
    if name == "spike": return [20]*10 + [40]*10 + [10]*16
    if name == "recovery": return [10]*8 + [40]*8 + [15]*20
    raise ValueError

def run_beergame(demand):
    state = {r: dict(I=20., B=0., SL=0.) for r in ROLES}
    orders_hist = {r: [10.] for r in ROLES}
    inv_hist, back_hist = {r: [] for r in ROLES}, {r: [] for r in ROLES}
    inv_cost = back_cost = 0.

    for t in range(WEEKS):
        D = {"retailer": demand[t]}
        D["wholesaler"]  = orders_hist["retailer"][-1]
        D["distributor"] = orders_hist["wholesaler"][-1]
        D["factory"]     = orders_hist["distributor"][-1]

        # Arrivals (supply line flow)
        for r in ROLES:
            arrive = state[r]["SL"]/LEAD
            state[r]["I"] += arrive
            state[r]["SL"] -= arrive

        # Fulfill orders
        for r in ROLES:
            need = D[r] + state[r]["B"]
            ship = min(need, state[r]["I"])
            state[r]["I"] -= ship
            state[r]["B"] = need - ship

        # Decision (Sterman rule)
        for r in ROLES:
            I, SL, B = state[r]["I"], state[r]["SL"], state[r]["B"]
            order = max(D[r] + ALPHA*(S_target - I) + BETA*(SL_target - SL), 0)
            state[r]["SL"] += order
            orders_hist[r].append(order)

        # Logging
        for r in ROLES:
            inv_hist[r].append(state[r]["I"])
            back_hist[r].append(state[r]["B"])

        inv_cost += sum(state[r]["I"] for r in ROLES)
        back_cost += sum(state[r]["B"] for r in ROLES)

    total = HOLD_COST*inv_cost + BACK_COST*back_cost
    return total, inv_cost, back_cost, orders_hist, inv_hist, back_hist

# -------------------- MAIN --------------------
if __name__=="__main__":
    demand = demand_pattern("flat")
    total, inv, back, orders, inv_hist, back_hist = run_beergame(demand)

    peak_inventory = max(sum(inv_hist[r][t] for r in ROLES) for t in range(WEEKS))
    peak_backlog   = max(sum(back_hist[r][t] for r in ROLES) for t in range(WEEKS))

    print(f"Inventory cost=${inv:.0f}")
    print(f"Backlog cost=${back:.0f}")
    print(f"Total cost=${total:.0f}")
    print(f"Peak Inventory={peak_inventory:.0f}")
    print(f"Peak Backlog={peak_backlog:.0f}")

    # ---- Plot 1: Order fluctuations ----
    plt.figure(figsize=(10,5))
    for r, color in zip(ROLES, ['tab:purple','tab:green','tab:orange','tab:red']):
        plt.plot(orders[r], label=f"{r.title()} Orders", color=color)
    plt.title("The Bullwhip Effect: Order Fluctuations")
    plt.xlabel("Week"); plt.ylabel("Orders"); plt.grid(True)
    plt.legend(); plt.tight_layout(); plt.show()

    # ---- Plot 2: Supply Chain Inventory Levels ----
    plt.figure(figsize=(10,5))
    total_inv = np.array([sum(inv_hist[r][t] for r in ROLES) for t in range(WEEKS)])
    total_back = np.array([sum(back_hist[r][t] for r in ROLES) for t in range(WEEKS)])
    plt.plot(total_inv, label="Total Inventory", color="tab:blue")
    plt.plot(total_back, label="Total Backlog", color="tab:red")
    plt.title("Supply Chain Inventory Levels")
    plt.xlabel("Week"); plt.ylabel("Units"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()
