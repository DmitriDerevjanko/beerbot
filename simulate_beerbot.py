# simulate_beerbot.py
# Deterministic, spec-accurate Beer Game simulator for local evaluation

import random
import math
from copy import deepcopy
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from rich.console import Console

console = Console()

# -------------------- Try to import your BeerBot --------------------
# We support both layouts:
#  - app.py with decide_* functions
#  - api/decision.py with decide_blackbox/decide_glassbox (Vercel layout)

def _load_decider():
    try:
        from api.decision import decide_blackbox, decide_glassbox
        return decide_blackbox, decide_glassbox
    except Exception as e:
        console.print(f"[red]Failed to import api.decision: {e}[/red]")
        raise


DECIDE_BLACKBOX, DECIDE_GLASSBOX = _load_decider()

# -------------------- Global simulation params --------------------
WEEKS = 36
ROLES = ["retailer", "wholesaler", "distributor", "factory"]

# Each link has L=2 weeks shipping delay (retailer<-wholesaler<-distributor<-factory)
LEAD = 2

HOLD_COST = 1.0   # inventory holding cost per unit per week
BACK_COST = 2.0   # backlog/shortage cost per unit per week

# -------------------- Demand scenarios (deterministic) --------------------
def generate_patterns(seed: int = 2025) -> Dict[str, List[int]]:
    random.seed(seed)
    flat   = [20] * WEEKS
    spike  = [20 if w < 10 else 40 if 10 <= w < 20 else 10 for w in range(WEEKS)]
    trend  = [15 + int(0.6 * w) for w in range(WEEKS)]
    shock  = [max(5, int(20 + random.gauss(0, 5))) for _ in range(WEEKS)]
    # spike followed by partial recovery
    recovery = [10] * 8 + [40] * 8 + [15] * 20
    return {
        "flat": flat,
        "spike": spike,
        "trend": trend,
        "random_shock": shock,
        "recovery": recovery,
    }

# -------------------- Low-level supply-chain engine --------------------
def run_world(
    demand: List[int],
    mode: str = "blackbox",
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Accurate MIT-style weekly loop:
      - Create a 'weeks' array from week 1 upward
      - For each week:
        1) external demand hits retailer
        2) shipments arrive (lead=2 per link)
        3) each role fulfills incoming_orders + backlog
        4) BeerBot GETS full history 'weeks' and RETURNS orders for next week
        5) push orders into pipelines (to arrive after lead)
    We also record 'arriving_shipments' and 'incoming_orders' correctly for every role.
    """

    # per-role state
    state = {
        r: {
            "inventory": 20,
            "backlog": 0,
            "incoming_orders": 0,
            "arriving_shipments": 0,
        }
        for r in ROLES
    }

    # pipelines: for each role, list of length=LEAD with future shipments
    # shipments sent from upstream arrive to current role
    pipeline = {r: [0] * LEAD for r in ROLES}

    # current orders placed last week (start with 10)
    orders = {r: 10 for r in ROLES}

    # history to build "weeks" array
    history_weeks: List[Dict] = []
    rows = []
    total_inv_cost = 0.0
    total_back_cost = 0.0

    for week in range(1, WEEKS + 1):
        # ============ 1) Demand flows downstream ============
        # External demand to retailer
        state["retailer"]["incoming_orders"] = int(demand[week - 1])
        # Internal demand is last week's orders of the downstream role
        # (the sim sends "orders" for this week, and upstream role sees them as incoming next)
        state["wholesaler"]["incoming_orders"] = int(orders["retailer"])
        state["distributor"]["incoming_orders"] = int(orders["wholesaler"])
        state["factory"]["incoming_orders"]    = int(orders["distributor"])

        # ============ 2) Shipments arrive (from upstream pipelines) ============
        for r in ROLES:
            arriving = pipeline[r].pop(0)             # front of queue arrives
            state[r]["arriving_shipments"] = int(arriving)
            state[r]["inventory"] += int(arriving)
            pipeline[r].append(0)                     # keep length = LEAD

        # ============ 3) Fulfillment (serve demand + backlog) ============
        for r in ROLES:
            demand_now = state[r]["incoming_orders"] + state[r]["backlog"]
            ship = min(demand_now, state[r]["inventory"])
            state[r]["inventory"] -= ship
            state[r]["backlog"] = demand_now - ship

        # ============ 4) Build 'weeks' array for BeerBot ============
        # The simulator always sends full history from week 1 to current week.
        # For past weeks, we store the real state & orders that occurred.
        weeks_payload = []
        for past in history_weeks:
            # shallow copy is OK (values are primitives)
            weeks_payload.append(deepcopy(past))

        # append "this week" snapshot (current state + last known orders)
        weeks_payload.append({
            "week": week,
            "roles": {
                r: {
                    "inventory": int(state[r]["inventory"]),
                    "backlog": int(state[r]["backlog"]),
                    "incoming_orders": int(state[r]["incoming_orders"]),
                    "arriving_shipments": int(state[r]["arriving_shipments"]),
                } for r in ROLES
            },
            "orders": {r: int(orders[r]) for r in ROLES},  # the sim also sends orders placed in each previous step
        })

        # ============ 5) Call your BeerBot ============
        if mode == "glassbox":
            new_orders = DECIDE_GLASSBOX(weeks_payload)
        else:
            new_orders = DECIDE_BLACKBOX(weeks_payload)

        # Validate & make non-negative ints
        for r in ROLES:
            v = int(max(0, round(new_orders.get(r, 0))))
            new_orders[r] = v

        # ============ 6) Push new_orders into upstream pipelines ============
        # Order placed by role r will arrive to the DOWNSTREAM role after LEAD
        # Pipeline queues here represent shipments *to the current role*
        # So we put "orders[r]" into pipeline of the role that r supplies:
        # factory -> distributor, distributor -> wholesaler, wholesaler -> retailer
        pipeline["retailer"][LEAD - 1]   += new_orders["wholesaler"]   # wh → ret
        pipeline["wholesaler"][LEAD - 1] += new_orders["distributor"]  # dist → wh
        pipeline["distributor"][LEAD - 1]+= new_orders["factory"]      # fac → dist
        # factory produces on its own; nothing pushes into "factory" pipeline

        # But every role places orders to its upstream immediately (to be delivered later)
        orders = new_orders

        # save current week into history_weeks (for the next iteration)
        history_weeks.append(deepcopy(weeks_payload[-1]))

        # ============ 7) Costs ============
        inv_total = sum(state[r]["inventory"] for r in ROLES)
        back_total = sum(state[r]["backlog"] for r in ROLES)
        total_inv_cost  += inv_total * HOLD_COST
        total_back_cost += back_total * BACK_COST

        # logging row
        rows.append({
            "week": week,
            "mode": mode,
            "ext_demand": demand[week - 1],
            **{f"{r}_inventory": int(state[r]["inventory"]) for r in ROLES},
            **{f"{r}_backlog": int(state[r]["backlog"]) for r in ROLES},
            **{f"{r}_order": int(orders[r]) for r in ROLES},
            "total_inventory": int(inv_total),
            "total_backlog": int(back_total),
        })

    df = pd.DataFrame(rows)
    total = total_inv_cost + total_back_cost
    return df, total_inv_cost, total_back_cost, total

# -------------------- Batch evaluation --------------------
def evaluate_all(seed: int = 2025):
    patterns = generate_patterns(seed)
    results = []

    all_frames = []
    for name, demand in patterns.items():
        for mode in ["blackbox", "glassbox"]:
            df, inv, back, total = run_world(demand, mode=mode)
            df["pattern"] = name
            all_frames.append(df)
            results.append([mode, name, int(inv), int(back), int(total)])

    summary = pd.DataFrame(results, columns=["Mode", "Pattern", "Inventory cost", "Backlog cost", "Total cost"])
    console.print("\n[bold cyan]📊 BEERBOT PERFORMANCE SUMMARY (Spec-accurate)[/bold cyan]")
    console.print(tabulate(summary, headers="keys", tablefmt="rounded_grid", showindex=False))

    avg_by_mode = summary.groupby("Mode")["Total cost"].mean().reset_index()
    console.print("\nAverages by Mode:")
    console.print(tabulate(avg_by_mode, headers="keys", tablefmt="rounded_grid", showindex=False))

    # Save full logs
    all_df = pd.concat(all_frames, ignore_index=True)
    all_df.to_csv("beerbot_log.csv", index=False)
    console.print("[dim]📁 Saved detailed logs to beerbot_log.csv[/dim]")

    # Visuals (optional)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=all_df, x="week", y="total_backlog", hue="pattern", style="mode")
    plt.title("Total Backlog by Scenario / Mode")
    plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=all_df, x="week", y="total_inventory", hue="pattern", style="mode")
    plt.title("Total Inventory by Scenario / Mode")
    plt.grid(True); plt.tight_layout(); plt.show()

    return summary

# -------------------- Handshake sanity check --------------------
def handshake_local():
    """
    Minimal handshake echo to ensure fields match the spec.
    We mimic what the central simulator expects (but locally).
    """
    from importlib import import_module
    try:
        mod = import_module("api.decision")
        handler = getattr(mod, "decision", None)  # FastAPI handler, not used here
        # Not calling the HTTP server; just print the identity we expect on /handshake
        console.print("\n[green]Handshake fields (expected on your endpoint):[/green]")
        console.print({
            "ok": True,
            "student_email": getattr(mod, "STUDENT_EMAIL", "<missing>"),
            "algorithm_name": getattr(mod, "ALGO_NAME", "<missing>"),
            "version": getattr(mod, "VERSION", "<missing>"),
            "supports": {"blackbox": True, "glassbox": True},
            "message": "BeerBot ready"
        })
    except Exception:
        # app.py case
        mod = import_module("app")
        console.print("\n[green]Handshake fields (app.py layout):[/green]")
        console.print({
            "ok": True,
            "student_email": getattr(mod, "STUDENT_EMAIL", "<missing>"),
            "algorithm_name": getattr(mod, "ALGO_NAME", "<missing>"),
            "version": getattr(mod, "VERSION", "<missing>"),
            "supports": {"blackbox": True, "glassbox": True},
            "message": "BeerBot ready"
        })

# -------------------- Main --------------------
if __name__ == "__main__":
    handshake_local()
    evaluate_all(seed=2025)
