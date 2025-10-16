import json, random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from rich.console import Console
from app import decide_beerbot  # import your main BeerBot decision logic
from copy import deepcopy

console = Console()

# === Simulation Parameters ===
WEEKS = 36
ROLES = ["retailer", "wholesaler", "distributor", "factory"]
HOLD_COST = 1.0     # Inventory holding cost per unit
BACK_COST = 2.0     # Backlog cost per unit
DELAY = 2           # Shipping delay (weeks)

# === Demand Pattern Generator ===
def generate_patterns():
    """Generate different demand scenarios to stress-test the BeerBot."""
    flat = [20] * WEEKS
    spike = [20 if w < 10 else 40 if 10 <= w < 20 else 10 for w in range(WEEKS)]
    trend = [15 + int(0.6 * w) for w in range(WEEKS)]
    random_shock = [max(5, int(20 + random.gauss(0, 5))) for _ in range(WEEKS)]
    recovery = [10] * 8 + [40] * 8 + [15] * 20
    return {
        "flat": flat,
        "spike": spike,
        "trend": trend,
        "random_shock": random_shock,
        "recovery": recovery
    }

# === Core Simulation ===
def run_simulation(decision_fn, pattern_name, demand_pattern):
    """Run a full 36-week Beer Distribution simulation for a given demand pattern."""
    history = []
    state = {r: {"inventory": 20, "backlog": 0, "incoming_orders": 0, "arriving_shipments": 0} for r in ROLES}
    shipments = {r: [0] * DELAY for r in ROLES}
    orders = {r: 10 for r in ROLES}

    total_inv_cost, total_back_cost = 0, 0

    for week in range(1, WEEKS + 1):
        customer_demand = demand_pattern[week - 1]

        # Update incoming orders downstream in the supply chain
        state["retailer"]["incoming_orders"] = customer_demand
        for i in range(1, len(ROLES)):
            lower, higher = ROLES[i - 1], ROLES[i]
            state[higher]["incoming_orders"] = orders[lower]

        # Handle arriving shipments
        for r in ROLES:
            arriving = shipments[r].pop(0)
            state[r]["inventory"] += arriving
            shipments[r].append(0)

        # Fulfill incoming orders
        for r in ROLES:
            demand = state[r]["incoming_orders"] + state[r]["backlog"]
            supply = min(demand, state[r]["inventory"])
            state[r]["inventory"] -= supply
            state[r]["backlog"] = demand - supply

        # === Prepare synthetic "weeks" structure for BeerBot ===
        fake_weeks = []
        for h in history:
            fake_weeks.append({
                "roles": {
                    r: {
                        "inventory": h.get(f"{r}_inventory", 20),
                        "backlog": h.get(f"{r}_backlog", 0),
                        "incoming_orders": 10,
                        "arriving_shipments": 10
                    } for r in ROLES
                },
                "orders": {r: h.get(f"{r}_order", 10) for r in ROLES}
            })
        fake_weeks.append({"roles": deepcopy(state), "orders": deepcopy(orders)})

        # === Get decisions from BeerBot ===
        order_decisions = {r: decision_fn(fake_weeks, r) for r in ROLES}
        orders = order_decisions
        for r in ROLES:
            shipments[r][DELAY - 1] += order_decisions[r]

        # === Cost computation ===
        inv_sum = sum(state[r]["inventory"] for r in ROLES)
        back_sum = sum(state[r]["backlog"] for r in ROLES)
        total_inv_cost += inv_sum * HOLD_COST
        total_back_cost += back_sum * BACK_COST

        # === Logging ===
        history.append({
            "week": week,
            "customer_demand": customer_demand,
            **{f"{r}_inventory": state[r]["inventory"] for r in ROLES},
            **{f"{r}_backlog": state[r]["backlog"] for r in ROLES},
            **{f"{r}_order": orders[r] for r in ROLES},
            "total_inventory": inv_sum,
            "total_backlog": back_sum
        })

    df = pd.DataFrame(history)
    df["pattern"] = pattern_name
    total_cost = total_inv_cost + total_back_cost
    return df, total_inv_cost, total_back_cost, total_cost

# === Performance Evaluation ===
def analyze_beerbot():
    """Run BeerBot across all demand scenarios and summarize performance."""
    results = []
    patterns = generate_patterns()
    all_data = []

    for name, pattern in patterns.items():
        df, inv, back, total = run_simulation(decide_beerbot, name, pattern)
        all_data.append(df)
        results.append([name, inv, back, total])

    summary = pd.DataFrame(results, columns=["Pattern", "Inventory cost", "Backlog cost", "Total cost"])

    console.print("\n[bold cyan]📊 BEERBOT PERFORMANCE SUMMARY[/bold cyan]")
    console.print(tabulate(summary, headers="keys", tablefmt="rounded_grid", showindex=False))

    avg_total = summary["Total cost"].mean()
    max_backlog = summary["Backlog cost"].max()

    # Performance tier classification
    if avg_total < 12000:
        rating = "⭐ Excellent (Top-tier efficiency)"
    elif avg_total < 18000:
        rating = "⚖️ Balanced and Stable"
    elif avg_total < 25000:
        rating = "🟠 Stable but Overstocked"
    else:
        rating = "🔴 Needs optimization"

    console.print(f"\n[bold yellow]Overall Efficiency Score:[/bold yellow] {rating}")
    console.print(f"Average Total Cost: [green]{avg_total:.0f}[/green]")
    console.print(f"Max Backlog Cost observed: [red]{max_backlog:.0f}[/red]\n")

    # Save full simulation logs
    all_df = pd.concat(all_data, ignore_index=True)
    all_df.to_csv("beerbot_log.csv", index=False)
    console.print("[dim]📁 Saved detailed logs to beerbot_log.csv[/dim]")

    # === Visualization ===
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=all_df, x="week", y="total_inventory", hue="pattern")
    plt.title("📦 Total Inventory by Scenario")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=all_df, x="week", y="total_backlog", hue="pattern")
    plt.title("📉 Total Backlog by Scenario")
    plt.grid(True)
    plt.show()

    return summary

# === Main Entry Point ===
if __name__ == "__main__":
    analyze_beerbot()
