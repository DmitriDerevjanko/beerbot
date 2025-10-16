import json, requests, numpy as np

files = ["1.json","3.json","4.json"]
url = "http://127.0.0.1:8000/api/decision"

for file in files:
    data = json.load(open(file))
    payload = {"mode":"glassbox","weeks_total":36,"weeks":data["weeklyData"]}
    res = requests.post(url,json=payload).json()
    m = res["metrics"]

    # анализ тренда backlog
    backlog_series = [w["roles"]["retailer"]["backlog"] for w in data["weeklyData"]]
    trend = np.polyfit(range(len(backlog_series)), backlog_series, 1)[0]

    print(f"\n📂 {file}")
    print(f"🧠 Orders: {res['orders']}")
    print(f"📦 Inv.cost={m['inventory_cost']} ⛔ Backlog={m['backlog_cost']} 💰 Total={m['total_cost']}")
    print(f"📈 PeakInv={m['peak_inventory']}, PeakBack={m['peak_backlog']}")
    print(f"📉 Backlog trend slope: {trend:+.2f}")
a