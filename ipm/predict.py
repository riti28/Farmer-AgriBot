"""
Step 6: Use the trained neural network to predict pest risk for ONE situation,
get the latest real weather for ANY place on Earth (NASA POWER),
and fetch matching IPM tips from the corpus with rag.py.
Used by app.py (Streamlit) and mcp_server.py.

Run: python predict.py   (prints one example)
"""
import os
from datetime import date, timedelta
import joblib
import pandas as pd
import rag
from fetch_data import fetch_daily
from make_dataset import THRESHOLDS

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(os.path.join(HERE, "models", "best_model.joblib"))


def predict_risk(crop, crop_stage, temp, humidity, rainfall):
    """Return (risk_label, confidence 0-1, {label: probability})."""
    row = pd.DataFrame([{"crop": crop, "crop_stage": crop_stage,
                         "temp": temp, "humidity": humidity, "rainfall": rainfall}])
    probs = MODEL.predict_proba(row)[0]           # one probability per class
    all_probs = {str(label): round(float(p), 3) for label, p in zip(MODEL.classes_, probs)}
    best = max(all_probs, key=all_probs.get)
    return best, float(all_probs[best]), all_probs


def latest_week_weather(lat, lon):
    """Mean temp, mean humidity and total rain of the latest 7 days NASA POWER has.
    (NASA POWER data arrive a few days late, so we ask for the last 3 weeks.)"""
    today = date.today()
    daily = fetch_daily(lat, lon, (today - timedelta(days=21)).strftime("%Y%m%d"),
                        today.strftime("%Y%m%d")).tail(7)
    return {"temp": round(float(daily["T2M"].mean()), 1),
            "humidity": round(float(daily["RH2M"].mean()), 1),
            "rainfall": round(float(daily["PRECTOTCORR"].sum()), 1),
            "from": daily.index[0].date().isoformat(), "to": daily.index[-1].date().isoformat()}


def ipm_tips(crop):
    """Search the corpus for management tips for this crop's main pests."""
    tips = []
    for rule in THRESHOLDS[crop]:
        pest_name = rule["pest"].split(" (")[0]      # "Brown planthopper (...)" -> "Brown planthopper"
        for hit in rag.search(f"{pest_name} {crop.replace('_', ' ')} management IPM", k=3):
            if hit["score"] >= rag.THRESHOLD and hit["text"] not in [t["text"] for t in tips]:
                tips.append(dict(hit, pest=pest_name))
    return tips


if __name__ == "__main__":
    print("Risk:", predict_risk("maize", "vegetative", 27.5, 75, 12))
    print("Latest weather, Nairobi:", latest_week_weather(-1.29, 36.82))
    for tip in ipm_tips("maize"):
        print("-", tip["pest"], "->", tip["title"], f"({tip['org']})")
