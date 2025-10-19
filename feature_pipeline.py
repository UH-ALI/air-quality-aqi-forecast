"""
feature_pipeline.py
Purpose:
    - Fetch latest weather & air quality data from Open-Meteo
    - Compute features for PM2.5 prediction
    - Store results into Hopsworks Feature Store

Run manually (for testing):
    python feature_pipeline.py
"""

import hopsworks
import pandas as pd
import requests
import datetime
import time
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

# --- SETTINGS ---
CITY = "Delhi"
LAT = 28.6139
LON = 77.2090

# --- FETCH DATA ---
def fetch_weather_data():
    print("📡 Fetching data from Open-Meteo APIs...")
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=1)  # last 24 hours
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly=pm2_5,pm10,so2,no2,o3,co,temperature_2m,relative_humidity_2m,"
        f"surface_pressure,wind_speed_10m,wind_direction_10m"
        f"&start_date={start.strftime('%Y-%m-%d')}"
        f"&end_date={end.strftime('%Y-%m-%d')}"
        "&timezone=UTC"
    )
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    print(f"✅ Data fetched: {df.shape[0]} rows")
    return df

# --- FEATURE ENGINEERING ---
def engineer_features(df):
    print("🧮 Engineering features...")
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["pm_ratio"] = df["pm10"] / (df["pm2_5"] + 1e-3)
    df["temp_humid_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df.dropna(inplace=True)
    return df

# --- STORE IN HOPSWORKS ---
def store_in_hopsworks(df):
    print("🗄️ Logging into Hopsworks...")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="pm25_features",
        version=1,
        primary_key=["time"],
        description="Latest hourly weather & air-quality features for PM2.5 forecasting."
    )
    fg.insert(df)
    print("✅ Data successfully stored in Hopsworks Feature Store!")

# --- MAIN ---
def main():
    df_raw = fetch_weather_data()
    df_feat = engineer_features(df_raw)
    store_in_hopsworks(df_feat)

if __name__ == "__main__":
    main()
