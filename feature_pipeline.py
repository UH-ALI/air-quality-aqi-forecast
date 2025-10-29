"""
feature_pipeline.py
Purpose:
    - Fetch latest weather & air quality data from Open-Meteo + OpenWeather
    - Compute features for PM2.5 prediction
    - Store results into Hopsworks Feature Store
"""

import hopsworks
import pandas as pd
import requests
import datetime
import os

# --- SETTINGS ---
CITY = "Delhi"
LAT = 28.6139
LON = 77.2090

def fetch_weather_data():
    print("📡 Fetching hourly weather data from Open-Meteo...")
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=1)
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m"
        f"&start_date={start.strftime('%Y-%m-%d')}"
        f"&end_date={end.strftime('%Y-%m-%d')}"
        f"&timezone=UTC"
    )
    response = requests.get(url)
    data = response.json()
    
    if "hourly" not in data:
        raise KeyError("❌ No 'hourly' key found in Open-Meteo response.")
    
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    print(f"✅ Weather data fetched: {df.shape[0]} rows")
    return df

def fetch_air_quality_data():
    print("🌫️ Fetching air quality data from Open-Meteo (no API key needed)...")
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=1)
    
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        f"&start_date={start.strftime('%Y-%m-%d')}"
        f"&end_date={end.strftime('%Y-%m-%d')}"
        f"&timezone=UTC"
    )

    response = requests.get(url)
    data = response.json()

    if "hourly" not in data:
        raise KeyError("❌ 'hourly' key not found in Open-Meteo air-quality response.")

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    # Rename columns for consistency with your feature engineering code
    df.rename(
        columns={
            "pm10": "pm10",
            "pm2_5": "pm2_5",
            "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2",
            "sulphur_dioxide": "so2",
            "ozone": "o3",
        },
        inplace=True,
    )

    print(f"✅ Air quality data fetched: {df.shape[0]} rows")
    return df

def engineer_features(df_weather, df_pollution):
    print("🧮 Merging and engineering features...")
    df = pd.merge_asof(
        df_pollution.sort_values("time"),
        df_weather.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("1h")
    )
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["pm_ratio"] = df["pm10"] / (df["pm2_5"] + 1e-3)
    df["temp_humid_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df.dropna(inplace=True)
    print(f"✅ Engineered features: {df.shape}")
    return df


def store_in_hopsworks(df):
    print("🗄️ Logging into Hopsworks...")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
    name="pm25_features_v2",     # new name
    version=1,
    primary_key=["time"],
    description="Updated features (Open-Meteo AQ + weather)"
    )

    fg.insert(df)
    print("✅ Data successfully stored in Hopsworks Feature Store!")


def main():
    df_weather = fetch_weather_data()
    df_pollution = fetch_air_quality_data()
    df_feat = engineer_features(df_weather, df_pollution)
    store_in_hopsworks(df_feat)


if __name__ == "__main__":
    main()
