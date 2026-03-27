"""
feature_pipeline.py
Purpose:
    - Fetch latest weather & air quality data from Open-Meteo
    - Compute features for PM2.5 prediction
    - Store results into Hopsworks Feature Store

Environment variables:
    HOPSWORKS_API_KEY  - Required. Hopsworks project API key.
    BACKFILL_DAYS      - Optional. Number of days of history to fetch (default: 1).
                         Set to 365 (or more) on the first run to backfill a full year.
"""

import hopsworks
import pandas as pd
import numpy as np
import requests
import datetime
import os

# --- SETTINGS ---
CITY = "Delhi"
LAT = 28.6139
LON = 77.2090

# How many days of data to fetch.  Override via env-var for initial backfill.
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "1"))

# Open-Meteo archive API only covers dates up to ~5 days ago; use forecast API
# with past_days for more recent data.
_ARCHIVE_LAG_DAYS = 5
_MAX_FORECAST_PAST_DAYS = 90  # forecast API past_days limit


def _fetch_weather_chunk(start_dt, end_dt):
    """Fetch one chunk of hourly weather data between two dates."""
    today = datetime.datetime.utcnow().date()
    cutoff = today - datetime.timedelta(days=_ARCHIVE_LAG_DAYS)

    if end_dt.date() <= cutoff:
        # Pure historical range → use archive API
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        # Includes recent / future dates → use forecast API
        base_url = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base_url}?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,"
        f"wind_speed_10m,wind_direction_10m"
        f"&start_date={start_dt.strftime('%Y-%m-%d')}"
        f"&end_date={end_dt.strftime('%Y-%m-%d')}"
        f"&timezone=UTC"
    )
    response = requests.get(url, timeout=60)
    data = response.json()

    if "hourly" not in data:
        raise KeyError(
            f"❌ No 'hourly' key in weather response "
            f"({start_dt.date()} → {end_dt.date()}): {data}"
        )

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather_data(days_back: int = 1) -> pd.DataFrame:
    """Fetch hourly weather data for the past *days_back* days.

    For large ranges the request is split into 90-day chunks so that we can
    mix archive-API and forecast-API calls as needed.
    """
    print(f"📡 Fetching {days_back}-day weather data from Open-Meteo...")
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=days_back)

    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + datetime.timedelta(days=_MAX_FORECAST_PAST_DAYS), end)
        chunks.append(_fetch_weather_chunk(chunk_start, chunk_end))
        chunk_start = chunk_end + datetime.timedelta(hours=1)

    df = pd.concat(chunks, ignore_index=True).drop_duplicates("time")
    df = df.sort_values("time").reset_index(drop=True)
    print(f"✅ Weather data fetched: {df.shape[0]} rows")
    return df


def _fetch_air_quality_chunk(start_dt, end_dt):
    """Fetch one chunk of hourly air-quality data."""
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        f"&start_date={start_dt.strftime('%Y-%m-%d')}"
        f"&end_date={end_dt.strftime('%Y-%m-%d')}"
        f"&timezone=UTC"
    )
    response = requests.get(url, timeout=60)
    data = response.json()

    if "hourly" not in data:
        raise KeyError(
            f"❌ No 'hourly' key in air-quality response "
            f"({start_dt.date()} → {end_dt.date()}): {data}"
        )

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.rename(
        columns={
            "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2",
            "sulphur_dioxide": "so2",
            "ozone": "o3",
        },
        inplace=True,
    )
    return df


def fetch_air_quality_data(days_back: int = 1) -> pd.DataFrame:
    """Fetch hourly air-quality data for the past *days_back* days.

    Splits large ranges into 90-day chunks.
    """
    print(f"🌫️ Fetching {days_back}-day air-quality data from Open-Meteo...")
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=days_back)

    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + datetime.timedelta(days=_MAX_FORECAST_PAST_DAYS), end)
        chunks.append(_fetch_air_quality_chunk(chunk_start, chunk_end))
        chunk_start = chunk_end + datetime.timedelta(hours=1)

    df = pd.concat(chunks, ignore_index=True).drop_duplicates("time")
    df = df.sort_values("time").reset_index(drop=True)
    print(f"✅ Air-quality data fetched: {df.shape[0]} rows")
    return df

def create_lag_features(df, lags=[1, 3, 6, 12, 24]):
    """
    Create lag features for time series prediction
    lags: hours to look back
    """
    df = df.sort_values('time').reset_index(drop=True)
    
    for lag in lags:
        df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag)
        df[f'pm10_lag_{lag}h'] = df['pm10'].shift(lag)
        df[f'temp_lag_{lag}h'] = df['temperature_2m'].shift(lag)
    
    # Rolling statistics - use min_periods=24 to calculate only from complete 24-hour windows
    # Earlier rows (hours 1-23) will have NaN values and be dropped later
    df['pm2_5_rolling_mean_24h'] = df['pm2_5'].rolling(window=24, min_periods=24).mean()
    df['pm2_5_rolling_std_24h'] = df['pm2_5'].rolling(window=24, min_periods=24).std()
    
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
    
    # Temporal features
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Cyclical encoding (captures circular nature of time)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Interaction features
    df["pm_ratio"] = df["pm10"] / (df["pm2_5"] + 1e-3)
    df["temp_humid_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df["wind_pollution_interaction"] = df["wind_speed_10m"] * df["pm2_5"]
    
    # Add lag features
    df = create_lag_features(df)
    
    # Drop rows with NaN (first 24 hours will be excluded due to lag features)
    # This is intentional - we need complete lag features for accurate predictions
    df.dropna(inplace=True)
    print(f"✅ Engineered features: {df.shape}")
    print(f"   Note: First 24 hours excluded due to lag feature requirements")
    return df


def store_in_hopsworks(df):
    print("🗄️ Logging into Hopsworks...")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ HOPSWORKS_API_KEY environment variable is not set.\n"
            "   Please add it as a GitHub repository secret:\n"
            "   Settings → Secrets and variables → Actions → New repository secret\n"
            "   Name: HOPSWORKS_API_KEY\n"
            "   Value: <your Hopsworks API key from Project Settings → API Keys>"
        )
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="pm25_features_v2",
        version=1,
        primary_key=["time"],
        description="Updated features (Open-Meteo AQ + weather)"
    )

    fg.insert(df)
    print("✅ Data successfully stored in Hopsworks Feature Store!")


def main():
    print(f"🔧 Running feature pipeline (BACKFILL_DAYS={BACKFILL_DAYS})")
    df_weather = fetch_weather_data(days_back=BACKFILL_DAYS)
    df_pollution = fetch_air_quality_data(days_back=BACKFILL_DAYS)
    df_feat = engineer_features(df_weather, df_pollution)

    if df_feat.empty:
        raise RuntimeError(
            "❌ Feature engineering produced an empty DataFrame. "
            "This usually means BACKFILL_DAYS is too small (< 2) to create "
            "the required 24-hour lag features. "
            "Set BACKFILL_DAYS=365 for an initial backfill."
        )

    store_in_hopsworks(df_feat)


if __name__ == "__main__":
    main()
