"""Inference pipeline that produces a local 72-hour AQI forecast."""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

import config
from hopsworks_client import download_latest_model_from_hopsworks
from data_sources import fetch_openmeteo_weather_forecast


DEFAULT_PM25 = 50.0
DEFAULT_PM10 = 125.0
DEFAULT_CO = 500.0
DEFAULT_NO2 = 40.0
DEFAULT_SO2 = 10.0
DEFAULT_O3 = 60.0


def pm25_to_aqi(pm25):
    """Convert PM2.5 concentration (µg/m³) to US EPA AQI."""
    if pm25 <= 12.0:
        return (50.0 / 12.0) * pm25
    if pm25 <= 35.4:
        return 50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1)
    if pm25 <= 55.4:
        return 100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5)
    if pm25 <= 150.4:
        return 150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5)
    if pm25 <= 250.4:
        return 200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)
    if pm25 <= 350.4:
        return 300 + ((400 - 300) / (350.4 - 250.5)) * (pm25 - 250.5)
    if pm25 <= 500.4:
        return 400 + ((500 - 400) / (500.4 - 350.5)) * (pm25 - 350.5)
    return 500.0


def get_aqi_category(aqi):
    """Get AQI health category and color code."""
    if aqi <= 50:
        return "Good", "🟢"
    if aqi <= 100:
        return "Moderate", "🟡"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    if aqi <= 200:
        return "Unhealthy", "🔴"
    if aqi <= 300:
        return "Very Unhealthy", "🟣"
    return "Hazardous", "🟤"


def safe_mean(series, default_value):
    value = float(series.mean()) if len(series) > 0 else default_value
    if np.isnan(value):
        return default_value
    return value


def fetch_weather_forecast():
    print("📡 Fetching 72-hour weather forecast from Open-Meteo...")
    df = fetch_openmeteo_weather_forecast(config.FORECAST_HORIZON_HOURS)
    if df.empty:
        raise RuntimeError("Open-Meteo returned an empty weather forecast.")
    print(f"✅ Weather forecast fetched: {len(df)} hours")
    print(f"   Time range: {df['time'].min()} to {df['time'].max()}")
    return df


def load_model_and_features():
    print("📡 Authenticating with Hopsworks and downloading the latest model...")
    try:
        registry_model, metadata = download_latest_model_from_hopsworks()
    except Exception as exc:
        print(f"⚠️ Hopsworks model registry connection/download failed: {exc}")
        registry_model, metadata = None, None

    if registry_model is not None:
        feature_cols = metadata.get("feature_cols", []) if metadata else []
        if not feature_cols:
            raise RuntimeError(
                "Hopsworks model download succeeded, but feature-column metadata is missing. "
                "Re-run training_pipeline.py to republish the model artifact."
            )
        best_model_name = metadata.get("best_model_name", config.HOPSWORKS_MODEL_NAME) if metadata else config.HOPSWORKS_MODEL_NAME
        print(f"✅ Loaded model from Hopsworks registry: {best_model_name}")
        print(f"   Feature count: {len(feature_cols)}")
        return registry_model, feature_cols

    print("⚠️ Could not download model from Hopsworks model registry. Checking local fallback...")
    if config.MODEL_BUNDLE_PATH.exists():
        bundle = joblib.load(config.MODEL_BUNDLE_PATH)
        model_obj = bundle["model"]
        feature_cols = bundle["feature_cols"]
        best_model_name = bundle.get("best_model_name", "unknown")
        print(f"✅ Loaded local model bundle: {best_model_name}")
        print(f"   Feature count: {len(feature_cols)}")
        return model_obj, feature_cols

    raise RuntimeError(
        "Model loading failed: Hopsworks Model Registry is unavailable (or empty), "
        f"and no local model bundle was found at {config.MODEL_BUNDLE_PATH}."
    )


def get_recent_historical_data(hours: int = config.FORECAST_HORIZON_HOURS):
    # Try to download from Hopsworks feature group first
    try:
        from hopsworks_client import download_recent_historical_data_from_hopsworks
        df = download_recent_historical_data_from_hopsworks(hours)
        if df is not None and not df.empty:
            print(f"✅ Loaded {len(df)} historical records from Hopsworks Feature Group")
            print(f"   Time range: {df['time'].min()} to {df['time'].max()}")
            return df
    except Exception as exc:
        print(f"⚠️ Hopsworks historical data download failed: {exc}. Trying local fallback...")

    # Local fallback
    if not config.RAW_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw history at {config.RAW_HISTORY_PATH}. Run feature_pipeline.py first."
        )

    df = pd.read_parquet(config.RAW_HISTORY_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df = df.tail(hours)

    print(f"✅ Loaded {len(df)} historical records from local path")
    print(f"   Time range: {df['time'].min()} to {df['time'].max()}")
    return df


def generate_recursive_forecast(model, feature_cols, df_weather, df_historical):
    """Generate a recursive forecast for the configured horizon."""
    print("🔮 Generating 72-hour recursive forecast...")

    predictions = []
    historical_pm25 = df_historical["pm2_5"].tolist()
    historical_pm10 = df_historical["pm10"].tolist()
    historical_temp = df_historical["temperature_2m"].tolist()
    forecast_temps = []

    pm10_avg = safe_mean(df_historical["pm10"], DEFAULT_PM10)
    co_avg = safe_mean(df_historical["carbon_monoxide"], DEFAULT_CO) if "carbon_monoxide" in df_historical.columns else DEFAULT_CO
    no2_avg = safe_mean(df_historical["nitrogen_dioxide"], DEFAULT_NO2) if "nitrogen_dioxide" in df_historical.columns else DEFAULT_NO2
    so2_avg = safe_mean(df_historical["sulphur_dioxide"], DEFAULT_SO2) if "sulphur_dioxide" in df_historical.columns else DEFAULT_SO2
    o3_avg = safe_mean(df_historical["ozone"], DEFAULT_O3) if "ozone" in df_historical.columns else DEFAULT_O3

    for index, row in df_weather.iterrows():
        all_pm25 = historical_pm25 + predictions
        all_pm10 = historical_pm10 + [pm10_avg] * len(predictions)
        all_temp = historical_temp + forecast_temps

        features = {}
        time = row["time"]
        features["hour"] = time.hour
        features["day"] = time.day
        features["month"] = time.month
        features["day_of_week"] = time.dayofweek
        features["is_weekend"] = 1 if time.dayofweek >= 5 else 0

        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        features["temperature_2m"] = row["temperature_2m"]
        features["relative_humidity_2m"] = row["relative_humidity_2m"]
        features["pressure_msl"] = row["pressure_msl"]
        features["wind_speed_10m"] = row["wind_speed_10m"]
        features["wind_direction_10m"] = row["wind_direction_10m"]

        for lag in config.FEATURE_LAGS:
            if len(all_pm25) >= lag:
                features[f"pm2_5_lag_{lag}h"] = all_pm25[-lag]
            if len(all_pm10) >= lag:
                features[f"pm10_lag_{lag}h"] = all_pm10[-lag]
            if len(all_temp) >= lag:
                features[f"temp_lag_{lag}h"] = all_temp[-lag]

        for window in config.ROLLING_WINDOWS:
            if len(all_pm25) >= window:
                features[f"pm2_5_rolling_mean_{window}h"] = float(np.mean(all_pm25[-window:]))
                features[f"pm2_5_rolling_std_{window}h"] = float(np.std(all_pm25[-window:]))

        recent_pm25 = all_pm25[-1] if all_pm25 else DEFAULT_PM25
        recent_pm10 = all_pm10[-1] if all_pm10 else DEFAULT_PM10
        features["pm_ratio"] = recent_pm10 / (recent_pm25 + 1e-3)
        features["temp_humid_interaction"] = row["temperature_2m"] * row["relative_humidity_2m"]
        features["wind_pollution_interaction"] = row["wind_speed_10m"] * recent_pm25

        features["pm10"] = pm10_avg
        features["carbon_monoxide"] = co_avg
        features["nitrogen_dioxide"] = no2_avg
        features["sulphur_dioxide"] = so2_avg
        features["ozone"] = o3_avg

        for col in feature_cols:
            if col not in features:
                features[col] = 0

        X = pd.DataFrame([features])[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        pm25_pred = float(model.predict(X)[0])
        pm25_pred = max(0.0, pm25_pred)
        predictions.append(pm25_pred)
        forecast_temps.append(row["temperature_2m"])

        if (index + 1) % 24 == 0:
            print(f"   ✓ Generated {index + 1}/72 hour predictions")

    df_forecast = df_weather[["time"]].copy()
    df_forecast["pm2_5_forecast"] = predictions
    df_forecast["aqi_forecast"] = df_forecast["pm2_5_forecast"].apply(pm25_to_aqi)
    df_forecast[["aqi_category", "emoji"]] = df_forecast["aqi_forecast"].apply(
        lambda value: pd.Series(get_aqi_category(value))
    )

    print("✅ Forecast generation complete!")
    return df_forecast


def display_forecast_summary(df_forecast):
    print("\n📈 FORECAST SUMMARY (Next 72 Hours)")
    print("=" * 60)

    avg_pm25 = df_forecast["pm2_5_forecast"].mean()
    max_pm25 = df_forecast["pm2_5_forecast"].max()
    min_pm25 = df_forecast["pm2_5_forecast"].min()

    avg_aqi = df_forecast["aqi_forecast"].mean()
    max_aqi = df_forecast["aqi_forecast"].max()

    print(f"PM2.5 | Average: {avg_pm25:.1f} µg/m³ | Max: {max_pm25:.1f} | Min: {min_pm25:.1f}")
    print(f"AQI   | Average: {avg_aqi:.0f} | Max: {max_aqi:.0f}")

    print("\nAQI Category Breakdown:")
    category_counts = df_forecast["aqi_category"].value_counts()
    for category, count in category_counts.items():
        emoji = df_forecast[df_forecast["aqi_category"] == category]["emoji"].iloc[0]
        percentage = (count / len(df_forecast)) * 100
        print(f"  {emoji} {category}: {count} hours ({percentage:.1f}%)")

    print("\nDaily Average AQI:")
    df_forecast["date"] = df_forecast["time"].dt.date
    daily_aqi = df_forecast.groupby("date")["aqi_forecast"].mean()
    for date, aqi in daily_aqi.items():
        category, emoji = get_aqi_category(aqi)
        print(f"  {date}: {aqi:.0f} {emoji} ({category})")

    print("\n🕐 Next 24 Hours (Hourly):")
    for _, row in df_forecast.head(24).iterrows():
        time_str = row["time"].strftime("%Y-%m-%d %H:%M")
        pm25 = row["pm2_5_forecast"]
        aqi = row["aqi_forecast"]
        emoji = row["emoji"]
        print(f"  {time_str} | PM2.5: {pm25:6.1f} µg/m³ | AQI: {aqi:5.0f} {emoji}")


def save_forecast(df_forecast):
    print("\n💾 Saving forecast...")
    config.FORECAST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_forecast.to_csv(config.FORECAST_OUTPUT_PATH, index=False)
    print(f"   ✓ Saved to {config.FORECAST_OUTPUT_PATH}")

    from hopsworks_client import publish_forecast_to_hopsworks
    publish_forecast_to_hopsworks(df_forecast)

    print("✅ Forecast saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate a 72-hour AQI forecast.")
    parser.parse_args()

    print("=" * 60)
    print("🌫️  PM2.5 & AQI FORECASTING - INFERENCE PIPELINE")
    print("=" * 60)

    model, feature_cols = load_model_and_features()
    df_historical = get_recent_historical_data(hours=config.FORECAST_HORIZON_HOURS)
    df_weather = fetch_weather_forecast()
    df_forecast = generate_recursive_forecast(model, feature_cols, df_weather, df_historical)
    display_forecast_summary(df_forecast)
    save_forecast(df_forecast)

    print("\n" + "=" * 60)
    print("✅ INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()