"""
inference_pipeline.py

Purpose:
    - Load trained model from Hopsworks
    - Fetch 72-hour weather forecast
    - Generate recursive PM2.5 predictions
    - Convert PM2.5 to AQI (US EPA standard)
    - Store forecasts for monitoring

Prediction Strategy:
    - Recursive forecasting: each hour's prediction uses previous predictions
    - Uses last 72 hours of actual data for initial lag features
    - Weather forecast provides temperature, humidity, wind, etc.
    - Outputs: PM2.5 (µg/m³), AQI (0-500), Health Category
"""

import hopsworks
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
import os
import json

# Delhi coordinates
LAT = 28.6139
LON = 77.2090

# Default values for missing pollutant data (based on typical Delhi averages)
DEFAULT_PM25 = 50.0  # µg/m³ - typical moderate PM2.5 level for Delhi
DEFAULT_PM10 = 125.0  # µg/m³ - typical moderate PM10 level for Delhi
PM10_PM25_RATIO = 2.5  # Typical ratio of PM10 to PM2.5 in urban environments
DEFAULT_CO = 500.0  # µg/m³ - typical CO level
DEFAULT_NO2 = 40.0  # µg/m³ - typical NO2 level
DEFAULT_SO2 = 10.0  # µg/m³ - typical SO2 level
DEFAULT_O3 = 60.0  # µg/m³ - typical O3 level

def pm25_to_aqi(pm25):
    """
    Convert PM2.5 concentration (µg/m³) to US EPA AQI
    
    AQI Breakpoints:
    PM2.5 (µg/m³)  |  AQI  |  Category
    0.0 - 12.0     |  0-50 |  Good
    12.1 - 35.4    | 51-100|  Moderate
    35.5 - 55.4    |101-150|  Unhealthy for Sensitive Groups
    55.5 - 150.4   |151-200|  Unhealthy
    150.5 - 250.4  |201-300|  Very Unhealthy
    250.5 - 350.4  |301-400|  Hazardous
    350.5 - 500.4  |401-500|  Hazardous
    500.5+         |  500  |  Beyond AQI (capped at 500)
    
    Formula: AQI = [(I_high - I_low) / (C_high - C_low)] * (C - C_low) + I_low
    """
    if pm25 <= 12.0:
        return (50.0 / 12.0) * pm25
    elif pm25 <= 35.4:
        return 50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1)
    elif pm25 <= 55.4:
        return 100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5)
    elif pm25 <= 150.4:
        return 150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5)
    elif pm25 <= 250.4:
        return 200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5)
    elif pm25 <= 350.4:
        return 300 + ((400 - 300) / (350.4 - 250.5)) * (pm25 - 250.5)
    elif pm25 <= 500.4:
        return 400 + ((500 - 400) / (500.4 - 350.5)) * (pm25 - 350.5)
    else:
        # Cap at 500 - AQI scale doesn't extend beyond 500
        return 500.0

def get_aqi_category(aqi):
    """Get AQI health category and color code"""
    if aqi <= 50:
        return "Good", "🟢"
    elif aqi <= 100:
        return "Moderate", "🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi <= 200:
        return "Unhealthy", "🔴"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣"
    else:
        return "Hazardous", "🟤"

def fetch_weather_forecast():
    """
    Fetch 72-hour weather forecast from Open-Meteo
    Returns: DataFrame with hourly weather data
    """
    print("📡 Fetching 72-hour weather forecast from Open-Meteo...")
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m"
        f"&forecast_days=3"  # 3 days = 72 hours
        f"&timezone=UTC"
    )
    
    response = requests.get(url)
    data = response.json()
    
    if "hourly" not in data:
        raise KeyError("❌ No 'hourly' key in weather forecast response")
    
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    
    print(f"✅ Weather forecast fetched: {len(df)} hours")
    print(f"   Time range: {df['time'].min()} to {df['time'].max()}")
    
    return df

def load_model_and_features():
    """
    Load both trained models and feature definitions from Hopsworks Model Registry.
    Falls back gracefully if only one model is available.

    Returns:
        models      – dict  {"xgboost": <model>, "random_forest": <model>}
        feature_cols – list of feature column names
        best_model  – name of the best model recorded during training
        fs, project – Hopsworks handles
    """
    print("🗄️ Loading models from Hopsworks Model Registry...")

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
    mr = project.get_model_registry()
    fs = project.get_feature_store()

    models = {}
    feature_cols = None
    best_model = "pm25_xgboost"

    # --- Load XGBoost ---
    try:
        xgb_reg = mr.get_model("pm25_xgboost", version=1)
        model_dir = xgb_reg.download()

        models["xgboost"] = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
        print(f"   ✓ XGBoost model loaded")

        # Read feature list (same for both models)
        features_path = os.path.join(model_dir, "features.json")
        if os.path.exists(features_path):
            with open(features_path) as f:
                feature_cols = json.load(f)["features"]

        # Read best-model annotation from saved metrics
        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                meta = json.load(f)
                best_model = meta.get("best_model", best_model)
    except Exception as e:
        print(f"   ⚠️  Could not load XGBoost model: {e}")

    # --- Load Random Forest ---
    try:
        rf_reg = mr.get_model("pm25_rf", version=1)
        rf_dir = rf_reg.download()
        models["random_forest"] = joblib.load(os.path.join(rf_dir, "rf_model.pkl"))
        print(f"   ✓ Random Forest model loaded")

        if feature_cols is None:
            features_path = os.path.join(rf_dir, "features.json")
            if os.path.exists(features_path):
                with open(features_path) as f:
                    feature_cols = json.load(f)["features"]
    except Exception as e:
        print(f"   ⚠️  Could not load Random Forest model: {e}")

    if not models:
        raise RuntimeError(
            "❌ No models could be loaded from Hopsworks. "
            "Please run the training pipeline first."
        )

    if feature_cols is None:
        raise RuntimeError("❌ Could not load feature column list from any model.")

    print(f"   ✓ {len(models)} model(s) ready | {len(feature_cols)} features")

    return models, feature_cols, best_model, fs, project

def get_recent_historical_data(fs, hours=72):
    """
    Get recent historical data for lag feature initialization
    We need past 72 hours to create lag features for first predictions
    """
    print(f"📊 Loading last {hours} hours of historical data...")
    
    # Note: Name 'pm25_features_v2' (schema name) with version=1 (first version of this schema)
    # Should match training pipeline. Consider making version configurable for production.
    fg = fs.get_feature_group(name="pm25_features_v2", version=1)
    df = fg.read()
    
    # Get most recent data
    df = df.sort_values('time')
    df = df.tail(hours)
    
    print(f"✅ Loaded {len(df)} historical records")
    print(f"   Time range: {df['time'].min()} to {df['time'].max()}")
    
    return df

def generate_recursive_forecast(models, feature_cols, df_weather, df_historical):
    """
    Generate 72-hour recursive forecast using one or more models.

    When multiple models are provided, each model predicts independently and
    the final PM2.5 forecast is the average of all predictions (ensemble).

    Recursive Forecasting:
    - Hour 1: Uses actual historical lag features
    - Hour 2: Uses Hour 1 ensemble prediction for lag_1h, actual data for older lags
    - Hour 3: Uses Hour 2 ensemble prediction, Hour 1 for lag_2h, etc.
    """
    print(f"🔮 Generating 72-hour recursive forecast with {len(models)} model(s)...")

    # One predictions list per model
    model_predictions = {name: [] for name in models}

    # Extract historical PM2.5 values for lag features
    historical_pm25 = df_historical['pm2_5'].tolist()
    historical_pm10 = df_historical['pm10'].tolist()
    historical_temp = df_historical['temperature_2m'].tolist()

    pm10_avg = df_historical['pm10'].mean() if len(df_historical) > 0 else DEFAULT_PM10

    forecast_temps = []
    first_model_name = next(iter(models))

    for i, row in df_weather.iterrows():
        # Ensemble predictions so far (average across models)
        n_so_far = len(model_predictions[first_model_name])
        if n_so_far > 0:
            ensemble_so_far = [
                np.mean([model_predictions[name][t] for name in models])
                for t in range(n_so_far)
            ]
        else:
            ensemble_so_far = []

        all_pm25 = historical_pm25 + ensemble_so_far
        all_pm10 = historical_pm10 + [pm10_avg] * len(ensemble_so_far)
        all_temp = historical_temp + forecast_temps

        # Build feature dictionary (shared across models)
        features = {}
        time = row['time']
        features['hour'] = time.hour
        features['day'] = time.day
        features['month'] = time.month
        features['day_of_week'] = time.dayofweek
        features['is_weekend'] = 1 if time.dayofweek >= 5 else 0

        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        features['temperature_2m'] = row['temperature_2m']
        features['relative_humidity_2m'] = row['relative_humidity_2m']
        features['pressure_msl'] = row['pressure_msl']
        features['wind_speed_10m'] = row['wind_speed_10m']
        features['wind_direction_10m'] = row['wind_direction_10m']

        if len(all_pm25) >= 1:
            features['pm2_5_lag_1h'] = all_pm25[-1]
        if len(all_pm25) >= 3:
            features['pm2_5_lag_3h'] = all_pm25[-3]
        if len(all_pm25) >= 6:
            features['pm2_5_lag_6h'] = all_pm25[-6]
        if len(all_pm25) >= 12:
            features['pm2_5_lag_12h'] = all_pm25[-12]
        if len(all_pm25) >= 24:
            features['pm2_5_lag_24h'] = all_pm25[-24]
            features['pm2_5_rolling_mean_24h'] = np.mean(all_pm25[-24:])
            features['pm2_5_rolling_std_24h'] = np.std(all_pm25[-24:])

        if len(all_pm10) >= 1:
            features['pm10_lag_1h'] = all_pm10[-1]
        if len(all_pm10) >= 3:
            features['pm10_lag_3h'] = all_pm10[-3]
        if len(all_pm10) >= 6:
            features['pm10_lag_6h'] = all_pm10[-6]
        if len(all_pm10) >= 12:
            features['pm10_lag_12h'] = all_pm10[-12]
        if len(all_pm10) >= 24:
            features['pm10_lag_24h'] = all_pm10[-24]

        if len(all_temp) >= 1:
            features['temp_lag_1h'] = all_temp[-1]
        if len(all_temp) >= 3:
            features['temp_lag_3h'] = all_temp[-3]
        if len(all_temp) >= 6:
            features['temp_lag_6h'] = all_temp[-6]
        if len(all_temp) >= 12:
            features['temp_lag_12h'] = all_temp[-12]
        if len(all_temp) >= 24:
            features['temp_lag_24h'] = all_temp[-24]

        recent_pm25 = all_pm25[-1] if all_pm25 else DEFAULT_PM25
        recent_pm10 = all_pm10[-1] if all_pm10 else DEFAULT_PM10

        features['pm_ratio'] = recent_pm10 / (recent_pm25 + 1e-3)
        features['temp_humid_interaction'] = row['temperature_2m'] * row['relative_humidity_2m']
        features['wind_pollution_interaction'] = row['wind_speed_10m'] * recent_pm25

        features['pm10'] = pm10_avg
        features['co'] = df_historical['co'].mean() if 'co' in df_historical.columns else DEFAULT_CO
        features['no2'] = df_historical['no2'].mean() if 'no2' in df_historical.columns else DEFAULT_NO2
        features['so2'] = df_historical['so2'].mean() if 'so2' in df_historical.columns else DEFAULT_SO2
        features['o3'] = df_historical['o3'].mean() if 'o3' in df_historical.columns else DEFAULT_O3

        for col in feature_cols:
            if col not in features:
                features[col] = 0

        X = pd.DataFrame([features])[feature_cols]

        # Predict with each model independently
        for name, model in models.items():
            pm25_pred = float(model.predict(X)[0])
            pm25_pred = max(0.0, pm25_pred)
            model_predictions[name].append(pm25_pred)

        forecast_temps.append(row['temperature_2m'])

        step = i + 1
        if step % 24 == 0:
            print(f"   ✓ Generated {step}/72 hour predictions")

    # Build forecast DataFrame with per-model columns + ensemble
    df_forecast = df_weather[['time']].copy().reset_index(drop=True)

    for name, preds in model_predictions.items():
        col = f"pm2_5_{name}"
        df_forecast[col] = preds
        df_forecast[f"aqi_{name}"] = df_forecast[col].apply(pm25_to_aqi)

    # Ensemble: average of all models
    pm25_cols = [f"pm2_5_{n}" for n in models]
    df_forecast['pm2_5_forecast'] = df_forecast[pm25_cols].mean(axis=1)
    df_forecast['aqi_forecast'] = df_forecast['pm2_5_forecast'].apply(pm25_to_aqi)

    df_forecast[['aqi_category', 'emoji']] = df_forecast['aqi_forecast'].apply(
        lambda x: pd.Series(get_aqi_category(x))
    )

    print("✅ Forecast generation complete!")
    return df_forecast

def display_forecast_summary(df_forecast, models):
    """Display forecast summary statistics including per-model and ensemble results."""
    print("\n📈 FORECAST SUMMARY (Next 72 Hours – 3-Day AQI Outlook)")
    print("=" * 60)

    # Per-model stats
    model_names = list(models.keys())
    if len(model_names) > 1:
        print("\n🔎 Per-Model Comparison (Validation):")
        for name in model_names:
            col = f"pm2_5_{name}"
            if col in df_forecast.columns:
                avg = df_forecast[col].mean()
                print(f"   {name:<20} avg PM2.5: {avg:.1f} µg/m³")
        print(f"   {'ensemble':<20} avg PM2.5: {df_forecast['pm2_5_forecast'].mean():.1f} µg/m³")

    # Overall ensemble statistics
    avg_pm25 = df_forecast['pm2_5_forecast'].mean()
    max_pm25 = df_forecast['pm2_5_forecast'].max()
    min_pm25 = df_forecast['pm2_5_forecast'].min()
    avg_aqi = df_forecast['aqi_forecast'].mean()
    max_aqi = df_forecast['aqi_forecast'].max()

    print(f"\nEnsemble PM2.5 | Avg: {avg_pm25:.1f} µg/m³ | Max: {max_pm25:.1f} | Min: {min_pm25:.1f}")
    print(f"Ensemble AQI   | Avg: {avg_aqi:.0f} | Max: {max_aqi:.0f}")

    # Category breakdown
    print("\nAQI Category Breakdown:")
    category_counts = df_forecast['aqi_category'].value_counts()
    for category, count in category_counts.items():
        emoji = df_forecast[df_forecast['aqi_category'] == category]['emoji'].iloc[0]
        percentage = (count / len(df_forecast)) * 100
        print(f"  {emoji} {category}: {count} hours ({percentage:.1f}%)")

    # 3-day daily summary
    print("\n🗓️  3-Day Daily AQI Forecast:")
    df_forecast['date'] = df_forecast['time'].dt.date
    daily = df_forecast.groupby('date').agg(
        avg_aqi=('aqi_forecast', 'mean'),
        max_aqi=('aqi_forecast', 'max'),
        avg_pm25=('pm2_5_forecast', 'mean'),
    )
    for date, row in daily.iterrows():
        category, emoji = get_aqi_category(row['avg_aqi'])
        print(
            f"  {date}  AQI avg: {row['avg_aqi']:.0f} (max {row['max_aqi']:.0f}) "
            f"| PM2.5 avg: {row['avg_pm25']:.1f} µg/m³  {emoji} {category}"
        )

    # Hourly breakdown for first 24 hours
    print("\n🕐 Next 24 Hours (Hourly):")
    for _, row in df_forecast.head(24).iterrows():
        time_str = row['time'].strftime('%Y-%m-%d %H:%M')
        pm25 = row['pm2_5_forecast']
        aqi = row['aqi_forecast']
        emoji = row['emoji']
        print(f"  {time_str} | PM2.5: {pm25:6.1f} µg/m³ | AQI: {aqi:5.0f} {emoji}")

def save_forecast(df_forecast, fs, project):
    """Save forecast to Hopsworks and local CSV"""
    print("\n💾 Saving forecast...")
    
    # Save to CSV
    csv_path = "forecast_72h.csv"
    df_forecast.to_csv(csv_path, index=False)
    print(f"   ✓ Saved to {csv_path}")
    
    # Save to Hopsworks Feature Store
    try:
        fg = fs.get_or_create_feature_group(
            name="pm25_forecasts",
            version=1,
            primary_key=["time"],
            description="72-hour PM2.5 and AQI forecasts generated by XGBoost model"
        )
        fg.insert(df_forecast, write_options={"wait_for_job": False})
        print(f"   ✓ Saved to Hopsworks Feature Group: pm25_forecasts")
    except Exception as e:
        print(f"   ⚠️  Could not save to Hopsworks: {e}")
    
    print("✅ Forecast saved successfully!")

def main():
    print("=" * 60)
    print("🌫️  PM2.5 & AQI FORECASTING - INFERENCE PIPELINE")
    print("=" * 60)

    # Step 1: Load models and features
    models, feature_cols, best_model, fs, project = load_model_and_features()

    # Step 2: Get historical data for lag features
    df_historical = get_recent_historical_data(fs, hours=72)

    # Step 3: Fetch weather forecast
    df_weather = fetch_weather_forecast()

    # Step 4: Generate forecast (ensemble of all available models)
    df_forecast = generate_recursive_forecast(models, feature_cols, df_weather, df_historical)

    # Step 5: Display summary
    display_forecast_summary(df_forecast, models)

    # Step 6: Save forecast
    save_forecast(df_forecast, fs, project)

    print("\n" + "=" * 60)
    print("✅ INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
