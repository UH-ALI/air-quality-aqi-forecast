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
    350.5+         |401-500|  Hazardous
    
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
    else:
        return 400 + ((500 - 400) / (500.4 - 350.5)) * (pm25 - 350.5)

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
    Load trained model and feature definitions from Hopsworks Model Registry
    """
    print("🗄️ Loading model from Hopsworks Model Registry...")
    
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found")
    
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    
    # Get latest version of model
    model = mr.get_model("pm25_xgboost", version=1)
    model_dir = model.download()
    
    # Load model binary
    model_obj = joblib.load(f"{model_dir}/xgboost_model.pkl")
    print(f"   ✓ Model loaded")
    
    # Load feature names
    with open(f"{model_dir}/features.json", "r") as f:
        feature_data = json.load(f)
        feature_cols = feature_data["features"]
    print(f"   ✓ Feature names loaded ({len(feature_cols)} features)")
    
    # Get feature store
    fs = project.get_feature_store()
    
    return model_obj, feature_cols, fs, project

def get_recent_historical_data(fs, hours=72):
    """
    Get recent historical data for lag feature initialization
    We need past 72 hours to create lag features for first predictions
    """
    print(f"📊 Loading last {hours} hours of historical data...")
    
    fg = fs.get_feature_group(name="pm25_features_v2", version=1)
    df = fg.read()
    
    # Get most recent data
    df = df.sort_values('time')
    df = df.tail(hours)
    
    print(f"✅ Loaded {len(df)} historical records")
    print(f"   Time range: {df['time'].min()} to {df['time'].max()}")
    
    return df

def generate_recursive_forecast(model, feature_cols, df_weather, df_historical):
    """
    Generate 72-hour recursive forecast
    
    Recursive Forecasting:
    - Hour 1: Uses actual historical lag features
    - Hour 2: Uses Hour 1 prediction for lag_1h, actual data for older lags
    - Hour 3: Uses Hour 2 prediction for lag_1h, Hour 1 for lag_2h, etc.
    
    This propagates predictions forward in time.
    """
    print("🔮 Generating 72-hour recursive forecast...")
    
    predictions = []
    
    # Extract historical PM2.5 values for lag features
    historical_pm25 = df_historical['pm2_5'].tolist()
    historical_pm10 = df_historical['pm10'].tolist()
    historical_temp = df_historical['temperature_2m'].tolist()
    
    for i, row in df_weather.iterrows():
        # Combine historical + predicted values
        all_pm25 = historical_pm25 + predictions
        all_pm10 = historical_pm10 + ([predictions[-1] * 2.5] * len(predictions) if predictions else [])  # Rough PM10 estimate
        all_temp = historical_temp + [row['temperature_2m']] * len(predictions)
        
        # Build feature dictionary
        features = {}
        
        # Temporal features
        time = row['time']
        features['hour'] = time.hour
        features['day'] = time.day
        features['month'] = time.month
        features['day_of_week'] = time.dayofweek
        features['is_weekend'] = 1 if time.dayofweek >= 5 else 0
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Weather features
        features['temperature_2m'] = row['temperature_2m']
        features['relative_humidity_2m'] = row['relative_humidity_2m']
        features['pressure_msl'] = row['pressure_msl']
        features['wind_speed_10m'] = row['wind_speed_10m']
        features['wind_direction_10m'] = row['wind_direction_10m']
        
        # Lag features from combined history + predictions
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
        
        # PM10 lag features
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
        
        # Temperature lag features
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
        
        # Interaction features
        recent_pm25 = all_pm25[-1] if all_pm25 else 50  # Default value
        recent_pm10 = all_pm10[-1] if all_pm10 else 125
        
        features['pm_ratio'] = recent_pm10 / (recent_pm25 + 1e-3)
        features['temp_humid_interaction'] = row['temperature_2m'] * row['relative_humidity_2m']
        features['wind_pollution_interaction'] = row['wind_speed_10m'] * recent_pm25
        
        # Pollutant features (use predictions or historical average)
        features['pm10'] = recent_pm10
        features['co'] = df_historical['co'].mean() if 'co' in df_historical.columns else 500
        features['no2'] = df_historical['no2'].mean() if 'no2' in df_historical.columns else 40
        features['so2'] = df_historical['so2'].mean() if 'so2' in df_historical.columns else 10
        features['o3'] = df_historical['o3'].mean() if 'o3' in df_historical.columns else 60
        
        # Fill any missing features with 0
        for col in feature_cols:
            if col not in features:
                features[col] = 0
        
        # Create feature vector in correct order
        X = pd.DataFrame([features])[feature_cols]
        
        # Predict PM2.5
        pm25_pred = model.predict(X)[0]
        pm25_pred = max(0, pm25_pred)  # Ensure non-negative
        
        predictions.append(pm25_pred)
        
        # Progress indicator
        if (i + 1) % 24 == 0:
            print(f"   ✓ Generated {i + 1}/72 hour predictions")
    
    # Create forecast DataFrame
    df_forecast = df_weather[['time']].copy()
    df_forecast['pm2_5_forecast'] = predictions
    df_forecast['aqi_forecast'] = df_forecast['pm2_5_forecast'].apply(pm25_to_aqi)
    
    # Add category and emoji
    df_forecast[['aqi_category', 'emoji']] = df_forecast['aqi_forecast'].apply(
        lambda x: pd.Series(get_aqi_category(x))
    )
    
    print("✅ Forecast generation complete!")
    
    return df_forecast

def display_forecast_summary(df_forecast):
    """Display forecast summary statistics"""
    print("\n📈 FORECAST SUMMARY (Next 72 Hours)")
    print("=" * 60)
    
    # Overall statistics
    avg_pm25 = df_forecast['pm2_5_forecast'].mean()
    max_pm25 = df_forecast['pm2_5_forecast'].max()
    min_pm25 = df_forecast['pm2_5_forecast'].min()
    
    avg_aqi = df_forecast['aqi_forecast'].mean()
    max_aqi = df_forecast['aqi_forecast'].max()
    
    print(f"PM2.5 | Average: {avg_pm25:.1f} µg/m³ | Max: {max_pm25:.1f} | Min: {min_pm25:.1f}")
    print(f"AQI   | Average: {avg_aqi:.0f} | Max: {max_aqi:.0f}")
    
    # Category breakdown
    print("\nAQI Category Breakdown:")
    category_counts = df_forecast['aqi_category'].value_counts()
    for category, count in category_counts.items():
        emoji = df_forecast[df_forecast['aqi_category'] == category]['emoji'].iloc[0]
        percentage = (count / len(df_forecast)) * 100
        print(f"  {emoji} {category}: {count} hours ({percentage:.1f}%)")
    
    # Daily breakdown
    print("\nDaily Average AQI:")
    df_forecast['date'] = df_forecast['time'].dt.date
    daily_aqi = df_forecast.groupby('date')['aqi_forecast'].mean()
    for date, aqi in daily_aqi.items():
        category, emoji = get_aqi_category(aqi)
        print(f"  {date}: {aqi:.0f} {emoji} ({category})")
    
    # Show first 24 hours
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
    
    # Step 1: Load model and features
    model, feature_cols, fs, project = load_model_and_features()
    
    # Step 2: Get historical data for lag features
    df_historical = get_recent_historical_data(fs, hours=72)
    
    # Step 3: Fetch weather forecast
    df_weather = fetch_weather_forecast()
    
    # Step 4: Generate forecast
    df_forecast = generate_recursive_forecast(model, feature_cols, df_weather, df_historical)
    
    # Step 5: Display summary
    display_forecast_summary(df_forecast)
    
    # Step 6: Save forecast
    save_forecast(df_forecast, fs, project)
    
    print("\n" + "=" * 60)
    print("✅ INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
