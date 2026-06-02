"""Streamlit dashboard for the AQI forecast and model comparison."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import config

# --- Page Configuration ---
st.set_page_config(
    page_title="Delhi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
)

forecast_path = config.FORECAST_OUTPUT_PATH
comparison_path = config.MODEL_COMPARISON_PATH
shap_path = config.SHAP_PLOT_PATH

# --- Load Data ---
try:
    df_forecast = pd.read_csv(forecast_path)
    df_forecast["time"] = pd.to_datetime(df_forecast["time"])
except FileNotFoundError:
    st.error(f"{forecast_path} not found. Please run the inference pipeline first.")
    st.stop()

# --- Header ---
st.title("🌫️ Delhi AQI Forecast")
st.markdown("### Air Quality Index (AQI) Forecast for the Next 3 Days")

if comparison_path.exists():
    comparison_df = pd.read_csv(comparison_path)
    best_model = comparison_df.iloc[0]["model"]
    st.caption(f"Best model from training: {best_model}")

# --- Main Dashboard ---
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average AQI (Next 72h)",
        f"{df_forecast['aqi_forecast'].mean():.0f}",
        f"{df_forecast['aqi_forecast'].max() - df_forecast['aqi_forecast'].min():.0f} (Range)",
    )

with col2:
    max_aqi_row = df_forecast.loc[df_forecast["aqi_forecast"].idxmax()]
    st.metric(
        "Peak AQI",
        f"{max_aqi_row['aqi_forecast']:.0f} ({max_aqi_row['aqi_category']})",
        f"at {max_aqi_row['time'].strftime('%A, %I %p')}",
    )

# --- AQI Forecast Chart ---
fig = px.line(
    df_forecast,
    x="time",
    y="aqi_forecast",
    title="72-Hour AQI Forecast",
    labels={"time": "Time", "aqi_forecast": "AQI"},
    hover_data=["aqi_category", "pm2_5_forecast"],
)
fig.update_traces(mode="lines+markers")
st.plotly_chart(fig, use_container_width=True)

# --- Hazardous AQI Alert ---
hazardous_hours = df_forecast[df_forecast["aqi_forecast"] > 150]
if not hazardous_hours.empty:
    st.warning(
        f"**Hazardous AQI Alert!** "
        f"The AQI is forecasted to be unhealthy for {len(hazardous_hours)} hours. "
        f"Peak AQI will be {hazardous_hours['aqi_forecast'].max():.0f}."
    )

# --- Detailed Forecast Table ---
st.subheader("Hourly Forecast Details")
st.dataframe(
    df_forecast[["time", "pm2_5_forecast", "aqi_forecast", "aqi_category"]].style.format(
        {"pm2_5_forecast": "{:.1f}", "aqi_forecast": "{:.0f}"}
    )
)

# --- Model Comparison ---
if comparison_path.exists():
    st.subheader("Model Comparison")
    st.dataframe(pd.read_csv(comparison_path), use_container_width=True)

# --- Feature Importance ---
st.subheader("Model Feature Importance")
if shap_path.exists():
    st.image(str(shap_path), use_container_width=True)
else:
    st.info(f"{shap_path} not found. Run the training pipeline to generate it.")

# --- AQI Categories Explained ---
st.sidebar.header("AQI Categories")
st.sidebar.markdown(
    """
| AQI Range | Health Concern |
|---|---|
| 0-50 | Good |
| 51-100 | Moderate |
| 101-150 | Unhealthy for Sensitive Groups |
| 151-200 | Unhealthy |
| 201-300 | Very Unhealthy |
| 301-500 | Hazardous |
"""
)
