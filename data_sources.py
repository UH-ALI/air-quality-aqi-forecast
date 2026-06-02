"""Data access helpers for historical backfill and live forecast retrieval."""

from __future__ import annotations

import datetime as dt
import time
from typing import Iterable, Iterator

import pandas as pd
import requests

import config


OPENMETEO_WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENWEATHER_AIR_POLLUTION_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

WEATHER_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
)

POLLUTION_COLUMNS = (
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
)


def _request_json(url: str, params: dict, retries: int = 3, backoff_factor: int = 5) -> dict:
    """Make a GET request to a URL and return the JSON response, with retries."""
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict) and payload.get("error"):
                reason = payload.get("reason", "Unknown API error")
                raise RuntimeError(f"API request failed: {reason}")

            return payload
        except (requests.exceptions.RequestException, RuntimeError) as e:
            if i == retries - 1:
                raise  # Re-raise the last exception if all retries fail

            print(f"⚠️ Request failed (attempt {i + 1}/{retries}): {e}. Retrying in {backoff_factor}s...")
            time.sleep(backoff_factor)

    raise RuntimeError("Request failed after multiple retries")


def _chunk_range(start: dt.datetime, end: dt.datetime, chunk_days: int) -> Iterator[tuple[dt.datetime, dt.datetime]]:
    current = start
    step = dt.timedelta(days=chunk_days)

    while current <= end:
        chunk_end = min(current + step - dt.timedelta(hours=1), end)
        yield current, chunk_end
        current = chunk_end + dt.timedelta(hours=1)


def _to_utc_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def fetch_openmeteo_weather_archive(start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Fetch hourly historical weather data from Open-Meteo."""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(WEATHER_HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    payload = _request_json(OPENMETEO_WEATHER_ARCHIVE_URL, params)
    hourly = payload.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty:
        return df

    df["time"] = _to_utc_timestamp(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def fetch_openmeteo_weather_forecast(horizon_hours: int = config.FORECAST_HORIZON_HOURS) -> pd.DataFrame:
    """Fetch hourly weather forecast from Open-Meteo."""
    forecast_days = max(1, (horizon_hours + 23) // 24)
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(WEATHER_HOURLY_VARIABLES),
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    payload = _request_json(OPENMETEO_WEATHER_FORECAST_URL, params)
    hourly = payload.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty:
        return df

    df["time"] = _to_utc_timestamp(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df.head(horizon_hours).copy()


def fetch_openweather_air_pollution_history(start: dt.datetime, end: dt.datetime, api_key: str) -> pd.DataFrame:
    """Fetch hourly historical air pollution data from OpenWeather."""
    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "start": int(start.replace(tzinfo=dt.timezone.utc).timestamp()),
        "end": int(end.replace(tzinfo=dt.timezone.utc).timestamp()),
        "appid": api_key,
    }

    payload = _request_json(OPENWEATHER_AIR_POLLUTION_HISTORY_URL, params)
    rows = []

    for record in payload.get("list", []):
        components = record.get("components", {})
        rows.append(
            {
                "time": pd.to_datetime(record["dt"], unit="s", utc=True),
                "pm10": components.get("pm10"),
                "pm2_5": components.get("pm2_5"),
                "carbon_monoxide": components.get("co"),
                "nitrogen_dioxide": components.get("no2"),
                "sulphur_dioxide": components.get("so2"),
                "ozone": components.get("o3"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def fetch_openmeteo_air_quality_recent(days: int) -> pd.DataFrame:
    """Fetch recent air quality history from Open-Meteo for a limited lookback window."""
    days = max(1, min(days, config.OPENMETEO_POLLUTION_LOOKBACK_DAYS))
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(POLLUTION_COLUMNS),
        "past_days": days,
        "timezone": "UTC",
    }

    payload = _request_json(OPENMETEO_AIR_QUALITY_URL, params)
    hourly = payload.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty:
        return df

    df["time"] = _to_utc_timestamp(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def merge_weather_and_pollution(weather_df: pd.DataFrame, pollution_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather and pollution data on timestamp."""
    weather_df = weather_df.copy()
    pollution_df = pollution_df.copy()

    if weather_df.empty or pollution_df.empty:
        return pd.DataFrame()

    weather_df["time"] = _to_utc_timestamp(weather_df["time"])
    pollution_df["time"] = _to_utc_timestamp(pollution_df["time"])

    merged = pd.merge(weather_df, pollution_df, on="time", how="outer", suffixes=("_weather", "_pollution"))
    merged = merged.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return merged


def backfill_history(start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Backfill a historical dataset using Open-Meteo weather and OpenWeather pollution."""
    weather_frames = []
    for chunk_start, chunk_end in _chunk_range(start, end, config.HISTORICAL_CHUNK_DAYS):
        weather_frames.append(fetch_openmeteo_weather_archive(chunk_start, chunk_end))

    weather_df = pd.concat(weather_frames, ignore_index=True).drop_duplicates("time").sort_values("time")

    pollution_frames = []
    if config.POLLUTION_HISTORY_SOURCE == "openmeteo":
        lookback_start = max(start, end - dt.timedelta(days=config.OPENMETEO_POLLUTION_LOOKBACK_DAYS))
        if lookback_start > start:
            print(
                f"⚠️ Open-Meteo pollution history is limited to about {config.OPENMETEO_POLLUTION_LOOKBACK_DAYS} days. "
                f"Only the most recent window from {lookback_start.date()} onward will be used."
            )
        pollution_df = fetch_openmeteo_air_quality_recent((end - lookback_start).days)
    else:
        if not config.OPENWEATHER_API_KEY:
            raise RuntimeError(
                "OPENWEATHER_API_KEY is required for a 1-year pollution backfill. Set it in your environment or switch POLLUTION_HISTORY_SOURCE to openmeteo for a shorter backfill."
            )

        for chunk_start, chunk_end in _chunk_range(start, end, config.HISTORICAL_CHUNK_DAYS):
            pollution_frames.append(
                fetch_openweather_air_pollution_history(chunk_start, chunk_end, config.OPENWEATHER_API_KEY)
            )

        pollution_df = pd.concat(pollution_frames, ignore_index=True).drop_duplicates("time").sort_values("time")

    merged = merge_weather_and_pollution(weather_df, pollution_df)
    return merged