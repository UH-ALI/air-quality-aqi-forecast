"""Small verification script for Open-Meteo historical coverage."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests


LATITUDE = 28.6139
LONGITUDE = 77.2090


def summarize_hourly_series(name: str, times: pd.DatetimeIndex, start: str, end: str):
    expected = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    gaps = (times.to_series().diff() > pd.Timedelta("1h")).sum()

    print(f"\n{name}")
    print("-" * len(name))
    print("first:", times.min())
    print("last:", times.max())
    print("got:", len(times))
    print("expected:", len(expected))
    print("missing:", max(0, len(expected) - len(times)))
    print("gaps:", int(gaps))


def verify_weather_archive(start_date: str, end_date: str):
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m",
            "timezone": "UTC",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    times = pd.to_datetime(payload["hourly"]["time"], utc=True)
    summarize_hourly_series("Weather archive", times, f"{start_date} 00:00", f"{end_date} 23:00")


def verify_air_quality_recent(past_days: int):
    response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "past_days": past_days,
            "timezone": "UTC",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    times = pd.to_datetime(payload["hourly"]["time"], utc=True)
    print(f"\nAir quality recent window ({past_days} days)")
    print("-" * 40)
    print("first:", times.min())
    print("last:", times.max())
    print("got:", len(times))
    print("approx days covered:", round(len(times) / 24, 1))


if __name__ == "__main__":
    end = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=364)

    verify_weather_archive(start.isoformat(), end.isoformat())
    verify_air_quality_recent(92)