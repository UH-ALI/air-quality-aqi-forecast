"""Feature pipeline for historical backfill and feature generation."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import config
from data_sources import backfill_history
from hopsworks_client import publish_feature_frame


def ensure_output_dirs() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def create_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = config.FEATURE_LAGS) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)

    for lag in lags:
        df[f"pm2_5_lag_{lag}h"] = df["pm2_5"].shift(lag)
        df[f"pm10_lag_{lag}h"] = df["pm10"].shift(lag)
        df[f"temp_lag_{lag}h"] = df["temperature_2m"].shift(lag)

    for window in config.ROLLING_WINDOWS:
        rolling = df["pm2_5"].rolling(window=window, min_periods=window)
        df[f"pm2_5_rolling_mean_{window}h"] = rolling.mean()
        df[f"pm2_5_rolling_std_{window}h"] = rolling.std()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["pm_ratio"] = df["pm10"] / (df["pm2_5"] + 1e-3)
    df["temp_humid_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df["wind_pollution_interaction"] = df["wind_speed_10m"] * df["pm2_5"]

    df = create_lag_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


def build_backfill_window(days: int, end_date: str | None = None) -> tuple[dt.datetime, dt.datetime]:
    if end_date:
        end = pd.Timestamp(end_date, tz="UTC").to_pydatetime()
    else:
        end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        end = end - dt.timedelta(hours=1)

    start = end - dt.timedelta(days=days - 1)
    return start, end


def save_outputs(raw_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    ensure_output_dirs()
    raw_df.to_parquet(config.RAW_HISTORY_PATH, index=False)
    feature_df.to_parquet(config.FEATURE_HISTORY_PATH, index=False)
    feature_df.to_csv(config.DATA_DIR / "historical_features.csv", index=False)
    print(f"✅ Saved raw history to {config.RAW_HISTORY_PATH}")
    print(f"✅ Saved feature history to {config.FEATURE_HISTORY_PATH}")


def publish_to_hopsworks(feature_df: pd.DataFrame) -> None:
    try:
        feature_group = publish_feature_frame(feature_df)
        if feature_group is None:
            print("⚠️ Hopsworks not configured — data saved locally only.")
            return
        print(f"✅ Published {len(feature_df)} rows to Hopsworks.")
    except Exception as e:
        print(f"⚠️ Hopsworks publish failed: {e}")
        print("✅ Data is saved locally. Fix Hopsworks credentials and re-run publish separately.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill and build historical AQI features.")
    parser.add_argument("--days", type=int, default=config.HISTORICAL_BACKFILL_DAYS)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    start, end = build_backfill_window(args.days, args.end_date)
    print("=" * 60)
    print("🌫️  HISTORICAL BACKFILL PIPELINE")
    print("=" * 60)
    print(f"Building backfill window: {start.isoformat()} -> {end.isoformat()}")

    raw_df = backfill_history(start, end)
    if raw_df.empty:
        raise RuntimeError("No historical data was returned from the data sources.")

    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    raw_df = raw_df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    feature_df = engineer_features(raw_df)

    print(f"Raw rows: {len(raw_df)}")
    print(f"Feature rows after lag drop: {len(feature_df)}")
    print(f"Feature columns: {len(feature_df.columns)}")

    save_outputs(raw_df, feature_df)
    publish_to_hopsworks(feature_df)

    print("✅ Historical backfill completed successfully")


if __name__ == "__main__":
    main()
