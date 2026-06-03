"""Configuration for the AQI prediction system."""

from __future__ import annotations

import os
from pathlib import Path

try:
	from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
	load_dotenv = None

if load_dotenv is not None:
	load_dotenv()


WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_ROOT / "data"
MODELS_DIR = WORKSPACE_ROOT / "models"
REPORTS_DIR = WORKSPACE_ROOT / "reports"


def _load_local_env_file(env_path: Path) -> None:
	if not env_path.exists():
		return

	for raw_line in env_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue
		if line.startswith("export "):
			line = line[len("export "):].strip()
		if "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if not key:
			continue

		if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
			value = value[1:-1]

		os.environ.setdefault(key, value)


_load_local_env_file(WORKSPACE_ROOT / ".env")

# Geographic coordinates for the location of interest.
CITY_NAME = "Delhi"
LATITUDE = 28.6139
LONGITUDE = 77.2090

# Data sources and backfill settings.
HISTORICAL_BACKFILL_DAYS = int(os.getenv("HISTORICAL_BACKFILL_DAYS", "365"))
HISTORICAL_CHUNK_DAYS = int(os.getenv("HISTORICAL_CHUNK_DAYS", "15"))
POLLUTION_HISTORY_SOURCE = os.getenv("POLLUTION_HISTORY_SOURCE", "openweather").lower()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
OPENMETEO_POLLUTION_LOOKBACK_DAYS = int(os.getenv("OPENMETEO_POLLUTION_LOOKBACK_DAYS", "92"))

# Hopsworks settings.
HOPSWORKS_ENABLED = os.getenv("HOPSWORKS_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "").strip()
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "").strip()
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "predicting_aqi").strip()
# HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT", "443"))
_port_str = os.getenv("HOPSWORKS_PORT", "443").strip()
HOPSWORKS_PORT = int(_port_str) if _port_str else 443
HOPSWORKS_FEATURE_GROUP_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP_NAME", "delhi_aqi_features")
HOPSWORKS_FEATURE_GROUP_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))
HOPSWORKS_FEATURE_VIEW_NAME = os.getenv("HOPSWORKS_FEATURE_VIEW_NAME", "delhi_aqi_feature_view")
HOPSWORKS_FEATURE_VIEW_VERSION = int(os.getenv("HOPSWORKS_FEATURE_VIEW_VERSION", "1"))
HOPSWORKS_MODEL_NAME = os.getenv("HOPSWORKS_MODEL_NAME", "delhi_aqi_model")

# Feature engineering settings.
TARGET_COLUMN = "pm2_5"
FEATURE_LAGS = (1, 3, 6, 12, 24, 48, 72)
ROLLING_WINDOWS = (24, 48, 72)

# Forecast settings.
FORECAST_HORIZON_HOURS = int(os.getenv("FORECAST_HORIZON_HOURS", "72"))

# File paths.
RAW_HISTORY_PATH = DATA_DIR / "historical_raw.parquet"
FEATURE_HISTORY_PATH = DATA_DIR / "historical_features.parquet"
FORECAST_OUTPUT_PATH = WORKSPACE_ROOT / "forecast_72h.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "best_model.joblib"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
SHAP_PLOT_PATH = REPORTS_DIR / "shap_summary.png"
