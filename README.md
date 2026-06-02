# Air Quality AQI Forecast

AQI forecasting system for Delhi, India with Hopsworks as the feature store and model registry.
The pipelines still keep local Parquet/joblib fallbacks for development and validation, but the primary runtime path now publishes features to Hopsworks and registers the champion model there.

## What it does

- Backfills one year of hourly weather history from Open-Meteo.
- Uses OpenWeather air pollution history for the one-year pollution backfill when `OPENWEATHER_API_KEY` is set.
- Falls back to the recent Open-Meteo air-quality window when OpenWeather is unavailable.
- Engineers lag, rolling, cyclical, and interaction features.
- Publishes the engineered feature frame to a Hopsworks feature group.
- Materializes training data from a Hopsworks feature view and compares multiple models.
- Registers the champion model in the Hopsworks model registry.
- Downloads the latest registered model for 72-hour forecast generation.

## Repository Layout

```
.
├── app.py                   # Streamlit dashboard
├── check_openmeteo.py       # Open-Meteo range verification script
├── config.py                # Centralized paths, Hopsworks names, and settings
├── data_sources.py          # Historical and forecast data access helpers
├── feature_pipeline.py      # Historical backfill and feature publishing
├── hopsworks_client.py      # Hopsworks login, feature store, and registry helpers
├── inference_pipeline.py    # 72-hour recursive forecast generation
├── training_pipeline.py     # Model comparison and registry publishing
├── requirements.txt         # Python dependencies
├── .github/workflows/       # GitHub Actions automation
└── README.md
```

## Setup

### Prerequisites

- Python 3.10 or newer
- A Hopsworks project, API key, and optional host/project/port overrides
- OpenWeather API key for the full-year pollution history

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Set these before running the pipelines locally or as GitHub repository variables in GitHub Actions. For local development, you can copy [.env.example](.env.example) to `.env` and fill in the values there:

```powershell
$env:HOPSWORKS_API_KEY="your-hopsworks-api-key"
$env:HOPSWORKS_HOST="your-hopsworks-host"
$env:HOPSWORKS_PROJECT="your-hopsworks-project"
$env:HOPSWORKS_PORT="443"
$env:OPENWEATHER_API_KEY="your-openweather-api-key"
$env:POLLUTION_HISTORY_SOURCE="openweather"
```

If `OPENWEATHER_API_KEY` is not set, the feature pipeline falls back to the recent Open-Meteo air-quality history window.
If the Hopsworks variables are missing, the code falls back to local artifacts so the project can still be validated offline.

For local runs, copy [.env.example](.env.example) to [.env](.env) and fill in the values you want to keep on your machine. The code loads `.env` automatically if it exists.

## Run the Pipeline

1. Verify Open-Meteo coverage:

```bash
python check_openmeteo.py
```

2. Build the historical backfill and publish the feature frame:

```bash
python feature_pipeline.py
```

3. Train and compare models, then register the champion model:

```bash
python training_pipeline.py
```

4. Generate the 72-hour forecast using the latest registry model:

```bash
python inference_pipeline.py
```

5. Launch the dashboard:

```bash
streamlit run app.py
```

## Outputs

The pipelines keep these local artifacts for convenience and fallback:

- `data/historical_raw.parquet`
- `data/historical_features.parquet`
- `forecast_72h.csv`
- `models/best_model.joblib`
- `reports/model_comparison.csv`
- `reports/feature_importance.csv`
- `reports/shap_summary.png` when SHAP generation is available

## Model Comparison

The training pipeline compares:

- Persistence baseline
- Ridge regression
- Random Forest regression
- XGBoost when the package is available in the active Python environment

The best model is registered in Hopsworks and also kept locally as a fallback bundle.

## GitHub Actions

Canonical workflow files:

- [feature-pipeline.yml](.github/workflows/feature-pipeline.yml)
- [training-pipeline.yml](.github/workflows/training-pipeline.yml)
- [inference-pipeline.yml](.github/workflows/inference-pipeline.yml)

Suggested triggers:

- Feature pipeline: hourly and on push to `main`
- Training pipeline: weekly on Sunday at midnight UTC
- Inference pipeline: every 6 hours

Required GitHub variables or secrets:

- `HOPSWORKS_API_KEY`
- `HOPSWORKS_HOST` and `HOPSWORKS_PROJECT` if your Hopsworks deployment requires them
- `OPENWEATHER_API_KEY` for the one-year pollution backfill

The workflows read these from GitHub Variables first and fall back to Secrets if needed.

## AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0-50 | 🟢 Good | Air quality is satisfactory |
| 51-100 | 🟡 Moderate | Acceptable for most people |
| 101-150 | 🟠 Unhealthy for Sensitive Groups | May affect sensitive individuals |
| 151-200 | 🔴 Unhealthy | Everyone may experience health effects |
| 201-300 | 🟣 Very Unhealthy | Health alert: everyone may be affected |
| 301-500 | 🟤 Hazardous | Emergency conditions |

## Technology Stack

- Python
- pandas, numpy, scikit-learn, joblib, pyarrow
- Hopsworks feature store and model registry
- Open-Meteo APIs
- OpenWeather pollution history API
- Streamlit and Plotly for the dashboard
- GitHub Actions for automation

## Notes

- The project now uses Hopsworks as the primary feature store and model registry path.
- Local parquet and joblib artifacts remain as a fallback path for validation and offline development.
- The inference pipeline expects a trained model in Hopsworks or a local fallback bundle plus the historical raw data file.

## License

MIT License - feel free to use and modify as needed.