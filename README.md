# Air Quality AQI Forecast

End-to-end AQI forecasting system with Hopsworks and GitHub Actions. Predicts PM2.5 and AQI levels for Delhi, India up to 72 hours in advance.

## 🌟 Features

- **Hourly Feature Collection**: Fetches weather and air quality data every hour
- **Advanced Feature Engineering**: Lag features, rolling statistics, cyclical encoding, and interaction features
- **Weekly Model Training**: XGBoost model trained on historical data with time-based validation
- **72-Hour AQI Forecasts**: Recursive predictions every 6 hours with EPA AQI health categories
- **Automated Pipelines**: Three separate GitHub Actions workflows for complete automation

## 📊 Pipelines

### 1. Feature Pipeline (Hourly)
- Fetches latest weather data from Open-Meteo
- Fetches air quality data (PM2.5, PM10, CO, NO2, SO2, O3)
- Engineers features with lag (1h, 3h, 6h, 12h, 24h) and rolling statistics
- Stores in Hopsworks Feature Store

### 2. Training Pipeline (Weekly - Sunday midnight)
- Loads all historical features from Hopsworks
- Trains XGBoost regression model
- Time-based split (80% train, 20% validation)
- Saves model to Hopsworks Model Registry
- Reports MAE, RMSE, and R² metrics

### 3. Inference Pipeline (Every 6 hours)
- Loads trained model from Hopsworks
- Fetches 72-hour weather forecast
- Generates recursive PM2.5 predictions
- Converts to AQI with health categories
- Saves forecast to CSV and Hopsworks

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Hopsworks account with API key
- GitHub repository with secrets configured

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Set the `HOPSWORKS_API_KEY` environment variable or GitHub secret:

```bash
export HOPSWORKS_API_KEY="your-api-key-here"
```

### Running Locally

```bash
# Run feature pipeline
python feature_pipeline.py

# Run training pipeline (requires historical data)
python training_pipeline.py

# Run inference pipeline (requires trained model)
python inference_pipeline.py
```

## 📈 AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0-50 | 🟢 Good | Air quality is satisfactory |
| 51-100 | 🟡 Moderate | Acceptable for most people |
| 101-150 | 🟠 Unhealthy for Sensitive Groups | May affect sensitive individuals |
| 151-200 | 🔴 Unhealthy | Everyone may experience health effects |
| 201-300 | 🟣 Very Unhealthy | Health alert: everyone may be affected |
| 301-500 | 🟤 Hazardous | Emergency conditions |

## 🔧 Model Features

The XGBoost model uses 30+ features including:
- **Temporal**: hour, day, month, day_of_week, is_weekend, cyclical encoding
- **Weather**: temperature, humidity, pressure, wind speed/direction
- **Lag Features**: PM2.5, PM10, temperature at 1h, 3h, 6h, 12h, 24h intervals
- **Rolling Statistics**: 24-hour mean and std deviation of PM2.5
- **Interaction Features**: PM ratio, temp-humidity, wind-pollution interactions
- **Pollutants**: CO, NO2, SO2, O3

## 📁 Repository Structure

```
.
├── feature_pipeline.py      # Hourly data collection & feature engineering
├── training_pipeline.py     # Weekly model training
├── inference_pipeline.py    # 72-hour forecast generation
├── requirements.txt         # Python dependencies
├── .github/workflows/
│   └── pipeline.yml        # GitHub Actions configuration
└── README.md               # This file
```

## 🔄 Workflow Schedule

- **Feature Pipeline**: Every hour (`0 * * * *`)
- **Training Pipeline**: Every Sunday at midnight UTC (`0 0 * * 0`)
- **Inference Pipeline**: Every 6 hours (`0 */6 * * *`)

All pipelines can also be triggered manually via `workflow_dispatch`.

## 📦 Output

The inference pipeline generates:
- `forecast_72h.csv`: CSV file with hourly predictions
- Forecast uploaded to Hopsworks Feature Store
- GitHub Actions artifact with 30-day retention

## 🛠️ Technologies

- **ML Framework**: XGBoost
- **Feature Store**: Hopsworks
- **Data Sources**: Open-Meteo (weather & air quality)
- **Orchestration**: GitHub Actions
- **Language**: Python 3.10

## 📝 License

MIT License - feel free to use and modify as needed.
