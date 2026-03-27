# Air Quality AQI Forecast

End-to-end AQI forecasting system with Hopsworks and GitHub Actions.  
Predicts PM2.5 and AQI levels for Delhi, India up to **3 days (72 hours)** in advance using **two ML models** (XGBoost + Random Forest) whose predictions are averaged into an ensemble.

---

## ⚠️ Common CI Failure – "API key not found"

If GitHub Actions emails you about a failed run with the message:

```
hopsworks_common.client.exceptions.RestAPIError: ...
HTTP code: 401, HTTP reason: Unauthorized, body: ... "API key not found in the database"
```

**This means the `HOPSWORKS_API_KEY` secret is missing or invalid.**  
Follow the steps in the *Quick Start* section below to fix it.

---

## 🌟 Features

- **Daily Feature Collection**: Fetches weather and air quality data once per day
- **Historical Backfill**: Backfill a full year of data with one manual workflow run
- **Advanced Feature Engineering**: Lag features, rolling statistics, cyclical encoding, interaction features
- **Two ML Models**: XGBoost and Random Forest trained and compared side-by-side
- **Ensemble Forecast**: Model predictions are averaged to improve robustness
- **3-Day AQI Forecast**: Recursive 72-hour predictions every 6 hours with EPA AQI health categories
- **Automated Pipelines**: Four GitHub Actions workflows for complete automation

---

## 📊 Pipelines

### 1. Feature Pipeline (Daily – midnight UTC)
- Fetches weather data from Open-Meteo (archive API for historical, forecast API for recent)
- Fetches air quality data (PM2.5, PM10, CO, NO2, SO2, O3)
- Engineers features with lag (1 h, 3 h, 6 h, 12 h, 24 h) and rolling statistics
- Stores data in Hopsworks Feature Store
- Controlled by `BACKFILL_DAYS` env var (default `2`)

### 2. Backfill Pipeline (Manual trigger only)
- Run **once** before the first training to populate a full year of history
- Accepts an optional `backfill_days` input (default `365`)

### 3. Training Pipeline (Weekly – Sunday midnight)
- Loads all historical features from Hopsworks
- Trains **XGBoost** and **Random Forest** models
- Compares validation MAE / RMSE / R²
- Saves both models to Hopsworks Model Registry

### 4. Inference Pipeline (Every 6 hours)
- Loads both models from Hopsworks
- Fetches 72-hour weather forecast
- Generates recursive PM2.5 predictions per model
- Averages predictions into an ensemble
- Converts to AQI with health categories and **3-day daily summary**
- Saves forecast CSV as a GitHub Actions artifact

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- **Hopsworks account** (free tier available at [app.hopsworks.ai](https://app.hopsworks.ai))
- GitHub repository with secrets configured

### Step 1 – Get your Hopsworks API key

1. Sign up / log in at [app.hopsworks.ai](https://app.hopsworks.ai)
2. Open your project → **Project Settings** → **API Keys**
3. Click **New API Key**, give it a name, and copy the key value

### Step 2 – Add the secret to GitHub

1. Go to your repository on GitHub
2. **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `HOPSWORKS_API_KEY`
5. Value: paste the API key from Step 1
6. Click **Add secret**

### Step 3 – Backfill historical data (run once)

1. Go to **Actions** → **Backfill Feature Pipeline**
2. Click **Run workflow** (leave `backfill_days` at `365`)
3. Wait for completion (~5–10 minutes)

### Step 4 – Train models

1. Go to **Actions** → **Training Pipeline**
2. Click **Run workflow**
3. Wait for completion

After that, the daily and 6-hourly schedules take over automatically.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Backfill one year of data
HOPSWORKS_API_KEY=<your-key> BACKFILL_DAYS=365 python feature_pipeline.py

# Train both ML models
HOPSWORKS_API_KEY=<your-key> python training_pipeline.py

# Run 3-day AQI inference
HOPSWORKS_API_KEY=<your-key> python inference_pipeline.py
```

---

## 📈 AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0–50 | 🟢 Good | Air quality is satisfactory |
| 51–100 | 🟡 Moderate | Acceptable for most people |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | May affect sensitive individuals |
| 151–200 | 🔴 Unhealthy | Everyone may experience health effects |
| 201–300 | 🟣 Very Unhealthy | Health alert: everyone may be affected |
| 301–500 | 🟤 Hazardous | Emergency conditions |

---

## 🔧 ML Models

### XGBoost
- Gradient-boosted decision trees
- 500 estimators with early stopping (50 rounds)
- max_depth=8, learning_rate=0.05, subsample=0.8

### Random Forest
- Bagged ensemble of 300 independent decision trees
- max_depth=20, min_samples_split=5, max_features='sqrt'
- Parallel training with all CPU cores

Both models are trained on the same 30+ features, evaluated on a held-out **time-based validation set** (last 20 % of data), and their predictions are averaged for the final ensemble forecast.

### Features
- **Temporal**: hour, day, month, day_of_week, is_weekend, cyclical sin/cos encoding
- **Weather**: temperature, humidity, pressure, wind speed/direction
- **Lag Features**: PM2.5, PM10, temperature at 1 h, 3 h, 6 h, 12 h, 24 h
- **Rolling Statistics**: 24-hour mean and std of PM2.5
- **Interaction Features**: PM ratio, temp×humidity, wind×pollution
- **Pollutants**: CO, NO2, SO2, O3

---

## 📁 Repository Structure

```
.
├── feature_pipeline.py          # Daily data collection & feature engineering
├── training_pipeline.py         # Weekly model training (XGBoost + Random Forest)
├── inference_pipeline.py        # 3-day ensemble forecast generation
├── requirements.txt             # Python dependencies
├── .github/workflows/
│   ├── feature-pipeline.yml    # Daily feature collection
│   ├── backfill-pipeline.yml   # One-time historical backfill (manual)
│   ├── training-pipeline.yml   # Weekly training
│   └── inference-pipeline.yml  # 6-hourly inference
└── README.md                    # This file
```

## 🔄 Workflow Schedule

| Workflow | Schedule | Trigger |
|----------|----------|---------|
| Feature Pipeline | Daily at 00:00 UTC | `0 0 * * *` |
| Backfill Pipeline | Manual only | `workflow_dispatch` |
| Training Pipeline | Sundays at 00:00 UTC | `0 0 * * 0` |
| Inference Pipeline | Every 6 hours | `0 */6 * * *` |

All pipelines can also be triggered manually from the **GitHub Actions** tab.

## 📦 Output

The inference pipeline generates:
- `forecast_72h.csv`: CSV with hourly ensemble predictions + per-model columns
- Forecast uploaded to Hopsworks Feature Store (`pm25_forecasts`)
- GitHub Actions artifact with 30-day retention

## 🛠️ Technologies

- **ML Frameworks**: XGBoost, scikit-learn (Random Forest)
- **Feature Store**: Hopsworks
- **Data Sources**: Open-Meteo (weather & air quality, no API key required)
- **Orchestration**: GitHub Actions
- **Language**: Python 3.10

## 📝 License

MIT License – feel free to use and modify as needed.
