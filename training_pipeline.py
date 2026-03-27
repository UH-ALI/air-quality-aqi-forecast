"""
training_pipeline.py

Purpose:
    - Load historical features from Hopsworks (past year + daily updates)
    - Train two ML models for PM2.5 prediction:
        1. XGBoost (gradient-boosted trees)
        2. Random Forest (bagged decision trees)
    - Compare model performance and select the best one
    - Save both models + the best-model selection to Hopsworks Model Registry

Training Strategy:
    - Uses ALL historical data stored in Hopsworks
    - Time-based split (80% train, 20% validation)
    - Early stopping for XGBoost to prevent overfitting
    - Saves models + feature names for inference
"""

import hopsworks
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os
import json

def load_features_from_hopsworks():
    """
    Load ALL historical features from Hopsworks Feature Store
    This includes: past year data + all daily hourly updates
    """
    print("🗄️ Connecting to Hopsworks...")
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
    fs = project.get_feature_store()

    # Get feature group (contains all historical data)
    fg = fs.get_feature_group(name="pm25_features_v2", version=1)

    # Read ALL data (no filters - gets everything)
    df = fg.read()

    print(f"✅ Loaded {len(df)} rows from feature store")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"   Total days: {(df['time'].max() - df['time'].min()).days}")

    return df, fs, project

def prepare_training_data(df):
    """
    Prepare features and target for model training
    Uses time-based split to respect temporal ordering
    """
    print("🔧 Preparing training data...")
    
    # Sort by time (CRITICAL for time series)
    df = df.sort_values('time').reset_index(drop=True)
    
    # Define target and features
    target_col = 'pm2_5'
    
    # Exclude non-feature columns
    exclude_cols = ['time', 'pm2_5']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Time-based split (last 20% for validation)
    # This prevents data leakage - we never train on future data
    split_idx = int(len(df) * 0.8)
    
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    print(f"✅ Training data prepared:")
    print(f"   Train samples: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"   Validation samples: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"   Number of features: {len(feature_cols)}")
    print(f"   Features: {feature_cols[:10]}... (showing first 10)")

    return X_train, X_val, y_train, y_val, feature_cols


def _compute_metrics(model_name, y_train, y_train_pred, y_val, y_val_pred):
    """Return a dict of performance metrics and print a summary."""
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n📊 {model_name} Performance:")
    print(f"   TRAIN | MAE: {train_mae:.2f} µg/m³ | RMSE: {train_rmse:.2f} µg/m³ | R²: {train_r2:.4f}")
    print(f"   VAL   | MAE: {val_mae:.2f} µg/m³ | RMSE: {val_rmse:.2f} µg/m³ | R²: {val_r2:.4f}")

    return {
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "train_r2": float(train_r2),
        "val_r2": float(val_r2),
    }


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost regression model for PM2.5 prediction.

    Hyperparameters:
    - n_estimators=500: up to 500 trees (early stopping cuts this short)
    - max_depth=8: tree depth balances expressiveness vs overfitting
    - learning_rate=0.05: conservative step size for stability
    - subsample/colsample_bytree=0.8: stochastic sampling reduces variance
    - early_stopping_rounds=50: halts training if val loss doesn't improve
    """
    print("\n🤖 Training XGBoost model...")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        objective='reg:squarederror',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    print("\n🔝 XGBoost – Top 10 Features:")
    print(importance_df.head(10).to_string(index=False))

    metrics = _compute_metrics("XGBoost", y_train, y_train_pred, y_val, y_val_pred)
    return model, metrics


def train_random_forest_model(X_train, y_train, X_val, y_val):
    """
    Train a Random Forest regression model for PM2.5 prediction.

    Random Forest is an ensemble of independent decision trees (bagging).
    It is robust to outliers and provides reliable feature importance estimates.

    Hyperparameters:
    - n_estimators=300: 300 independent trees for stable predictions
    - max_depth=20: allows deep trees since bagging controls variance
    - min_samples_split=5: avoids over-partitioning small leaves
    - max_features='sqrt': standard heuristic for regression RF
    - n_jobs=-1: use all CPU cores for parallel training
    """
    print("\n🌲 Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    print("\n🔝 Random Forest – Top 10 Features:")
    print(importance_df.head(10).to_string(index=False))

    metrics = _compute_metrics("Random Forest", y_train, y_train_pred, y_val, y_val_pred)
    return model, metrics


def _save_single_model(model, model_filename, registry_name, metrics, feature_cols, mr, model_dir):
    """Persist one model to disk and upload it to the Hopsworks Model Registry."""
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"   ✓ Saved {registry_name} → {model_path}")

    registry_model = mr.python.create_model(
        name=registry_name,
        metrics=metrics,
        description=f"{registry_name} model for PM2.5 3-day forecasting (Delhi).",
    )
    registry_model.save(model_dir)
    print(f"   ✓ Uploaded to Hopsworks as '{registry_name}' v{registry_model.version}")
    return registry_model


def save_models_to_hopsworks(
    xgb_model, xgb_metrics,
    rf_model, rf_metrics,
    feature_cols, project
):
    """
    Save both models to the Hopsworks Model Registry together with:
    - Feature column list (required by inference pipeline)
    - Individual metrics for each model
    - A comparison summary indicating which model is best
    """
    print("\n💾 Saving models to Hopsworks Model Registry...")

    model_dir = "pm25_model"
    os.makedirs(model_dir, exist_ok=True)

    # Determine the best model by validation MAE (lower is better)
    if xgb_metrics["val_mae"] <= rf_metrics["val_mae"]:
        best_name = "pm25_xgboost"
        print(f"\n🏆 Best model: XGBoost (val MAE {xgb_metrics['val_mae']:.2f} vs RF {rf_metrics['val_mae']:.2f})")
    else:
        best_name = "pm25_rf"
        print(f"\n🏆 Best model: Random Forest (val MAE {rf_metrics['val_mae']:.2f} vs XGB {xgb_metrics['val_mae']:.2f})")

    # Save feature names (shared by both models)
    feature_path = os.path.join(model_dir, "features.json")
    with open(feature_path, "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)

    # Save comparison summary
    comparison = {
        "xgboost": xgb_metrics,
        "random_forest": rf_metrics,
        "best_model": best_name,
    }
    with open(os.path.join(model_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    mr = project.get_model_registry()

    # --- Upload XGBoost ---
    # Write model-specific metrics file before uploading; avoids overwriting RF metrics.
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(xgb_metrics, f, indent=2)
    _save_single_model(
        xgb_model, "xgboost_model.pkl", "pm25_xgboost",
        xgb_metrics, feature_cols, mr, model_dir,
    )

    # --- Upload Random Forest ---
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(rf_metrics, f, indent=2)
    _save_single_model(
        rf_model, "rf_model.pkl", "pm25_rf",
        rf_metrics, feature_cols, mr, model_dir,
    )

    print("\n✅ Both models uploaded to Hopsworks Model Registry!")
    print(f"   Best model for inference: {best_name}")
    return best_name


def main():
    print("=" * 60)
    print("🚀 PM2.5 FORECASTING MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load historical data
    df, fs, project = load_features_from_hopsworks()

    # Step 2: Prepare training data
    X_train, X_val, y_train, y_val, feature_cols = prepare_training_data(df)

    # Step 3a: Train XGBoost
    xgb_model, xgb_metrics = train_xgboost_model(X_train, y_train, X_val, y_val)

    # Step 3b: Train Random Forest
    rf_model, rf_metrics = train_random_forest_model(X_train, y_train, X_val, y_val)

    # Step 4: Print model comparison
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<18} {'XGBoost':>12} {'RandomForest':>14}")
    print("-" * 46)
    for key in ("val_mae", "val_rmse", "val_r2"):
        label = key.replace("val_", "Val ").upper()
        print(f"{label:<18} {xgb_metrics[key]:>12.4f} {rf_metrics[key]:>14.4f}")

    # Step 5: Save both models to Hopsworks
    save_models_to_hopsworks(
        xgb_model, xgb_metrics,
        rf_model, rf_metrics,
        feature_cols, project,
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
