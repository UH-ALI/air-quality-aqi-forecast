"""
training_pipeline.py

Purpose:
    - Load historical features from Hopsworks (past year + daily updates)
    - Train XGBoost model for PM2.5 prediction
    - Evaluate model performance
    - Save model to Hopsworks Model Registry

Training Strategy:
    - Uses ALL historical data stored in Hopsworks
    - Time-based split (80% train, 20% validation)
    - Early stopping to prevent overfitting
    - Saves model + feature names for inference
"""

import hopsworks
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        raise ValueError("HOPSWORKS_API_KEY not found in environment variables")
    
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

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost regression model for PM2.5 prediction
    
    Model hyperparameters explained:
    - n_estimators: Number of trees (500 is good for medium datasets)
    - max_depth: Maximum tree depth (8 prevents overfitting)
    - learning_rate: Step size (0.05 is conservative but stable)
    - subsample: Fraction of samples per tree (0.8 adds randomness)
    - colsample_bytree: Fraction of features per tree (0.8 prevents overfitting)
    - early_stopping_rounds: Stop if no improvement for 50 rounds
    """
    print("🤖 Training XGBoost model...")
    
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        objective='reg:squarederror'
    )
    
    # Train with validation monitoring
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50  # Print progress every 50 rounds
    )
    
    # Evaluate on validation set
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   TRAIN | MAE: {train_mae:.2f} µg/m³ | RMSE: {train_rmse:.2f} µg/m³ | R²: {train_r2:.4f}")
    print(f"   VAL   | MAE: {val_mae:.2f} µg/m³ | RMSE: {val_rmse:.2f} µg/m³ | R²: {val_r2:.4f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    print(f"\n🔝 Top 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10).to_string(index=False))
    
    metrics = {
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "train_r2": float(train_r2),
        "val_r2": float(val_r2)
    }
    
    return model, metrics

def save_model_to_hopsworks(model, metrics, feature_cols, project):
    """
    Save trained model to Hopsworks Model Registry
    
    Saves:
    1. Model binary (XGBoost model)
    2. Feature names (for inference)
    3. Metrics (for tracking performance)
    """
    print("💾 Saving model to Hopsworks Model Registry...")
    
    # Create local directory
    model_dir = "pm25_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = f"{model_dir}/xgboost_model.pkl"
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved to {model_path}")
    
    # Save feature names (CRITICAL for inference)
    feature_path = f"{model_dir}/features.json"
    with open(feature_path, "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)
    print(f"   ✓ Feature names saved to {feature_path}")
    
    # Save metrics
    metrics_path = f"{model_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics saved to {metrics_path}")
    
    # Upload to Hopsworks Model Registry
    mr = project.get_model_registry()
    
    pm25_model = mr.python.create_model(
        name="pm25_xgboost",
        metrics=metrics,
        description="XGBoost model for PM2.5 forecasting (next 72 hours). Trained on historical data with lag features."
    )
    
    pm25_model.save(model_dir)
    
    print("✅ Model successfully uploaded to Hopsworks Model Registry!")
    print(f"   Model name: pm25_xgboost")
    print(f"   Version: {pm25_model.version}")

def main():
    print("=" * 60)
    print("🚀 PM2.5 FORECASTING MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load historical data
    df, fs, project = load_features_from_hopsworks()
    
    # Step 2: Prepare training data
    X_train, X_val, y_train, y_val, feature_cols = prepare_training_data(df)
    
    # Step 3: Train model
    model, metrics = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Step 4: Save model to Hopsworks
    save_model_to_hopsworks(model, metrics, feature_cols, project)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
