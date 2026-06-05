"""Training pipeline that compares multiple models on local feature history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from hopsworks_client import load_training_frame_from_hopsworks, register_model_with_hopsworks

try:
    import shap
except ImportError:  # pragma: no cover - environment dependent
    shap = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - environment dependent
    XGBRegressor = None


class PersistenceBaseline:
    """Simple last-observation baseline for comparison."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame):
        if "pm2_5_lag_1h" not in X.columns:
            raise KeyError("pm2_5_lag_1h is required for the persistence baseline")
        return X["pm2_5_lag_1h"].to_numpy()


def ensure_output_dirs() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> pd.DataFrame:
    hopsworks_df = load_training_frame_from_hopsworks()
    if hopsworks_df is not None and not hopsworks_df.empty:
        hopsworks_df["time"] = pd.to_datetime(hopsworks_df["time"], utc=True)
        print(
            f"Loaded {len(hopsworks_df)} training rows from Hopsworks feature view "
            f"spanning {hopsworks_df['time'].min()} to {hopsworks_df['time'].max()}"
        )
        return hopsworks_df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    if not config.FEATURE_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature history at {config.FEATURE_HISTORY_PATH}. Run feature_pipeline.py first."
        )

    df = pd.read_parquet(config.FEATURE_HISTORY_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def prepare_training_data(df: pd.DataFrame):
    target_col = config.TARGET_COLUMN
    feature_cols = [col for col in df.columns if col not in {"time", target_col}]

    feature_frame = df[feature_cols].copy()
    target = df[target_col].copy()

    split_idx = int(len(df) * 0.8)
    X_train = feature_frame.iloc[:split_idx].reset_index(drop=True)
    X_val = feature_frame.iloc[split_idx:].reset_index(drop=True)
    y_train = target.iloc[:split_idx].reset_index(drop=True)
    y_val = target.iloc[split_idx:].reset_index(drop=True)

    return X_train, X_val, y_train, y_val, feature_cols


def fit_and_score(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "val_mae": float(mean_absolute_error(y_val, val_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
        "train_r2": float(r2_score(y_train, train_pred)),
        "val_r2": float(r2_score(y_val, val_pred)),
    }
    return model, metrics


def build_models():
    models = {
        "persistence": PersistenceBaseline(),
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

    return models


def save_feature_importance(model, feature_cols):
    if isinstance(model, Pipeline):
        estimator = model[-1]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_)
    else:
        return None

    importance_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)
    return importance_df


def save_shap_plot(model, X_val: pd.DataFrame):
    if shap is None:
        print("⚠️ SHAP is not available in this environment; skipping explanation plot.")
        return None

    estimator = model[-1] if isinstance(model, Pipeline) else model
    if not hasattr(estimator, "predict"):
        return None

    try:
        explainer = shap.Explainer(estimator, X_val)
        shap_values = explainer(X_val)
        plt.figure()
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(config.SHAP_PLOT_PATH, dpi=200, bbox_inches="tight")
        plt.close()
        return config.SHAP_PLOT_PATH
    except Exception as exc:
        print(f"⚠️ SHAP plot skipped: {exc}")
        return None


def train_models(X_train, X_val, y_train, y_val):
    results = []
    trained_models = {}

    for name, model in build_models().items():
        print(f"🤖 Training {name}...")
        trained_model, metrics = fit_and_score(model, X_train, y_train, X_val, y_val)
        trained_models[name] = trained_model
        results.append({"model": name, **metrics})
        print(
            f"   VAL | MAE: {metrics['val_mae']:.2f} | RMSE: {metrics['val_rmse']:.2f} | R²: {metrics['val_r2']:.4f}"
        )

    comparison_df = pd.DataFrame(results).sort_values("val_rmse").reset_index(drop=True)
    comparison_df.to_csv(config.MODEL_COMPARISON_PATH, index=False)

    best_row = comparison_df.iloc[0]
    best_name = best_row["model"]
    best_model = trained_models[best_name]

    return best_name, best_model, comparison_df, trained_models


def save_bundle(best_name, best_model, feature_cols, comparison_df, trained_models):
    bundle = {
        "best_model_name": best_name,
        "model": best_model,
        "feature_cols": feature_cols,
        "comparison": comparison_df.to_dict(orient="records"),
    }
    joblib.dump(bundle, config.MODEL_BUNDLE_PATH)
    return config.MODEL_BUNDLE_PATH


def main():
    parser = argparse.ArgumentParser(description="Train and compare AQI models.")
    parser.parse_args()

    ensure_output_dirs()
    print("=" * 60)
    print("🚀 MODEL COMPARISON TRAINING PIPELINE")
    print("=" * 60)

    df = load_features()
    print(f"Loaded {len(df)} feature rows spanning {df['time'].min()} to {df['time'].max()}")

    X_train, X_val, y_train, y_val, feature_cols = prepare_training_data(df)
    print(f"Train rows: {len(X_train)} | Validation rows: {len(X_val)} | Features: {len(feature_cols)}")

    best_name, best_model, comparison_df, trained_models = train_models(X_train, X_val, y_train, y_val)
    best_metrics = comparison_df.iloc[0].to_dict()
    registry_metrics = {key: value for key, value in best_metrics.items() if key != "model"}

    importance_source = "xgboost" if "xgboost" in trained_models else best_name
    save_feature_importance(trained_models[importance_source], feature_cols)

    if "xgboost" in trained_models:
        save_shap_plot(trained_models["xgboost"], X_val)
    registry_model = register_model_with_hopsworks(best_model, registry_metrics, best_name, feature_cols)
    bundle_path = save_bundle(best_name, best_model, feature_cols, comparison_df, trained_models)

    print("\nModel comparison results:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    if registry_model is not None:
        print(f"Hopsworks model registry: {config.HOPSWORKS_MODEL_NAME} v{registry_model.version}")
    print(f"Saved bundle: {bundle_path}")
    print(f"Comparison CSV: {config.MODEL_COMPARISON_PATH}")
    print(f"Feature importance CSV: {config.FEATURE_IMPORTANCE_PATH}")
    print(f"SHAP plot: {config.SHAP_PLOT_PATH}")

    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
