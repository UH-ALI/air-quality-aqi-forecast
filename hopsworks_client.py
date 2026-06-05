"""Reusable Hopsworks integration helpers with local fallbacks."""

from __future__ import annotations

import json
import tempfile
import sys
from functools import lru_cache
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import pandas as pd

import config

try:
    import hopsworks
except ImportError:  # pragma: no cover - optional dependency
    hopsworks = None


def is_enabled() -> bool:
    return (
        config.HOPSWORKS_ENABLED
        and hopsworks is not None
        and bool(config.HOPSWORKS_API_KEY)
    )


@lru_cache(maxsize=1)
def get_project():
    if not is_enabled():
        return None

    host = config.HOPSWORKS_HOST
    if host == "app.hopsworks.ai":
        print("ℹ️ Auto-correcting Hopsworks host from 'app.hopsworks.ai' to 'c.app.hopsworks.ai' for SDK compatibility.")
        host = "c.app.hopsworks.ai"

    login_kwargs = {"api_key_value": config.HOPSWORKS_API_KEY}
    if host:
        login_kwargs["host"] = host
    if config.HOPSWORKS_PROJECT:
        login_kwargs["project"] = config.HOPSWORKS_PROJECT
    if config.HOPSWORKS_PORT:
        login_kwargs["port"] = config.HOPSWORKS_PORT

    print(f"📡 Logging in to Hopsworks project '{config.HOPSWORKS_PROJECT}' on host '{host or 'c.app.hopsworks.ai'}'...")
    return hopsworks.login(**login_kwargs)


def get_feature_store():
    project = get_project()
    return project.get_feature_store() if project is not None else None


def get_model_registry():
    project = get_project()
    return project.get_model_registry() if project is not None else None


def get_or_create_feature_group(feature_store):
    return feature_store.get_or_create_feature_group(
        name=config.HOPSWORKS_FEATURE_GROUP_NAME,
        version=config.HOPSWORKS_FEATURE_GROUP_VERSION,
        primary_key=["time"],
        event_time="time",
        description="Delhi AQI feature group with weather and pollution history.",
    )


def get_or_create_feature_view(feature_store):
    fv = None
    try:
        fv = feature_store.get_feature_view(
            name=config.HOPSWORKS_FEATURE_VIEW_NAME,
            version=config.HOPSWORKS_FEATURE_VIEW_VERSION,
        )
    except Exception as e:
        print(f"ℹ️ Feature view lookup exception: {e}. Will attempt to create one.")

    if fv is not None:
        return fv

    print("ℹ️ Feature view does not exist. Creating new feature view...")
    feature_group = get_or_create_feature_group(feature_store)
    try:
        return feature_store.create_feature_view(
            name=config.HOPSWORKS_FEATURE_VIEW_NAME,
            version=config.HOPSWORKS_FEATURE_VIEW_VERSION,
            query=feature_group.select_all(),
            labels=[config.TARGET_COLUMN],
            description="Delhi AQI training view.",
        )
    except Exception as create_exc:
        print(f"⚠️ Failed to create feature view: {create_exc}. Checking fallback lookup...")
        try:
            return feature_store.get_feature_view(
                name=config.HOPSWORKS_FEATURE_VIEW_NAME,
                version=config.HOPSWORKS_FEATURE_VIEW_VERSION,
            )
        except Exception:
            raise create_exc


def publish_feature_frame(feature_frame: pd.DataFrame):
    feature_store = get_feature_store()
    if feature_store is None:
        return None

    feature_group = get_or_create_feature_group(feature_store)
    feature_group.insert(feature_frame, write_options={"wait_for_job": True})
    return feature_group


def load_training_frame_from_hopsworks() -> pd.DataFrame | None:
    try:
        feature_store = get_feature_store()
        if feature_store is None:
            return None

        feature_view = get_or_create_feature_view(feature_store)
        
        try:
            print("ℹ️ Attempting to retrieve training data version 1...")
            features, labels = feature_view.get_training_data(1)
        except Exception:
            print("ℹ️ Version 1 not found or error. Creating new training data...")
            try:
                features, labels = feature_view.create_training_data(description="Delhi AQI training dataset")
            except Exception as create_exc:
                print(f"⚠️ create_training_data failed: {create_exc}. Falling back to reading feature group directly...")
                feature_group = get_or_create_feature_group(feature_store)
                combined = feature_group.read()
                if combined is None or combined.empty:
                    return None
                combined["time"] = pd.to_datetime(combined["time"], utc=True)
                return combined.sort_values("time").drop_duplicates("time").reset_index(drop=True)

        if labels is None:
            return None

        if isinstance(labels, pd.Series):
            labels_frame = labels.to_frame(name=config.TARGET_COLUMN)
        else:
            labels_frame = pd.DataFrame(labels).reset_index(drop=True)
            if config.TARGET_COLUMN not in labels_frame.columns and len(labels_frame.columns) == 1:
                labels_frame = labels_frame.rename(columns={labels_frame.columns[0]: config.TARGET_COLUMN})

        combined = pd.concat([features.reset_index(drop=True), labels_frame.reset_index(drop=True)], axis=1)
        if "time" not in combined.columns:
            return None

        combined["time"] = pd.to_datetime(combined["time"], utc=True)

        return combined.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    except Exception as exc:
        print(f"⚠️ Failed to load training frame from Hopsworks: {exc}")
        return None


def register_model_with_hopsworks(model, metrics: dict, best_model_name: str, feature_cols: list[str]):
    try:
        model_registry = get_model_registry()
        if model_registry is None:
            return None

        feature_store = get_feature_store()
        try:
            feature_view = get_or_create_feature_view(feature_store) if feature_store is not None else None
        except Exception as e:
            print(f"⚠️ Could not load feature view for model registration: {e}")
            feature_view = None

        create_kwargs = {
            "name": config.HOPSWORKS_MODEL_NAME,
            "metrics": metrics,
            "description": f"Champion AQI model: {best_model_name}",
        }
        
        # Only attach feature_view if it exists to avoid validation errors
        if feature_view is not None:
            create_kwargs["input_example"] = None # Avoid versioning issues
            create_kwargs["feature_view"] = feature_view

        registry_model = model_registry.sklearn.create_model(**create_kwargs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = Path(tmp_dir) / "model"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, artifact_dir / "model.pkl")
            metadata = {
                "best_model_name": best_model_name,
                "metrics": metrics,
                "feature_cols": feature_cols,
            }
            (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            registry_model.save(str(artifact_dir))

        return registry_model
    except Exception as exc:
        print(f"⚠️ Failed to register model with Hopsworks: {exc}")
        return None


def download_latest_model_from_hopsworks():
    model_registry = get_model_registry()
    if model_registry is None:
        return None, None

    models = model_registry.get_models(config.HOPSWORKS_MODEL_NAME)
    if not models:
        return None, None

    latest_model = max(models, key=lambda item: getattr(item, "version", 0))
    model_dir = Path(latest_model.download())
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        candidates = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
        if candidates:
            model_path = candidates[0]

    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return joblib.load(model_path), metadata


def download_recent_historical_data_from_hopsworks(hours: int = 72) -> pd.DataFrame | None:
    """Download the most recent historical records from the Hopsworks Feature Group."""
    if not is_enabled():
        return None

    try:
        feature_store = get_feature_store()
        if feature_store is None:
            return None

        feature_group = get_or_create_feature_group(feature_store)
        df = feature_group.read()
        if df is None or df.empty:
            return None

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        return df.tail(hours)
    except Exception as exc:
        print(f"⚠️ Failed to download recent historical data from Hopsworks: {exc}")
        return None