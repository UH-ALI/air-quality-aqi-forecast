"""Reusable Hopsworks integration helpers with local fallbacks."""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

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

    login_kwargs = {"api_key_value": config.HOPSWORKS_API_KEY}
    if config.HOPSWORKS_HOST:
        login_kwargs["host"] = config.HOPSWORKS_HOST
    if config.HOPSWORKS_PROJECT:
        login_kwargs["project"] = config.HOPSWORKS_PROJECT
    if config.HOPSWORKS_PORT and str(config.HOPSWORKS_PORT).isdigit():
        login_kwargs["port"] = config.HOPSWORKS_PORT

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
    try:
        return feature_store.get_feature_view(
            name=config.HOPSWORKS_FEATURE_VIEW_NAME,
            version=config.HOPSWORKS_FEATURE_VIEW_VERSION,
        )
    except Exception:
        feature_group = get_or_create_feature_group(feature_store)
        return feature_store.create_feature_view(
            name=config.HOPSWORKS_FEATURE_VIEW_NAME,
            version=config.HOPSWORKS_FEATURE_VIEW_VERSION,
            query=feature_group.select_all(),
            labels=[config.TARGET_COLUMN],
            description="Delhi AQI training view.",
        )


def publish_feature_frame(feature_frame: pd.DataFrame):
    feature_store = get_feature_store()
    if feature_store is None:
        return None

    feature_group = get_or_create_feature_group(feature_store)
    feature_group.insert(feature_frame, write_options={"wait_for_job": True})
    return feature_group


def load_training_frame_from_hopsworks() -> pd.DataFrame | None:
    feature_store = get_feature_store()
    if feature_store is None:
        return None

    feature_view = get_or_create_feature_view(feature_store)
    features, labels = feature_view.create_training_data()

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


def register_model_with_hopsworks(model, metrics: dict, best_model_name: str, feature_cols: list[str]):
    model_registry = get_model_registry()
    if model_registry is None:
        return None

    feature_store = get_feature_store()
    feature_view = get_or_create_feature_view(feature_store) if feature_store is not None else None

    registry_model = model_registry.sklearn.create_model(
        name=config.HOPSWORKS_MODEL_NAME,
        metrics=metrics,
        description=f"Champion AQI model: {best_model_name}",
        feature_view=feature_view,
        training_dataset_version=1,
    )

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