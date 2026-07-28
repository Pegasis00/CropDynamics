from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from backend.config.loader import GEEConfig
from backend.config.loader import get_settings
from backend.exporters.crop_ndvi_writer import CropNDVIWriter
from backend.gee.extractor import GEEDataExtractor
from backend.models.registry import ModelRegistry
from backend.pipelines.inference_pipeline import InferencePipeline

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


def _configure_google_credentials_from_env() -> None:
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    credentials_path = Path(tempfile.gettempdir()) / "google-application-credentials.json"
    credentials_path.write_text(credentials_json, encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)


def _service_account_credentials(ee):
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if credentials_json:
        credentials_data = json.loads(credentials_json)
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            credentials_path = str(Path(tempfile.gettempdir()) / "google-application-credentials.json")
            Path(credentials_path).write_text(credentials_json, encoding="utf-8")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        return ee.ServiceAccountCredentials(credentials_data["client_email"], credentials_path)

    if credentials_path and Path(credentials_path).exists():
        with open(credentials_path, "r", encoding="utf-8") as f:
            credentials_data = json.load(f)
        if credentials_data.get("type") == "service_account":
            return ee.ServiceAccountCredentials(credentials_data["client_email"], credentials_path)

    return None


def _init_earth_engine(config: GEEConfig):
    import ee

    _configure_google_credentials_from_env()

    try:
        credentials = _service_account_credentials(ee)
        if config.earth_engine_project:
            ee.Initialize(credentials=credentials, project=config.earth_engine_project)
        else:
            ee.Initialize(credentials=credentials)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine could not initialize. Run `earthengine authenticate` "
            "once in a terminal, and set `earth_engine_project` in "
            "configs/gee.yaml or GOOGLE_CLOUD_PROJECT before starting the API."
        ) from exc
    return ee


def build_pipeline() -> InferencePipeline:
    settings = get_settings()
    ee_module = _init_earth_engine(settings.gee)
    gee_extractor = GEEDataExtractor(settings.gee, ee_module=ee_module)

    model_registry = ModelRegistry(settings.model, base_dir=BASE_DIR)
    model_registry.load_all()

    crop_ndvi_writer = CropNDVIWriter(BASE_DIR / "saved_crop_ndvi")
    return InferencePipeline(settings, gee_extractor, model_registry, crop_ndvi_writer)
