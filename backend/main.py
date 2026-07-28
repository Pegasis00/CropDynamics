"""
FastAPI application entrypoint.

Wires together: Settings (all YAML configs) -> GEEDataExtractor ->
ModelRegistry (loads model_weights/*.joblib) -> InferencePipeline, then
exposes it via backend.api.routes.

Run with:
    uvicorn backend.main:app --reload --port 8000

Earth Engine auth: set GOOGLE_APPLICATION_CREDENTIALS to a service-account
key with Earth Engine access before starting the app, or run
`earthengine authenticate` locally for interactive use.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.config.loader import GEEConfig
from backend.config.loader import get_settings
from backend.exporters.crop_ndvi_writer import CropNDVIWriter
from backend.gee.extractor import GEEDataExtractor
from backend.models.registry import ModelRegistry
from backend.pipelines.inference_pipeline import InferencePipeline
from backend.utils.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


def _configure_google_credentials_from_env() -> None:
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    credentials_path = Path(tempfile.gettempdir()) / "google-application-credentials.json"
    credentials_path.write_text(credentials_json, encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)


def _init_earth_engine(config: GEEConfig):
    import ee

    _configure_google_credentials_from_env()

    try:
        if config.earth_engine_project:
            ee.Initialize(project=config.earth_engine_project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine could not initialize. Run `earthengine authenticate` "
            "once in a terminal, and set `earth_engine_project` in "
            "configs/gee.yaml or GOOGLE_CLOUD_PROJECT before starting the API."
        ) from exc
    return ee


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    ee_module = _init_earth_engine(settings.gee)
    gee_extractor = GEEDataExtractor(settings.gee, ee_module=ee_module)

    model_registry = ModelRegistry(settings.model, base_dir=BASE_DIR)
    model_registry.load_all()

    crop_ndvi_writer = CropNDVIWriter(BASE_DIR / "saved_crop_ndvi")
    app.state.pipeline = InferencePipeline(settings, gee_extractor, model_registry, crop_ndvi_writer)
    logger.info("Crop classification service ready.")
    yield


app = FastAPI(title="Crop Classification API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Crop Classification API is running. Open frontend/index.html to use the app.",
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
