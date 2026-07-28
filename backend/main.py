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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.runtime import FRONTEND_DIST
from backend.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Crop Classification API", version="1.0.0")


@app.get("/")
def root():
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "ok", "message": "Crop Classification API is running."}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
