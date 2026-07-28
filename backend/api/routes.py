"""FastAPI routes."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.api.schemas import (
    ManualLifecycleRequest,
    PredictionRequest,
    PredictionResponse,
    SavePredictionRequest,
    SavePredictionResponse,
)
from backend.pipelines.inference_pipeline import InferencePipeline
from backend.runtime import build_pipeline
from backend.utils.exceptions import (
    GEEExtractionError,
    InsufficientObservationsError,
    ModelArtifactNotFoundError,
)
from backend.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Crop Classification API is running. Open frontend/index.html to use the app.",
    }


def get_pipeline(request: Request) -> InferencePipeline:
    if not hasattr(request.app.state, "pipeline"):
        try:
            request.app.state.pipeline = build_pipeline()
            logger.info("Crop classification service ready.")
        except Exception as exc:
            logger.exception("Pipeline initialization failed.")
            raise HTTPException(status_code=503, detail=f"Pipeline initialization failed: {type(exc).__name__}: {exc}") from exc
    return request.app.state.pipeline


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, pipeline: InferencePipeline = Depends(get_pipeline)) -> PredictionResponse:
    location_id = f"{payload.latitude:.5f}_{payload.longitude:.5f}_{payload.query_date.isoformat()}"

    try:
        result = pipeline.run(
            location_id=location_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            query_date=payload.query_date,
            feature_set=payload.feature_set,
        )
    except InsufficientObservationsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GEEExtractionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelArtifactNotFoundError as exc:
        logger.error("Model artifact missing: %s", exc)
        raise HTTPException(status_code=503, detail="Prediction service is not fully configured yet.") from exc

    return PredictionResponse(
        predicted_crop=result.predicted_crop,
        confidence=result.confidence,
        is_other_crop=result.is_other_crop,
        feature_set=result.feature_set,
        lifecycle=result.lifecycle,
        plots=result.plots,
        saved_crop_id=result.saved_crop_id,
    )


@router.post("/predict/manual-lifecycle", response_model=PredictionResponse)
def predict_manual_lifecycle(
    payload: ManualLifecycleRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> PredictionResponse:
    try:
        result = pipeline.rerun_manual_lifecycle(
            dates=payload.dates,
            raw_ndvi=payload.raw_ndvi,
            smoothed_ndvi=payload.smoothed_ndvi,
            sos_date=payload.sos_date,
            eos_date=payload.eos_date,
            query_date=payload.query_date,
            feature_set=payload.feature_set,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelArtifactNotFoundError as exc:
        logger.error("Model artifact missing: %s", exc)
        raise HTTPException(status_code=503, detail="Prediction service is not fully configured yet.") from exc

    return PredictionResponse(
        predicted_crop=result.predicted_crop,
        confidence=result.confidence,
        is_other_crop=result.is_other_crop,
        feature_set=result.feature_set,
        lifecycle=result.lifecycle,
        plots=result.plots,
        saved_crop_id=result.saved_crop_id,
    )


@router.post("/predictions/save", response_model=SavePredictionResponse)
def save_prediction(
    payload: SavePredictionRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> SavePredictionResponse:
    try:
        saved_crop_id = pipeline.save_prediction_cycle(
            crop_label=payload.crop_label,
            dates=payload.dates,
            raw_ndvi=payload.raw_ndvi,
            query_date=payload.query_date,
            latitude=payload.latitude,
            longitude=payload.longitude,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Could not save prediction CSV.") from exc

    return SavePredictionResponse(saved_crop_id=saved_crop_id)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/diagnostics/runtime")
def runtime_diagnostics(request: Request) -> dict[str, object]:
    result: dict[str, object] = {
        "google_cloud_project": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "has_google_credentials_json": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")),
        "google_credentials_json_length": len(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON") or ""),
        "has_google_credentials_path": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
    }
    try:
        get_pipeline(request)
        result["pipeline"] = "ok"
    except HTTPException as exc:
        result["pipeline"] = "error"
        result["error"] = exc.detail
    return result


@router.get("/crops")
def crops(pipeline: InferencePipeline = Depends(get_pipeline)) -> dict[str, list[str]]:
    return {"crops": pipeline.settings.model.classes}
