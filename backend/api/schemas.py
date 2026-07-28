"""Pydantic request/response models for the crop-classification API."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Query latitude, e.g. 18.5204")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Query longitude, e.g. 73.8567")
    coordinates: Optional[str] = Field(None, description="Optional pasted latitude/longitude pair, e.g. 18.5204, 73.8567")
    query_date: dt.date = Field(..., description="Date to center the NDVI extraction window on, e.g. 2024-10-15")
    feature_set: Literal["smooth", "raw"] = "smooth"

    @model_validator(mode="before")
    @classmethod
    def split_pasted_coordinates(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("latitude") is not None or data.get("longitude") is not None:
            return data

        coordinates = data.get("coordinates")
        if not coordinates:
            return data

        parts = re.findall(r"[-+]?\d+(?:\.\d+)?", str(coordinates))
        if len(parts) >= 2:
            data = data.copy()
            data["latitude"] = float(parts[0])
            data["longitude"] = float(parts[1])
        return data

    @model_validator(mode="after")
    def require_coordinates(self) -> "PredictionRequest":
        if self.latitude is None or self.longitude is None:
            raise ValueError("Provide latitude and longitude, or paste them together as coordinates.")
        return self


class ManualLifecycleRequest(BaseModel):
    dates: list[str] = Field(..., min_length=2)
    raw_ndvi: list[float] = Field(..., min_length=2)
    smoothed_ndvi: list[float] = Field(..., min_length=2)
    sos_date: dt.date
    eos_date: dt.date
    query_date: dt.date
    feature_set: Literal["smooth", "raw"] = "smooth"

    @model_validator(mode="after")
    def require_matching_series(self) -> "ManualLifecycleRequest":
        if not (len(self.dates) == len(self.raw_ndvi) == len(self.smoothed_ndvi)):
            raise ValueError("dates, raw_ndvi, and smoothed_ndvi must have the same length.")
        if self.eos_date <= self.sos_date:
            raise ValueError("EOS date must be after SOS date.")
        return self


class SavePredictionRequest(BaseModel):
    crop_label: str
    dates: list[str] = Field(..., min_length=1)
    raw_ndvi: list[float] = Field(..., min_length=1)
    query_date: dt.date
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    @model_validator(mode="after")
    def require_matching_cycle(self) -> "SavePredictionRequest":
        if len(self.dates) != len(self.raw_ndvi):
            raise ValueError("dates and raw_ndvi must have the same length.")
        return self


class LifecycleSummary(BaseModel):
    sos_date: str
    peak_date: str
    eos_date: str
    duration_days: int


class PredictionResponse(BaseModel):
    predicted_crop: str
    confidence: float
    is_other_crop: bool
    feature_set: Literal["smooth", "raw"] = "smooth"
    lifecycle: Optional[LifecycleSummary]
    plots: dict[str, Any]
    saved_crop_id: Optional[str] = None


class SavePredictionResponse(BaseModel):
    saved_crop_id: str


class ErrorResponse(BaseModel):
    detail: str
