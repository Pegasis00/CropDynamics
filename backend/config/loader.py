"""
Central configuration loader.

Every tunable parameter in the pipeline is read from the YAML files in
configs/ — never hardcoded in the pipeline classes. Each dataclass below is
a thin, typed view over one YAML file so the rest of the codebase gets
autocomplete + type checking instead of raw dict access.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class GEEConfig:
    earth_engine_project: str | None
    collection: str
    cloud_threshold_pct: float
    months_before: int
    months_after: int
    pixel_scale_m: float
    ndvi_nir_band: str
    ndvi_red_band: str
    min_observations: int

    @classmethod
    def load(cls) -> "GEEConfig":
        raw = _load_yaml("gee.yaml")
        return cls(
            earth_engine_project=os.getenv("GOOGLE_CLOUD_PROJECT") or raw.get("earth_engine_project"),
            collection=raw["collection"],
            cloud_threshold_pct=raw["cloud_threshold_pct"],
            months_before=raw["extraction_window_months"]["before"],
            months_after=raw["extraction_window_months"]["after"],
            pixel_scale_m=raw["pixel_scale_m"],
            ndvi_nir_band=raw["ndvi_bands"]["nir"],
            ndvi_red_band=raw["ndvi_bands"]["red"],
            min_observations=raw["min_observations"],
        )


@dataclass(frozen=True)
class PipelineConfig:
    window_by_length: list[dict[str, Any]]
    polyorder: int
    min_series_length: int
    prominence_factor: float
    prominence_min: float
    prominence_max: float
    sig_multiplier: float
    growth_score_window: int
    decline_score_window: int
    stable_peak_fraction: float
    plateau_fraction: float

    @classmethod
    def load(cls) -> "PipelineConfig":
        raw = _load_yaml("pipeline.yaml")
        smoothing = raw["smoothing"]
        peaks = raw["peak_valley_detection"]
        boundary = raw["boundary_selection"]
        quality = raw["candidate_quality"]
        return cls(
            window_by_length=smoothing["window_by_length"],
            polyorder=smoothing["polyorder"],
            min_series_length=smoothing["min_series_length"],
            prominence_factor=peaks["prominence_factor"],
            prominence_min=peaks["prominence_min"],
            prominence_max=peaks["prominence_max"],
            sig_multiplier=peaks["sig_multiplier"],
            growth_score_window=boundary["growth_score_window"],
            decline_score_window=boundary["decline_score_window"],
            stable_peak_fraction=quality["stable_peak_fraction"],
            plateau_fraction=quality["plateau_fraction"],
        )


@dataclass(frozen=True)
class ThresholdsConfig:
    duration_iqr_multiplier: float
    duration_stats: dict[str, dict[str, float]]
    amplitude_floor: float
    min_ndvi_mean: float
    min_ndvi_max: float
    min_cycle_peak_ndvi: float
    min_cycle_amplitude: float

    @classmethod
    def load(cls) -> "ThresholdsConfig":
        raw = _load_yaml("thresholds.yaml")
        location = raw.get("location_validation", {})
        crop_signal = raw.get("crop_signal_validation", {})
        return cls(
            duration_iqr_multiplier=raw["duration_iqr_multiplier"],
            duration_stats=raw["duration_stats"],
            amplitude_floor=raw["validation"]["amplitude_floor"],
            min_ndvi_mean=location.get("min_ndvi_mean", 0.15),
            min_ndvi_max=location.get("min_ndvi_max", 0.25),
            min_cycle_peak_ndvi=crop_signal.get("min_peak_ndvi", 0.40),
            min_cycle_amplitude=crop_signal.get("min_amplitude", 0.12),
        )


@dataclass(frozen=True)
class ModelConfig:
    selected_algorithm: str
    selected_feature_set: str
    model_weights_dir: str
    smooth_model_file_tpl: str
    raw_model_file_tpl: str
    label_encoder_file: str
    classes: list[str]
    unknown_crop_confidence_floor: float
    enable_agreement_check: bool

    @classmethod
    def load(cls) -> "ModelConfig":
        raw = _load_yaml("model.yaml")
        naming = raw["artifact_naming"]
        return cls(
            selected_algorithm=raw["selected_algorithm"],
            selected_feature_set=raw["selected_feature_set"],
            model_weights_dir=raw["model_weights_dir"],
            smooth_model_file_tpl=naming["smooth_model_file"],
            raw_model_file_tpl=naming["raw_model_file"],
            label_encoder_file=naming["label_encoder_file"],
            classes=raw["classes"],
            unknown_crop_confidence_floor=raw["unknown_crop_confidence_floor"],
            enable_agreement_check=raw["enable_agreement_check"],
        )


@dataclass(frozen=True)
class CropRulesConfig:
    enabled: bool
    fallback_label: str
    crops: dict[str, dict[str, Any]]

    @classmethod
    def load(cls) -> "CropRulesConfig":
        raw = _load_yaml("crop_rules.yaml")
        return cls(
            enabled=raw.get("enabled", True),
            fallback_label=raw.get("fallback_label", "Other Crop"),
            crops=raw.get("crops", {}),
        )


@dataclass(frozen=True)
class Settings:
    gee: GEEConfig = field(default_factory=GEEConfig.load)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig.load)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig.load)
    model: ModelConfig = field(default_factory=ModelConfig.load)
    crop_rules: CropRulesConfig = field(default_factory=CropRulesConfig.load)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings singleton. Call get_settings.cache_clear()
    in tests if you need to reload configs after editing YAML on disk."""
    return Settings()
