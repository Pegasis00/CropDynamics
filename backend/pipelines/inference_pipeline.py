"""
InferencePipeline — orchestrates the exact sequence from the spec for one
live (lat, lon, query_date) request:

  1. Retrieve NDVI (GEE)
  2. Build location record (Stage B)
  3. Adaptive Savitzky-Golay smoothing (Stage C)
  4. Robust noise estimation (Stage C)
  5. Peak detection (Stage C)
  6. Valley detection (Stage C)
  7. SOS/EOS boundary selection (Stage D)
  8. Candidate lifecycle generation (Stage D)
  9. Candidate quality metrics (Stage D)
  10. Candidate validation (Stage D)
  11. Best-candidate selection (Stage D)
  12. Feature extraction on raw + smooth curves (Stage G)
  13. Run the selected model on both feature vectors' matching set
  14. (No blending — single selected model only, per spec)
  15. Final crop prediction + confidence
  16. Visualization payload
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.config.loader import Settings
from backend.exporters.crop_ndvi_writer import CropNDVIWriter
from backend.features.extractor import FeatureExtractor
from backend.gee.extractor import GEEDataExtractor
from backend.lifecycle.boundary import BoundarySelector
from backend.lifecycle.candidates import CandidateGenerator
from backend.lifecycle.farm_builder import FarmBuilder, LocationRecord
from backend.lifecycle.selector import LifecycleSelector
from backend.lifecycle.validator import CandidateValidator
from backend.models.prediction_service import PredictionResult, PredictionService
from backend.models.registry import ModelRegistry
from backend.signal_processing.peaks import PeakDetector, ValleyDetector
from backend.signal_processing.smoothing import SignalProcessor
from backend.utils.exceptions import (
    InsufficientObservationsError,
)
from backend.utils.logging import get_logger
from backend.visualization.plots import PlotGenerator

logger = get_logger(__name__)
NO_PLANT_LABEL = "No plant found"


@dataclass
class InferenceResult:
    predicted_crop: str
    confidence: float
    is_other_crop: bool
    lifecycle: Optional[dict[str, Any]]
    plots: dict[str, Any]
    saved_crop_id: Optional[str] = None
    feature_set: str = "smooth"


class InferencePipeline:
    def __init__(
        self,
        settings: Settings,
        gee_extractor: GEEDataExtractor,
        model_registry: ModelRegistry,
        crop_ndvi_writer: Optional[CropNDVIWriter] = None,
    ):
        self.settings = settings
        self.gee_extractor = gee_extractor
        self.farm_builder = FarmBuilder()
        self.signal_processor = SignalProcessor(settings.pipeline)
        self.peak_detector = PeakDetector(settings.pipeline, self.signal_processor)
        self.valley_detector = ValleyDetector(settings.pipeline, self.signal_processor)
        self.boundary_selector = BoundarySelector(settings.pipeline)
        self.candidate_generator = CandidateGenerator(self.boundary_selector, self.signal_processor)
        self.candidate_validator = CandidateValidator(settings.thresholds)
        self.lifecycle_selector = LifecycleSelector()
        self.feature_extractor = FeatureExtractor(
            stable_peak_fraction=settings.pipeline.stable_peak_fraction,
            plateau_fraction=settings.pipeline.plateau_fraction,
        )
        self.prediction_service = PredictionService(settings.model, model_registry)
        self.plot_generator = PlotGenerator()
        self.crop_ndvi_writer = crop_ndvi_writer

    def run(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        query_date: dt.date,
        feature_set: str = "smooth",
    ) -> InferenceResult:
        # Step 1 — NDVI retrieval
        observations = self.gee_extractor.extract(latitude, longitude, query_date)

        if len(observations) < self.settings.gee.min_observations:
            raise InsufficientObservationsError()

        # Step 2 — build location record
        record = self.farm_builder.build(location_id, latitude, longitude, query_date, observations)

        # Step 3 — smoothing
        record.smoothed = self.signal_processor.adaptive_smoothing(record.ndvi)
        lifecycle_signal = record.ndvi if feature_set == "raw" else record.smoothed

        # Step 4-5 — peak detection (also yields the noise estimate)
        peaks, _, noise = self.peak_detector.detect(record.ndvi, lifecycle_signal)
        record.peaks = peaks
        record.noise_estimate = noise

        # Step 6 — valley detection
        valleys, _, _ = self.valley_detector.detect(record.ndvi, lifecycle_signal)
        record.valleys = valleys

        if not self._is_valid_location(record):
            plots = self._build_plots(record, cycle=None, query_date=query_date, feature_set=feature_set)
            return InferenceResult(
                predicted_crop="Not agricultural land",
                confidence=0.0,
                is_other_crop=True,
                lifecycle=None,
                plots=plots,
                feature_set=feature_set,
            )

        # Steps 7-8 — candidate generation (internally does SOS/EOS boundary selection)
        candidates = self.candidate_generator.generate_candidates(record, feature_set=feature_set)

        # Step 9 — quality metrics
        candidates = [self.candidate_generator.compute_quality_metrics(c, feature_set=feature_set) for c in candidates]

        # Step 10 — validation
        candidates = [self.candidate_validator.validate_candidate(c, mode="inference") for c in candidates]
        record.candidate_cycles = candidates

        # Step 11 — best-candidate selection
        cycle = self.lifecycle_selector.select(record)

        if cycle is None:
            plots = self._build_plots(record, cycle=None, query_date=query_date, feature_set=feature_set)
            return InferenceResult(
                predicted_crop="No valid crop lifecycle",
                confidence=0.0,
                is_other_crop=True,
                lifecycle=None,
                plots=plots,
                feature_set=feature_set,
            )

        if not self._is_valid_crop_signal(cycle, feature_set):
            plots = self._build_plots(record, cycle=None, query_date=query_date, feature_set=feature_set)
            return InferenceResult(
                predicted_crop=NO_PLANT_LABEL,
                confidence=0.0,
                is_other_crop=True,
                lifecycle=None,
                plots=plots,
                feature_set=feature_set,
            )

        # Step 12 — feature extraction, raw + smooth
        raw_features = self.feature_extractor.extract(cycle, "raw_curve")
        smooth_features = self.feature_extractor.extract(cycle, "smooth_curve")

        # Steps 13-15 — single selected-model inference + confidence
        prediction: PredictionResult = self.prediction_service.predict(raw_features, smooth_features, feature_set=feature_set)
        prediction = self._apply_post_prediction_rules(prediction, cycle)

        # Step 16 — visualization payload
        plots = self._build_plots(record, cycle=cycle, query_date=query_date, feature_set=feature_set)
        return InferenceResult(
            predicted_crop=prediction.predicted_crop,
            confidence=prediction.confidence,
            is_other_crop=prediction.is_other_crop,
            feature_set=feature_set,
            lifecycle={
                "sos_date": str(cycle["sos_date"].date()),
                "peak_date": str(cycle["peak_date"].date()),
                "eos_date": str(cycle["eos_date"].date()),
                "duration_days": cycle["duration"],
            },
            plots=plots,
            saved_crop_id=None,
        )

    def save_prediction_cycle(
        self,
        crop_label: str,
        dates: list[str],
        raw_ndvi: list[float],
        query_date: dt.date,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> str:
        if not self.crop_ndvi_writer:
            raise ValueError("CSV saving is not configured.")
        if len(dates) != len(raw_ndvi):
            raise ValueError("dates and raw_ndvi must have the same length.")

        cycle = {
            "dates": pd.to_datetime(dates).tolist(),
            "raw_curve": [float(v) for v in raw_ndvi],
        }

        try:
            crop_id = self.crop_ndvi_writer.append_cycle(
                crop_label,
                cycle,
                query_date,
                latitude=latitude,
                longitude=longitude,
            )
        except OSError:
            logger.exception("Failed to save predicted crop NDVI cycle for %s", crop_label)
            raise

        if not crop_id:
            raise ValueError("Only accepted trained crop predictions can be saved.")
        return crop_id

    def rerun_manual_lifecycle(
        self,
        dates: list[str],
        raw_ndvi: list[float],
        smoothed_ndvi: list[float],
        sos_date: dt.date,
        eos_date: dt.date,
        query_date: dt.date,
        feature_set: str = "smooth",
    ) -> InferenceResult:
        all_dates = pd.to_datetime(dates)
        raw = np.asarray(raw_ndvi, dtype=float)
        smooth = np.asarray(smoothed_ndvi, dtype=float)

        if len(all_dates) != len(raw) or len(all_dates) != len(smooth):
            raise ValueError("dates, raw_ndvi, and smoothed_ndvi must have the same length.")

        date_keys = [d.date().isoformat() for d in all_dates]
        sos_key = sos_date.isoformat()
        eos_key = eos_date.isoformat()
        if sos_key not in date_keys or eos_key not in date_keys:
            raise ValueError("SOS and EOS must be selected from available NDVI observation dates.")

        left_idx = date_keys.index(sos_key)
        right_idx = date_keys.index(eos_key)
        if right_idx <= left_idx:
            raise ValueError("EOS date must be after SOS date.")

        local_raw = raw[left_idx : right_idx + 1]
        local_smooth = smooth[left_idx : right_idx + 1]
        selected_curve = local_smooth if feature_set == "smooth" else local_raw
        if len(selected_curve) < 2:
            raise ValueError("Manual lifecycle must include at least two NDVI observations.")

        peak_idx = left_idx + int(np.nanargmax(selected_curve))
        cycle_dates = all_dates[left_idx : right_idx + 1]
        cycle = {
            "query_date": query_date,
            "left_idx": left_idx,
            "peak_idx": peak_idx,
            "right_idx": right_idx,
            "sos_date": pd.Timestamp(all_dates[left_idx]),
            "peak_date": pd.Timestamp(all_dates[peak_idx]),
            "eos_date": pd.Timestamp(all_dates[right_idx]),
            "sos_ndvi_raw": float(raw[left_idx]),
            "peak_ndvi_raw": float(raw[peak_idx]),
            "eos_ndvi_raw": float(raw[right_idx]),
            "sos_ndvi_smooth": float(smooth[left_idx]),
            "peak_ndvi_smooth": float(smooth[peak_idx]),
            "eos_ndvi_smooth": float(smooth[right_idx]),
            "raw_curve": local_raw.tolist(),
            "smooth_curve": local_smooth.tolist(),
            "dates": cycle_dates.tolist(),
            "duration": int((cycle_dates[-1] - cycle_dates[0]) / pd.Timedelta(days=1)),
            "noise_estimate": self.signal_processor.estimate_noise(
                raw[left_idx : right_idx + 1],
                local_smooth,
            ),
            "status": "Manual",
            "reason": None,
        }
        cycle = self.candidate_generator.compute_quality_metrics(cycle, feature_set=feature_set)

        if not self._is_valid_crop_signal(cycle, feature_set):
            record = LocationRecord(
                location_id="manual_lifecycle",
                latitude=0.0,
                longitude=0.0,
                query_date=query_date,
                dates=all_dates,
                ndvi=raw,
                smoothed=smooth,
            )
            plots = self._build_plots(record, cycle=None, query_date=query_date, feature_set=feature_set)
            return InferenceResult(
                predicted_crop=NO_PLANT_LABEL,
                confidence=0.0,
                is_other_crop=True,
                lifecycle=None,
                plots=plots,
                saved_crop_id=None,
                feature_set=feature_set,
            )

        raw_features = self.feature_extractor.extract(cycle, "raw_curve")
        smooth_features = self.feature_extractor.extract(cycle, "smooth_curve")
        prediction: PredictionResult = self.prediction_service.predict(raw_features, smooth_features, feature_set=feature_set)
        prediction = self._apply_post_prediction_rules(prediction, cycle)

        record = LocationRecord(
            location_id="manual_lifecycle",
            latitude=0.0,
            longitude=0.0,
            query_date=query_date,
            dates=all_dates,
            ndvi=raw,
            smoothed=smooth,
        )
        plots = self._build_plots(record, cycle=cycle, query_date=query_date, feature_set=feature_set)

        return InferenceResult(
            predicted_crop=prediction.predicted_crop,
            confidence=prediction.confidence,
            is_other_crop=prediction.is_other_crop,
            feature_set=feature_set,
            lifecycle={
                "sos_date": str(cycle["sos_date"].date()),
                "peak_date": str(cycle["peak_date"].date()),
                "eos_date": str(cycle["eos_date"].date()),
                "duration_days": cycle["duration"],
            },
            plots=plots,
            saved_crop_id=None,
        )

    def _build_plots(
        self,
        record: LocationRecord,
        cycle: Optional[dict[str, Any]],
        query_date: dt.date,
        feature_set: str = "smooth",
    ) -> dict[str, Any]:
        return {
            "raw_vs_smoothed": self.plot_generator.raw_vs_smoothed(record, query_date),
            "signal_with_markers": self.plot_generator.signal_with_markers(record, query_date, feature_set=feature_set),
            "selected_lifecycle": self.plot_generator.selected_lifecycle(record, cycle, query_date, feature_set=feature_set),
        }

    def _is_valid_location(self, record: LocationRecord) -> bool:
        cfg = self.settings.thresholds
        mean_ndvi = float(np.nanmean(record.ndvi))
        max_ndvi = float(np.nanmax(record.ndvi))

        return not (mean_ndvi < cfg.min_ndvi_mean and max_ndvi < cfg.min_ndvi_max)

    def _is_valid_crop_signal(self, cycle: dict[str, Any], feature_set: str) -> bool:
        cfg = self.settings.thresholds
        peak_key = "peak_ndvi_raw" if feature_set == "raw" else "peak_ndvi_smooth"
        peak_ndvi = float(cycle.get(peak_key, np.nan))
        amplitude = float(cycle.get("amplitude", np.nan))

        if not np.isfinite(peak_ndvi) or not np.isfinite(amplitude):
            return False
        return peak_ndvi >= cfg.min_cycle_peak_ndvi and amplitude >= cfg.min_cycle_amplitude

    def _apply_post_prediction_rules(
        self, prediction: PredictionResult, cycle: dict[str, Any]
    ) -> PredictionResult:
        rules_cfg = self.settings.crop_rules
        if not rules_cfg.enabled or prediction.is_other_crop:
            return prediction

        crop_rules = rules_cfg.crops.get(prediction.predicted_crop, {})
        if not crop_rules:
            return prediction

        metrics = {
            "duration_days": cycle.get("duration"),
            "amplitude": cycle.get("amplitude"),
            "baseline_ndvi": cycle.get("baseline_ndvi"),
            "peak_ndvi_raw": cycle.get("peak_ndvi_raw"),
            "peak_ndvi_smooth": cycle.get("peak_ndvi_smooth"),
            "sos_ndvi_raw": cycle.get("sos_ndvi_raw"),
            "sos_ndvi_smooth": cycle.get("sos_ndvi_smooth"),
            "eos_ndvi_raw": cycle.get("eos_ndvi_raw"),
            "eos_ndvi_smooth": cycle.get("eos_ndvi_smooth"),
            "signal_to_noise": cycle.get("signal_to_noise"),
        }
        month_metrics = {
            "sos_month": self._cycle_month(cycle, "sos_date"),
            "start_month": self._cycle_month(cycle, "sos_date"),
            "peak_month": self._cycle_month(cycle, "peak_date"),
            "eos_month": self._cycle_month(cycle, "eos_date"),
            "end_month": self._cycle_month(cycle, "eos_date"),
            "query_month": self._cycle_month(cycle, "query_date"),
        }

        for metric_name, metric_rules in crop_rules.items():
            if metric_name in month_metrics:
                value = month_metrics[metric_name]
                passes = value is not None and self._month_rule_passes(int(value), metric_rules)
            else:
                value = metrics.get(metric_name)
                passes = value is not None and self._numeric_rule_passes(float(value), metric_rules)

            if not passes:
                logger.info(
                    "Post-model crop rule rejected %s: %s=%s did not satisfy %s",
                    prediction.predicted_crop,
                    metric_name,
                    value,
                    metric_rules,
                )
                return PredictionResult(
                    predicted_crop=rules_cfg.fallback_label,
                    confidence=prediction.confidence,
                    class_probabilities=prediction.class_probabilities,
                    is_other_crop=True,
                )

        return prediction

    @staticmethod
    def _cycle_month(cycle: dict[str, Any], key: str) -> int | None:
        value = cycle.get(key)
        if value is None:
            return None
        return int(pd.Timestamp(value).month)

    @staticmethod
    def _month_rule_passes(value: int, rules: dict[str, Any]) -> bool:
        allowed = rules.get("allowed")
        if allowed is not None and value not in {int(month) for month in allowed}:
            return False
        return InferencePipeline._numeric_rule_passes(float(value), rules)

    @staticmethod
    def _numeric_rule_passes(value: float, rules: dict[str, Any]) -> bool:
        if "min" in rules and value < float(rules["min"]):
            return False
        if "min_exclusive" in rules and value <= float(rules["min_exclusive"]):
            return False
        if "max" in rules and value > float(rules["max"]):
            return False
        if "max_exclusive" in rules and value >= float(rules["max_exclusive"]):
            return False
        return True
