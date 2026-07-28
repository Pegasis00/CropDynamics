"""
CandidateGenerator — Stage D of the reference notebook.

build_candidate(): one candidate lifecycle per detected peak, using the
boundary selector's best-SOS/best-EOS logic.

compute_quality_metrics(): baseline, amplitude, residuals, SNR — computed
once per candidate right after generation, exactly as in the notebook.

Two notebook-only fields, `crop` and `ground_truth`, only exist during
offline training (where they come from labeled farm data). For a live
query neither is known, so both default to None and CandidateValidator
falls back to query-date coverage instead of ground-truth-month coverage.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.lifecycle.boundary import BoundarySelector
from backend.lifecycle.farm_builder import LocationRecord
from backend.signal_processing.smoothing import SignalProcessor


class CandidateGenerator:
    def __init__(self, boundary_selector: BoundarySelector, signal_processor: SignalProcessor):
        self.boundary_selector = boundary_selector
        self.signal_processor = signal_processor

    def build_candidate(
        self,
        record: LocationRecord,
        peak_idx: int,
        crop: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        feature_set: str = "smooth",
    ) -> Optional[dict[str, Any]]:
        boundary_signal = record.ndvi if feature_set == "raw" else record.smoothed
        valleys_arr = record.valleys

        left_idx = self.boundary_selector.find_best_sos(boundary_signal, valleys_arr, peak_idx)
        right_idx = self.boundary_selector.find_best_eos(boundary_signal, valleys_arr, peak_idx)

        if right_idx <= left_idx:
            return None

        raw_curve = record.ndvi[left_idx : right_idx + 1]
        smooth_curve = record.smoothed[left_idx : right_idx + 1]
        cycle_dates = pd.to_datetime(record.dates[left_idx : right_idx + 1])
        cycle_noise = self.signal_processor.estimate_noise(raw_curve, smooth_curve)

        candidate = {
            "location_id": record.location_id,
            "crop": crop,
            "ground_truth": ground_truth,
            "query_date": record.query_date,
            "left_idx": left_idx,
            "peak_idx": peak_idx,
            "right_idx": right_idx,
            "sos_date": pd.Timestamp(record.dates[left_idx]),
            "peak_date": pd.Timestamp(record.dates[peak_idx]),
            "eos_date": pd.Timestamp(record.dates[right_idx]),
            "sos_ndvi_raw": record.ndvi[left_idx],
            "peak_ndvi_raw": record.ndvi[peak_idx],
            "eos_ndvi_raw": record.ndvi[right_idx],
            "sos_ndvi_smooth": record.smoothed[left_idx],
            "peak_ndvi_smooth": record.smoothed[peak_idx],
            "eos_ndvi_smooth": record.smoothed[right_idx],
            "raw_curve": raw_curve.tolist(),
            "smooth_curve": smooth_curve.tolist(),
            "dates": cycle_dates.tolist(),
            "duration": int((cycle_dates[-1] - cycle_dates[0]) / pd.Timedelta(days=1)),
            "farm_noise_estimate": record.noise_estimate,
            "noise_estimate": cycle_noise,
            "status": "Candidate",
            "reason": None,
        }
        return candidate

    def generate_candidates(
        self,
        record: LocationRecord,
        crop: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        feature_set: str = "smooth",
    ) -> list[dict[str, Any]]:
        candidates = []
        for peak in record.peaks:
            candidate = self.build_candidate(record, peak, crop=crop, ground_truth=ground_truth, feature_set=feature_set)
            if candidate is not None:
                candidates.append(candidate)

        record.candidate_cycles = candidates
        return candidates

    @staticmethod
    def compute_quality_metrics(candidate: dict[str, Any], feature_set: str = "smooth") -> dict[str, Any]:
        raw = np.asarray(candidate["raw_curve"], dtype=float)
        smooth = np.asarray(candidate["smooth_curve"], dtype=float)
        selected = raw if feature_set == "raw" else smooth

        baseline = (selected[0] + selected[-1]) / 2

        peak_idx_local = int(candidate["peak_idx"] - candidate["left_idx"])
        peak_idx_local = int(np.clip(peak_idx_local, 0, len(selected) - 1))

        amplitude = selected[peak_idx_local] - baseline

        residual = raw - smooth
        mean_abs_residual = np.nanmean(np.abs(residual))
        rmse_residual = np.sqrt(np.nanmean(residual**2))
        cycle_noise = SignalProcessor.estimate_noise(raw, smooth)

        candidate["baseline_ndvi"] = baseline
        candidate["amplitude"] = amplitude
        candidate["mean_residual"] = mean_abs_residual
        candidate["rmse_residual"] = rmse_residual
        candidate["noise_estimate"] = cycle_noise
        candidate["signal_to_noise"] = (
            amplitude / cycle_noise if cycle_noise not in (0, None) and not pd.isna(cycle_noise) else np.nan
        )
        candidate["num_points"] = len(raw)

        return candidate
