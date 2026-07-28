"""
FeatureExtractor — Stage G of the reference notebook (`build_cycle_features`),
ported feature-for-feature. Run once on the raw curve and once on the
smoothed curve to produce the two parallel feature vectors the model layer
expects.

Categories preserved exactly: temporal, amplitude/baseline, distribution
stats, area-under-curve, peak shape, derivative-based, phenological
timing, observation density/quality, signal quality, threshold crossings.
"""
from __future__ import annotations

from itertools import groupby
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import peak_prominences, peak_widths
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis, skew


def _safe_div(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    for value, group in groupby(mask):
        if value:
            longest = max(longest, len(list(group)))
    return longest


def _first_crossing_day(days, values, threshold, descending=False):
    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(values) < 2 or np.all(~np.isfinite(values)):
        return np.nan

    for i in range(1, len(values)):
        v0, v1 = values[i - 1], values[i]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue

        crossed = (v0 >= threshold >= v1) if descending else (v0 <= threshold <= v1)
        if not crossed:
            continue

        if v1 == v0:
            return days[i]

        frac = (threshold - v0) / (v1 - v0)
        return days[i - 1] + frac * (days[i] - days[i - 1])

    return np.nan


def _safe_gradient(values, days):
    values = np.asarray(values, dtype=float)
    days = np.asarray(days, dtype=float)

    if len(values) < 2:
        return np.full(len(values), np.nan)

    if np.all(np.isfinite(days)) and np.all(np.diff(days) > 0):
        return np.gradient(values, days)

    return np.gradient(values)


def _safe_peak_width(ndvi, peak_idx):
    if len(ndvi) < 3 or peak_idx <= 0 or peak_idx >= len(ndvi) - 1:
        return np.nan
    try:
        return peak_widths(ndvi, [peak_idx], rel_height=0.5)[0][0]
    except Exception:
        return np.nan


def _safe_peak_prominence(ndvi, peak_idx):
    if len(ndvi) < 3 or peak_idx <= 0 or peak_idx >= len(ndvi) - 1:
        return np.nan
    try:
        return peak_prominences(ndvi, [peak_idx])[0][0]
    except Exception:
        return np.nan


class FeatureExtractor:
    """Stateless; `stable_peak_fraction` / `plateau_fraction` come from
    configs/pipeline.yaml so the 0.90 / 0.95 thresholds stay file-driven."""

    def __init__(self, stable_peak_fraction: float = 0.90, plateau_fraction: float = 0.95):
        self.stable_peak_fraction = stable_peak_fraction
        self.plateau_fraction = plateau_fraction

    def extract(self, cycle: dict[str, Any], signal_key: str) -> dict[str, Any]:
        """signal_key: "raw_curve" or "smooth_curve"."""
        ndvi = np.asarray(cycle[signal_key], dtype=float)
        dates = pd.to_datetime(cycle["dates"])
        n = len(ndvi)

        if n == 0:
            return {}

        sos_date = pd.Timestamp(cycle["sos_date"])
        eos_date = pd.Timestamp(cycle["eos_date"])
        duration = max(int(cycle["duration"]), 1)

        days_elapsed = ((dates - dates[0]) / pd.Timedelta(days=1)).astype(float).to_numpy()
        time_gaps = np.diff(days_elapsed)
        mean_time_gap = np.nanmean(time_gaps) if len(time_gaps) else np.nan

        baseline = (ndvi[0] + ndvi[-1]) / 2

        if signal_key == "smooth_curve":
            peak_idx = int(cycle["peak_idx"] - cycle["left_idx"])
            peak_idx = int(np.clip(peak_idx, 0, n - 1))
            peak_date = pd.Timestamp(cycle["peak_date"])
            peak_ndvi = float(ndvi[peak_idx])
        else:
            peak_idx = int(np.nanargmax(ndvi))
            peak_date = pd.Timestamp(dates[peak_idx])
            peak_ndvi = float(ndvi[peak_idx])

        amplitude = peak_ndvi - baseline
        rise_days = max((peak_date - sos_date).days, 0)
        fall_days = max((eos_date - peak_date).days, 0)

        growth_rate = _safe_div(amplitude, rise_days)
        drop_rate = _safe_div(amplitude, fall_days)
        peak_position = _safe_div(rise_days, duration)
        growth_ratio = _safe_div(rise_days, duration)
        decline_ratio = _safe_div(fall_days, duration)
        rise_fall_ratio = _safe_div(rise_days, fall_days)
        symmetry_index = 1 - abs(
            (growth_ratio if pd.notna(growth_ratio) else np.nan)
            - (decline_ratio if pd.notna(decline_ratio) else np.nan)
        )

        mean_ndvi = np.nanmean(ndvi)
        std_ndvi = np.nanstd(ndvi)
        median_ndvi = np.nanmedian(ndvi)
        min_ndvi = np.nanmin(ndvi)
        max_ndvi = np.nanmax(ndvi)
        range_ndvi = max_ndvi - min_ndvi
        variance_ndvi = np.nanvar(ndvi)
        q25_ndvi, q75_ndvi = np.nanpercentile(ndvi, [25, 75])
        iqr_ndvi = q75_ndvi - q25_ndvi
        mad_ndvi = np.nanmedian(np.abs(ndvi - median_ndvi))
        coeff_variation = _safe_div(std_ndvi, mean_ndvi)

        total_auc = trapezoid(ndvi, x=days_elapsed) if n >= 2 else np.nan
        auc_above_baseline = trapezoid(ndvi - baseline, x=days_elapsed) if n >= 2 else np.nan
        growth_auc = (
            trapezoid(ndvi[: peak_idx + 1] - baseline, x=days_elapsed[: peak_idx + 1]) if peak_idx > 0 else np.nan
        )
        decline_auc = (
            trapezoid(ndvi[peak_idx:] - baseline, x=days_elapsed[peak_idx:]) if peak_idx < n - 1 else np.nan
        )
        auc_ratio = _safe_div(growth_auc, decline_auc)

        peak_width_samples = _safe_peak_width(ndvi, peak_idx)
        peak_width_days = (
            peak_width_samples * mean_time_gap
            if pd.notna(peak_width_samples) and pd.notna(mean_time_gap)
            else np.nan
        )
        peak_prominence = _safe_peak_prominence(ndvi, peak_idx)
        peak_sharpness = _safe_div(amplitude, peak_width_days)

        peak_threshold = baseline + (self.stable_peak_fraction * amplitude)
        plateau_threshold = baseline + (self.plateau_fraction * amplitude)
        stable_peak_mask = ndvi >= peak_threshold
        plateau_mask = ndvi >= plateau_threshold
        stable_peak_observation_count = int(np.sum(stable_peak_mask))
        plateau_observation_count = _longest_true_run(plateau_mask)
        stable_peak_days = (
            stable_peak_observation_count * mean_time_gap if pd.notna(mean_time_gap) else np.nan
        )
        peak_flatness = np.nanvar(ndvi[plateau_mask]) if np.sum(plateau_mask) > 1 else np.nan

        first_deriv = _safe_gradient(ndvi, days_elapsed)
        growth_mask = first_deriv > 0
        decline_mask = first_deriv < 0

        max_growth_rate = np.nanmax(first_deriv[growth_mask]) if growth_mask.any() else np.nan
        mean_growth_rate = np.nanmean(first_deriv[growth_mask]) if growth_mask.any() else np.nan
        max_decline_rate = np.nanmax(np.abs(first_deriv[decline_mask])) if decline_mask.any() else np.nan
        mean_decline_rate = np.nanmean(np.abs(first_deriv[decline_mask])) if decline_mask.any() else np.nan

        second_deriv = _safe_gradient(first_deriv, days_elapsed) if n >= 3 else np.full(n, np.nan)
        max_curvature = np.nanmax(second_deriv) if np.isfinite(second_deriv).any() else np.nan
        min_curvature = np.nanmin(second_deriv) if np.isfinite(second_deriv).any() else np.nan
        mean_curvature = np.nanmean(second_deriv) if np.isfinite(second_deriv).any() else np.nan
        peak_curvature = second_deriv[peak_idx] if n >= 3 else np.nan

        if n >= 3 and np.isfinite(second_deriv).any():
            signs = np.sign(second_deriv)
            signs[~np.isfinite(signs)] = 0
            change_idx = np.where(np.diff(signs) != 0)[0]
        else:
            change_idx = np.array([], dtype=int)

        inflection_count = len(change_idx)
        first_inflection_day = days_elapsed[change_idx[0] + 1] if inflection_count > 0 else np.nan
        last_inflection_day = days_elapsed[change_idx[-1] + 1] if inflection_count > 0 else np.nan

        amp_denom = peak_ndvi - baseline
        norm_curve = (ndvi - baseline) / amp_denom if amp_denom != 0 else np.full(n, np.nan)
        rising_days, rising_norm = days_elapsed[: peak_idx + 1], norm_curve[: peak_idx + 1]
        falling_days, falling_norm = days_elapsed[peak_idx:], norm_curve[peak_idx:]

        time_to_20pct = _first_crossing_day(rising_days, rising_norm, 0.2)
        time_to_50pct = _first_crossing_day(rising_days, rising_norm, 0.5)
        time_to_80pct = _first_crossing_day(rising_days, rising_norm, 0.8)
        time_80_desc = _first_crossing_day(falling_days, falling_norm, 0.8, descending=True)
        time_50_desc = _first_crossing_day(falling_days, falling_norm, 0.5, descending=True)
        time_20_desc = _first_crossing_day(falling_days, falling_norm, 0.2, descending=True)
        time_from_80_to_50 = (
            (time_50_desc - time_80_desc) if not (pd.isna(time_50_desc) or pd.isna(time_80_desc)) else np.nan
        )
        time_from_50_to_20 = (
            (time_20_desc - time_50_desc) if not (pd.isna(time_20_desc) or pd.isna(time_50_desc)) else np.nan
        )

        observation_count = n
        observation_density = _safe_div(observation_count, duration)
        mean_days_per_observation = _safe_div(duration, observation_count)
        largest_time_gap = np.nanmax(time_gaps) if len(time_gaps) else np.nan
        noise_estimate = cycle.get("noise_estimate", np.nan)
        farm_noise_estimate = cycle.get("farm_noise_estimate", np.nan)
        mean_residual = cycle.get("mean_residual", np.nan)
        rmse_residual = cycle.get("rmse_residual", np.nan)
        signal_to_noise = _safe_div(amplitude, noise_estimate)

        observations_above_03 = int(np.sum(ndvi > 0.3))
        observations_above_04 = int(np.sum(ndvi > 0.4))
        observations_above_05 = int(np.sum(ndvi > 0.5))
        observations_above_06 = int(np.sum(ndvi > 0.6))

        ndvi_skewness = skew(ndvi, nan_policy="omit") if n > 2 else np.nan
        ndvi_kurtosis = kurtosis(ndvi, nan_policy="omit") if n > 2 else np.nan
        hist_counts, _ = np.histogram(ndvi[np.isfinite(ndvi)], bins="auto")
        hist_counts = hist_counts[hist_counts > 0]
        ndvi_entropy = scipy_entropy(hist_counts) if len(hist_counts) > 0 else np.nan

        return {
            "Duration": duration,
            "Rise_Days": rise_days,
            "Fall_Days": fall_days,
            "Peak_Position": peak_position,
            "Growth_Ratio": growth_ratio,
            "Decline_Ratio": decline_ratio,
            "Rise_Fall_Ratio": rise_fall_ratio,
            "Symmetry_Index": symmetry_index,
            "Baseline_NDVI": baseline,
            "Peak_NDVI": peak_ndvi,
            "Amplitude": amplitude,
            "Growth_Rate": growth_rate,
            "Drop_Rate": drop_rate,
            "Mean_NDVI": mean_ndvi,
            "Std_NDVI": std_ndvi,
            "Median_NDVI": median_ndvi,
            "Min_NDVI": min_ndvi,
            "Max_NDVI": max_ndvi,
            "Range_NDVI": range_ndvi,
            "Variance_NDVI": variance_ndvi,
            "Q25_NDVI": q25_ndvi,
            "Q75_NDVI": q75_ndvi,
            "IQR_NDVI": iqr_ndvi,
            "MAD_NDVI": mad_ndvi,
            "Coeff_Variation": coeff_variation,
            "Skewness": ndvi_skewness,
            "Kurtosis": ndvi_kurtosis,
            "Entropy": ndvi_entropy,
            "AUC": total_auc,
            "AUC_Above_Baseline": auc_above_baseline,
            "Growth_AUC": growth_auc,
            "Decline_AUC": decline_auc,
            "AUC_Ratio": auc_ratio,
            "Peak_Width_Samples": peak_width_samples,
            "Peak_Width_Days": peak_width_days,
            "Peak_Prominence": peak_prominence,
            "Peak_Sharpness": peak_sharpness,
            "Stable_Peak_Observation_Count": stable_peak_observation_count,
            "Stable_Peak_Days": stable_peak_days,
            "Plateau_Observation_Count": plateau_observation_count,
            "Peak_Flatness": peak_flatness,
            "Max_Growth_Rate": max_growth_rate,
            "Mean_Growth_Rate": mean_growth_rate,
            "Max_Decline_Rate": max_decline_rate,
            "Mean_Decline_Rate": mean_decline_rate,
            "Max_Curvature": max_curvature,
            "Min_Curvature": min_curvature,
            "Mean_Curvature": mean_curvature,
            "Peak_Curvature": peak_curvature,
            "Inflection_Count": inflection_count,
            "First_Inflection_Day": first_inflection_day,
            "Last_Inflection_Day": last_inflection_day,
            "Time_to_20pct": time_to_20pct,
            "Time_to_50pct": time_to_50pct,
            "Time_to_80pct": time_to_80pct,
            "Time_from_80_to_50": time_from_80_to_50,
            "Time_from_50_to_20": time_from_50_to_20,
            "Observation_Count": observation_count,
            "Observation_Density": observation_density,
            "Mean_Days_Per_Observation": mean_days_per_observation,
            "Largest_Time_Gap": largest_time_gap,
            "Mean_Time_Gap": mean_time_gap,
            "Noise_Estimate": noise_estimate,
            "Farm_Noise_Estimate": farm_noise_estimate,
            "Mean_Residual": mean_residual,
            "RMSE_Residual": rmse_residual,
            "Signal_to_Noise": signal_to_noise,
            "Observations_above_03": observations_above_03,
            "Observations_above_04": observations_above_04,
            "Observations_above_05": observations_above_05,
            "Observations_above_06": observations_above_06,
        }
