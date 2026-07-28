import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backend.features.extractor import FeatureExtractor


@pytest.fixture
def synthetic_cycle():
    dates = pd.date_range("2024-06-01", periods=15, freq="8D")
    days = np.arange(len(dates))
    raw = 0.15 + 0.55 * np.exp(-((days - 7) ** 2) / (2 * 3**2))
    smooth = raw.copy()

    peak_idx = int(np.argmax(smooth))

    return {
        "raw_curve": raw.tolist(),
        "smooth_curve": smooth.tolist(),
        "dates": dates.tolist(),
        "sos_date": dates[0],
        "peak_date": dates[peak_idx],
        "eos_date": dates[-1],
        "peak_idx": peak_idx,
        "left_idx": 0,
        "right_idx": len(dates) - 1,
        "duration": (dates[-1] - dates[0]).days,
        "noise_estimate": 0.01,
        "farm_noise_estimate": 0.01,
        "mean_residual": 0.005,
        "rmse_residual": 0.007,
    }


def test_extract_returns_all_expected_feature_groups(synthetic_cycle):
    extractor = FeatureExtractor()
    features = extractor.extract(synthetic_cycle, "smooth_curve")

    expected_keys = {
        "Duration", "Rise_Days", "Fall_Days", "Peak_Position",
        "Baseline_NDVI", "Peak_NDVI", "Amplitude",
        "Mean_NDVI", "Std_NDVI", "Skewness", "Kurtosis", "Entropy",
        "AUC", "AUC_Above_Baseline", "Peak_Width_Days", "Peak_Prominence",
        "Max_Growth_Rate", "Max_Curvature", "Inflection_Count",
        "Time_to_50pct", "Observation_Count", "Signal_to_Noise",
        "Observations_above_03",
    }
    assert expected_keys.issubset(features.keys())
    # spec calls for ~71 features per curve
    assert len(features) >= 60


def test_extract_handles_raw_and_smooth_consistently(synthetic_cycle):
    extractor = FeatureExtractor()
    raw_features = extractor.extract(synthetic_cycle, "raw_curve")
    smooth_features = extractor.extract(synthetic_cycle, "smooth_curve")

    assert raw_features.keys() == smooth_features.keys()


def test_amplitude_is_positive_for_a_clear_season(synthetic_cycle):
    extractor = FeatureExtractor()
    features = extractor.extract(synthetic_cycle, "smooth_curve")
    assert features["Amplitude"] > 0
