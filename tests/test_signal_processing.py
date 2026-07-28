import numpy as np
import pytest

from backend.config.loader import PipelineConfig
from backend.signal_processing.peaks import PeakDetector, ValleyDetector
from backend.signal_processing.smoothing import SignalProcessor


@pytest.fixture
def pipeline_config():
    return PipelineConfig.load()


@pytest.fixture
def synthetic_season():
    """A single bell-shaped growing season with a bit of noise, similar in
    shape to a real NDVI lifecycle."""
    days = np.arange(0, 120, 8)
    clean = 0.15 + 0.55 * np.exp(-((days - 60) ** 2) / (2 * 25**2))
    rng = np.random.default_rng(42)
    noisy = clean + rng.normal(0, 0.01, size=len(days))
    return noisy


def test_adaptive_smoothing_preserves_length(pipeline_config, synthetic_season):
    processor = SignalProcessor(pipeline_config)
    smoothed = processor.adaptive_smoothing(synthetic_season)
    assert len(smoothed) == len(synthetic_season)


def test_adaptive_smoothing_short_series_passthrough(pipeline_config):
    processor = SignalProcessor(pipeline_config)
    short = np.array([0.2, 0.3, 0.25])
    result = processor.adaptive_smoothing(short)
    np.testing.assert_array_equal(result, short)


def test_estimate_noise_is_nonnegative(synthetic_season):
    smoothed = np.convolve(synthetic_season, np.ones(3) / 3, mode="same")
    noise = SignalProcessor.estimate_noise(synthetic_season, smoothed)
    assert noise >= 0


def test_peak_detector_finds_the_single_season_peak(pipeline_config, synthetic_season):
    processor = SignalProcessor(pipeline_config)
    smoothed = processor.adaptive_smoothing(synthetic_season)
    detector = PeakDetector(pipeline_config, processor)

    peaks, _, noise = detector.detect(synthetic_season, smoothed)
    assert len(peaks) >= 1
    # the peak should land near the middle of the series (index ~7-8 of 15)
    assert 4 <= peaks[0] <= 11


def test_valley_detector_runs_without_error(pipeline_config, synthetic_season):
    processor = SignalProcessor(pipeline_config)
    smoothed = processor.adaptive_smoothing(synthetic_season)
    detector = ValleyDetector(pipeline_config, processor)

    valleys, _, _ = detector.detect(synthetic_season, smoothed)
    assert isinstance(valleys, np.ndarray)
