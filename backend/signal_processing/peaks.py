"""
PeakDetector / ValleyDetector — Stage C of the reference notebook.

Both use find_peaks with the dynamic prominence threshold, then further
filter to peaks/valleys whose prominence clears
max(prominence, SIG_MULTIPLIER * noise). ValleyDetector runs the identical
procedure on the inverted signal.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from backend.config.loader import PipelineConfig
from backend.signal_processing.smoothing import SignalProcessor


class PeakDetector:
    def __init__(self, config: PipelineConfig, signal_processor: SignalProcessor):
        self.config = config
        self.signal_processor = signal_processor

    def detect(self, raw_signal: np.ndarray, smooth_signal: np.ndarray) -> tuple[np.ndarray, dict, float]:
        prominence = self.signal_processor.dynamic_prominence(smooth_signal)
        noise = self.signal_processor.estimate_noise(raw_signal, smooth_signal)

        threshold = max(prominence, self.config.sig_multiplier * noise)

        peaks_raw, props = find_peaks(smooth_signal, prominence=prominence)
        peaks = peaks_raw[props["prominences"] >= threshold]

        return peaks, props, noise


class ValleyDetector:
    def __init__(self, config: PipelineConfig, signal_processor: SignalProcessor):
        self.config = config
        self.signal_processor = signal_processor

    def detect(self, raw_signal: np.ndarray, smooth_signal: np.ndarray) -> tuple[np.ndarray, dict, float]:
        prominence = self.signal_processor.dynamic_prominence(smooth_signal)
        noise = self.signal_processor.estimate_noise(raw_signal, smooth_signal)

        threshold = max(prominence, self.config.sig_multiplier * noise)

        valleys_raw, props = find_peaks(-smooth_signal, prominence=prominence)
        valleys = valleys_raw[props["prominences"] >= threshold]

        return valleys, props, noise
