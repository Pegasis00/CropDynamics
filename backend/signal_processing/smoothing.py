"""
SignalProcessor — Stage C of the reference notebook.

Reproduces exactly:
  * adaptive_smoothing()  -> Savitzky-Golay, window chosen by series length
  * dynamic_prominence()  -> clipped 0.15 * (p90 - p10) spread
  * estimate_noise()      -> 1.4826 * MAD of raw-smooth residuals
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from backend.config.loader import PipelineConfig


class SignalProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def adaptive_smoothing(self, ndvi: np.ndarray) -> np.ndarray:
        """Adaptive Savitzky-Golay smoothing based on series length."""
        ndvi = np.asarray(ndvi, dtype=float)
        n = len(ndvi)

        if n < self.config.min_series_length:
            return ndvi

        window = self._window_for_length(n)
        window = min(window, n if n % 2 else n - 1)

        if window < self.config.min_series_length:
            return ndvi

        return savgol_filter(ndvi, window_length=window, polyorder=self.config.polyorder)

    def _window_for_length(self, n: int) -> int:
        for rule in self.config.window_by_length:
            if rule["max_n"] is None or n < rule["max_n"]:
                return rule["window"]
        # Should not happen given the null-terminated config, but fall back
        # to the widest configured window.
        return self.config.window_by_length[-1]["window"]

    def dynamic_prominence(self, signal: np.ndarray) -> float:
        signal = np.asarray(signal, dtype=float)
        signal = signal[np.isfinite(signal)]

        if len(signal) < 5:
            return self.config.prominence_min

        p10, p90 = np.nanpercentile(signal, [10, 90])
        return float(
            np.clip(
                self.config.prominence_factor * (p90 - p10),
                self.config.prominence_min,
                self.config.prominence_max,
            )
        )

    @staticmethod
    def estimate_noise(raw_signal: np.ndarray, smooth_signal: np.ndarray) -> float:
        """Robust NDVI noise estimate using the MAD of raw-smoothed residuals."""
        raw_signal = np.asarray(raw_signal, dtype=float)
        smooth_signal = np.asarray(smooth_signal, dtype=float)

        valid = np.isfinite(raw_signal) & np.isfinite(smooth_signal)
        if not valid.any():
            return np.nan

        residual = raw_signal[valid] - smooth_signal[valid]
        center = np.nanmedian(residual)
        mad = np.nanmedian(np.abs(residual - center))

        if np.isfinite(mad):
            return float(1.4826 * mad)

        return float(np.nanstd(residual))
