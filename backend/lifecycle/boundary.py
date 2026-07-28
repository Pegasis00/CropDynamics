"""
Boundary selection — Stage D of the reference notebook.

find_left_boundary / find_right_boundary: prefer a detected valley,
fall back to the argmin of the signal on the relevant side.

growth_score / decline_score: how strongly the signal rises after a
valley / declines before a valley, used to pick the *best* SOS/EOS when
several valleys are candidates.

find_best_sos / find_best_eos: pick the valley maximizing that score.
"""
from __future__ import annotations

import numpy as np

from backend.config.loader import PipelineConfig


class BoundarySelector:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def find_left_boundary(self, smoothed: np.ndarray, valleys: np.ndarray, peak_idx: int) -> int:
        left = valleys[valleys < peak_idx]
        if len(left):
            return int(left[-1])
        return int(np.argmin(smoothed[: peak_idx + 1]))

    def find_right_boundary(self, smoothed: np.ndarray, valleys: np.ndarray, peak_idx: int) -> int:
        right = valleys[valleys > peak_idx]
        if len(right):
            return int(right[0])
        return int(peak_idx + np.argmin(smoothed[peak_idx:]))

    def growth_score(self, signal: np.ndarray, valley_idx: int, peak_idx: int) -> float:
        """Higher score = better SOS candidate."""
        window = self.config.growth_score_window

        if valley_idx >= peak_idx:
            return -np.inf

        end = min(valley_idx + window, len(signal) - 1)
        segment = signal[valley_idx : end + 1]

        if len(segment) < 2:
            return -np.inf

        rise = segment[-1] - segment[0]
        slopes = np.diff(segment)
        positive_fraction = np.mean(slopes > 0)

        return rise + positive_fraction

    def decline_score(self, signal: np.ndarray, peak_idx: int, valley_idx: int) -> float:
        """Higher score = better EOS candidate."""
        window = self.config.decline_score_window

        if valley_idx <= peak_idx:
            return -np.inf

        start = max(peak_idx, valley_idx - window)
        segment = signal[start : valley_idx + 1]

        if len(segment) < 2:
            return -np.inf

        drop = segment[0] - segment[-1]
        slopes = np.diff(segment)
        negative_fraction = np.mean(slopes < 0)

        return drop + negative_fraction

    def find_best_sos(self, signal: np.ndarray, valleys: np.ndarray, peak_idx: int) -> int:
        left = valleys[valleys < peak_idx]

        if len(left) == 0:
            return int(np.argmin(signal[: peak_idx + 1]))

        scores = [self.growth_score(signal, v, peak_idx) for v in left]
        return int(left[int(np.argmax(scores))])

    def find_best_eos(self, signal: np.ndarray, valleys: np.ndarray, peak_idx: int) -> int:
        right = valleys[valleys > peak_idx]

        if len(right) == 0:
            return int(peak_idx + np.argmin(signal[peak_idx:]))

        scores = [self.decline_score(signal, peak_idx, v) for v in right]
        return int(right[int(np.argmax(scores))])
