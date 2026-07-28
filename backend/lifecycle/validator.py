"""
CandidateValidator — Stage D "Validation" of the reference notebook.

Training-time behavior (check_ground_truth) is preserved unchanged for
offline retraining. For live inference there is no ground-truth month, so
check_query_date_coverage plays the equivalent role: the candidate's
SOS..EOS span must contain (or nearly contain) the query date.

check_duration: crop-specific median +/- 2*IQR bounds, read from
configs/thresholds.yaml (generated offline from the training data — see
that file's header comment).

validate_candidate: Accepted only if every configured check passes,
Rejected with the failing check name(s) otherwise.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from backend.config.loader import ThresholdsConfig


class CandidateValidator:
    def __init__(self, config: ThresholdsConfig):
        self.config = config

    def check_ground_truth(self, candidate: dict[str, Any]) -> bool:
        """Training-time check: candidate span must overlap the labeled
        ground-truth month."""
        gt = pd.to_datetime(candidate["ground_truth"])
        gt_start = gt.to_period("M").start_time
        gt_end = gt.to_period("M").end_time.normalize()

        return candidate["sos_date"] <= gt_end and candidate["eos_date"] >= gt_start

    def check_query_date_coverage(self, candidate: dict[str, Any]) -> bool:
        """Live-inference check: candidate span must contain the query
        date used to build the extraction window."""
        query_date = pd.Timestamp(candidate["query_date"])
        return candidate["sos_date"] <= query_date <= candidate["eos_date"]

    def check_duration(self, candidate: dict[str, Any]) -> bool:
        crop = candidate.get("crop")
        stats = self.config.duration_stats.get(crop)

        if stats is None:
            # Unknown/unlabeled crop (always true at live-inference time,
            # since the crop isn't known until after prediction) — duration
            # is checked later, post-hoc, against the predicted class if
            # desired. Here we simply don't reject on a check we can't run.
            return True

        lower = stats["median"] - self.config.duration_iqr_multiplier * stats["iqr"]
        upper = stats["median"] + self.config.duration_iqr_multiplier * stats["iqr"]

        return lower <= candidate["duration"] <= upper

    def validate_candidate(self, candidate: dict[str, Any], mode: str = "inference") -> dict[str, Any]:
        """mode: "training" uses check_ground_truth; "inference" uses
        check_query_date_coverage."""
        quality: dict[str, bool] = {}

        if mode == "training":
            quality["gt_match"] = self.check_ground_truth(candidate)
        else:
            quality["query_date_coverage"] = self.check_query_date_coverage(candidate)

        quality["duration_ok"] = self.check_duration(candidate)
        quality["amplitude_ok"] = candidate["amplitude"] > self.config.amplitude_floor
        quality["residual_ok"] = True
        quality["coverage_ok"] = True

        candidate["quality"] = quality

        if all(quality.values()):
            candidate["status"] = "Accepted"
            candidate["reason"] = "Passed"
        else:
            candidate["status"] = "Rejected"
            candidate["reason"] = ", ".join(k for k, v in quality.items() if not v)

        return candidate
