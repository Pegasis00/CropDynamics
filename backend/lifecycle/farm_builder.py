"""
FarmBuilder — Stage B of the reference notebook, adapted for a single live
query instead of a batch of ground-truthed farms.

The notebook keyed everything off `farms[farm_id]`, a dict with dates,
ndvi, and placeholders (smoothed, peaks, valleys, candidate_cycles,
accepted_cycle, rejected_cycles, features) filled in by later stages.
`LocationRecord` is that same structure, typed, for one query point.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from backend.gee.extractor import NDVIObservation


@dataclass
class LocationRecord:
    location_id: str
    latitude: float
    longitude: float
    query_date: dt.date

    dates: np.ndarray  # dtype=datetime64
    ndvi: np.ndarray

    smoothed: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None
    valleys: Optional[np.ndarray] = None
    noise_estimate: float = np.nan

    candidate_cycles: list[dict[str, Any]] = field(default_factory=list)
    accepted_cycle: Optional[dict[str, Any]] = None
    rejected_cycles: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Notebook-compatible dict view — the Stage C/D/G functions below
        were ported to operate on this shape (or on LocationRecord directly
        via attribute access, since dataclasses support both)."""
        return self.__dict__


class FarmBuilder:
    """Builds a LocationRecord from raw GEE observations for one query."""

    def build(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        query_date: dt.date,
        observations: list[NDVIObservation],
    ) -> LocationRecord:
        observations = sorted(observations, key=lambda o: o.date)

        dates = np.array([np.datetime64(o.date) for o in observations])
        ndvi = np.array([o.ndvi for o in observations], dtype=float)

        return LocationRecord(
            location_id=location_id,
            latitude=latitude,
            longitude=longitude,
            query_date=query_date,
            dates=dates,
            ndvi=ndvi,
        )
