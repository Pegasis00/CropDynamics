"""
PlotGenerator — Stage E of the reference notebook, adapted for a web
frontend: instead of rendering matplotlib figures server-side, it emits
the same series/markers the notebook's plot_smoothing / plot_signal /
plot_selected_cycle functions drew, as plain JSON the frontend charts
directly. The underlying data and marker logic is unchanged.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.lifecycle.farm_builder import LocationRecord


class PlotGenerator:
    @staticmethod
    def _format_query_date(query_date) -> Optional[str]:
        if query_date is None:
            return None
        return query_date.isoformat() if hasattr(query_date, "isoformat") else str(query_date)

    def raw_vs_smoothed(self, record: LocationRecord, query_date=None) -> dict[str, Any]:
        """Mirrors plot_smoothing(): raw NDVI + smoothed NDVI over time."""
        dates = pd.to_datetime(record.dates)
        return {
            "dates": [d.date().isoformat() for d in dates],
            "raw_ndvi": [float(v) for v in record.ndvi],
            "smoothed_ndvi": [float(v) for v in record.smoothed] if record.smoothed is not None else [],
            "query_date": self._format_query_date(query_date),
        }

    def signal_with_markers(self, record: LocationRecord, query_date=None, feature_set: str = "smooth") -> dict[str, Any]:
        """Mirrors plot_signal(): raw + smoothed series plus detected
        peak/valley markers."""
        base = self.raw_vs_smoothed(record, query_date)
        dates = pd.to_datetime(record.dates)
        marker_values = record.ndvi if feature_set == "raw" else record.smoothed

        peaks = record.peaks if record.peaks is not None else np.array([], dtype=int)
        valleys = record.valleys if record.valleys is not None else np.array([], dtype=int)

        base["peaks"] = [
            {"date": dates[i].date().isoformat(), "ndvi": float(marker_values[i])} for i in peaks
        ]
        base["valleys"] = [
            {"date": dates[i].date().isoformat(), "ndvi": float(marker_values[i])} for i in valleys
        ]
        base["feature_set"] = feature_set
        return base

    def selected_lifecycle(
        self,
        record: LocationRecord,
        cycle: Optional[dict[str, Any]],
        query_date,
        feature_set: str = "smooth",
    ) -> dict[str, Any]:
        """Mirrors plot_selected_cycle(): raw + smoothed series, the
        selected SOS..EOS span highlighted, and SOS/Peak/EOS/query-date
        markers. Uses the lifecycle actually produced by the backend, not
        a placeholder — if no cycle was accepted, only the raw/smoothed
        series and the query-date marker are returned."""
        base = self.raw_vs_smoothed(record, query_date)

        if cycle is None:
            base["lifecycle"] = None
            return base

        dates = pd.to_datetime(cycle["dates"])
        suffix = "raw" if feature_set == "raw" else "smooth"
        selected_curve_key = "raw_curve" if feature_set == "raw" else "smooth_curve"

        base["lifecycle"] = {
            "feature_set": feature_set,
            "sos": {"date": pd.Timestamp(cycle["sos_date"]).date().isoformat(), "ndvi": float(cycle[f"sos_ndvi_{suffix}"])},
            "peak": {
                "date": pd.Timestamp(cycle["peak_date"]).date().isoformat(),
                "ndvi": float(cycle[f"peak_ndvi_{suffix}"]),
            },
            "eos": {"date": pd.Timestamp(cycle["eos_date"]).date().isoformat(), "ndvi": float(cycle[f"eos_ndvi_{suffix}"])},
            "span_dates": [d.date().isoformat() for d in dates],
            "span_selected_ndvi": [float(v) for v in cycle[selected_curve_key]],
            "span_smooth_ndvi": [float(v) for v in cycle["smooth_curve"]],
            "span_raw_ndvi": [float(v) for v in cycle["raw_curve"]],
        }
        return base
