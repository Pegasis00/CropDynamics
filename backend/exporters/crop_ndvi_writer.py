"""Append predicted crop lifecycle NDVI points to per-crop CSV files."""
from __future__ import annotations

import csv
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd


CSV_COLUMNS = ["CROP_LABEL", "CROP_ID", "NDVI", "DATE", "QUERY_DATE", "LATITUDE", "LONGITUDE"]
CROP_ID_PREFIX = {
    "Cotton": "C",
    "Onion": "O",
    "Paddy": "P",
    "Sugarcane": "S",
    "Wheat": "W",
}


class CropNDVIWriter:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._lock = threading.Lock()

    def append_cycle(
        self,
        crop_label: str,
        cycle: dict[str, Any],
        query_date,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> str:
        if crop_label not in CROP_ID_PREFIX:
            return ""

        dates = pd.to_datetime(cycle["dates"])
        ndvi_values = [float(v) for v in cycle["raw_curve"]]
        query_date_text = query_date.isoformat() if hasattr(query_date, "isoformat") else str(query_date)

        with self._lock:
            csv_path = self._csv_path(crop_label)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema(csv_path)
            crop_id = self._next_crop_id(crop_label, csv_path)
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0

            with csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                if write_header:
                    writer.writeheader()
                for date, ndvi in zip(dates, ndvi_values):
                    writer.writerow(
                        {
                            "CROP_LABEL": crop_label,
                            "CROP_ID": crop_id,
                            "NDVI": ndvi,
                            "DATE": date.date().isoformat(),
                            "QUERY_DATE": query_date_text,
                            "LATITUDE": "" if latitude is None else latitude,
                            "LONGITUDE": "" if longitude is None else longitude,
                        }
                    )

        return crop_id

    def _csv_path(self, crop_label: str) -> Path:
        folder = crop_label.lower()
        return self.base_dir / folder / f"{folder}.csv"

    def _next_crop_id(self, crop_label: str, csv_path: Path) -> str:
        prefix = CROP_ID_PREFIX[crop_label]
        max_number = 0
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with csv_path.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    crop_id = row.get("CROP_ID", "")
                    match = re.fullmatch(rf"{prefix}(\d+)", crop_id)
                    if match:
                        max_number = max(max_number, int(match.group(1)))
        return f"{prefix}{max_number + 1:03d}"

    def _ensure_schema(self, csv_path: Path) -> None:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            if existing_fields == CSV_COLUMNS:
                return
            rows = list(reader)

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})
