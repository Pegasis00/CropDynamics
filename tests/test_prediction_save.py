import datetime as dt

from backend.exporters.crop_ndvi_writer import CropNDVIWriter
from backend.pipelines.inference_pipeline import InferencePipeline


def test_save_prediction_cycle_writes_csv_only_when_called(tmp_path):
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.crop_ndvi_writer = CropNDVIWriter(tmp_path)

    crop_id = pipeline.save_prediction_cycle(
        crop_label="Wheat",
        dates=["2024-11-01", "2024-11-11"],
        raw_ndvi=[0.22, 0.48],
        query_date=dt.date(2025, 1, 15),
        latitude=19.0,
        longitude=74.0,
    )

    rows = (tmp_path / "wheat" / "wheat.csv").read_text().splitlines()

    assert crop_id == "W001"
    assert rows[1] == "Wheat,W001,0.22,2024-11-01,2025-01-15,19.0,74.0"
