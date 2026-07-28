import datetime as dt

import pandas as pd

from backend.exporters.crop_ndvi_writer import CropNDVIWriter


def test_crop_ndvi_writer_appends_sos_to_eos_points(tmp_path):
    writer = CropNDVIWriter(tmp_path)
    cycle = {
        "dates": pd.date_range("2024-07-01", periods=3, freq="10D").tolist(),
        "raw_curve": [0.2, 0.5, 0.3],
    }

    crop_id = writer.append_cycle("Cotton", cycle, dt.date(2024, 7, 15), latitude=18.52, longitude=73.85)
    writer.append_cycle("Cotton", cycle, dt.date(2024, 8, 15), latitude=18.53, longitude=73.86)

    csv_path = tmp_path / "cotton" / "cotton.csv"
    rows = csv_path.read_text().splitlines()

    assert crop_id == "C001"
    assert rows[0] == "CROP_LABEL,CROP_ID,NDVI,DATE,QUERY_DATE,LATITUDE,LONGITUDE"
    assert rows[1] == "Cotton,C001,0.2,2024-07-01,2024-07-15,18.52,73.85"
    assert rows[3] == "Cotton,C001,0.3,2024-07-21,2024-07-15,18.52,73.85"
    assert rows[4] == "Cotton,C002,0.2,2024-07-01,2024-08-15,18.53,73.86"


def test_crop_ndvi_writer_ignores_unknown_labels(tmp_path):
    writer = CropNDVIWriter(tmp_path)
    cycle = {
        "dates": pd.date_range("2024-07-01", periods=1).tolist(),
        "raw_curve": [0.2],
    }

    crop_id = writer.append_cycle("Other Crop", cycle, dt.date(2024, 7, 15))

    assert crop_id == ""
    assert not any(tmp_path.iterdir())


def test_crop_ndvi_writer_saves_wheat_predictions(tmp_path):
    writer = CropNDVIWriter(tmp_path)
    cycle = {
        "dates": pd.date_range("2024-11-01", periods=2, freq="10D").tolist(),
        "raw_curve": [0.22, 0.48],
    }

    crop_id = writer.append_cycle("Wheat", cycle, dt.date(2025, 1, 15), latitude=19.0, longitude=74.0)

    csv_path = tmp_path / "wheat" / "wheat.csv"
    rows = csv_path.read_text().splitlines()

    assert crop_id == "W001"
    assert rows[1] == "Wheat,W001,0.22,2024-11-01,2025-01-15,19.0,74.0"
    assert rows[2] == "Wheat,W001,0.48,2024-11-11,2025-01-15,19.0,74.0"


def test_crop_ndvi_writer_migrates_existing_csv_schema(tmp_path):
    csv_dir = tmp_path / "onion"
    csv_dir.mkdir()
    csv_path = csv_dir / "onion.csv"
    csv_path.write_text(
        "CROP_LABEL,CROP_ID,NDVI,DATE,QUERY_DATE\n"
        "Onion,O001,0.2,2024-07-01,2024-07-15\n"
    )
    writer = CropNDVIWriter(tmp_path)
    cycle = {
        "dates": pd.date_range("2024-08-01", periods=1).tolist(),
        "raw_curve": [0.4],
    }

    crop_id = writer.append_cycle("Onion", cycle, dt.date(2024, 8, 15), latitude=18.1, longitude=73.1)
    rows = csv_path.read_text().splitlines()

    assert crop_id == "O002"
    assert rows[0] == "CROP_LABEL,CROP_ID,NDVI,DATE,QUERY_DATE,LATITUDE,LONGITUDE"
    assert rows[1] == "Onion,O001,0.2,2024-07-01,2024-07-15,,"
    assert rows[2] == "Onion,O002,0.4,2024-08-01,2024-08-15,18.1,73.1"
