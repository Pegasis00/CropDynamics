from types import SimpleNamespace
import datetime as dt

from backend.lifecycle.candidates import CandidateGenerator
from backend.lifecycle.farm_builder import LocationRecord
from backend.pipelines.inference_pipeline import InferencePipeline


def _pipeline():
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.settings = SimpleNamespace(
        thresholds=SimpleNamespace(
            min_cycle_peak_ndvi=0.40,
            min_cycle_amplitude=0.12,
        )
    )
    return pipeline


def test_low_flat_smooth_cycle_is_not_a_crop_signal():
    cycle = {
        "peak_ndvi_smooth": 0.33,
        "amplitude": 0.08,
    }

    assert _pipeline()._is_valid_crop_signal(cycle, "smooth") is False


def test_strong_smooth_cycle_is_a_crop_signal():
    cycle = {
        "peak_ndvi_smooth": 0.58,
        "amplitude": 0.24,
    }

    assert _pipeline()._is_valid_crop_signal(cycle, "smooth") is True


def test_raw_cycle_uses_raw_peak_ndvi():
    cycle = {
        "peak_ndvi_raw": 0.32,
        "peak_ndvi_smooth": 0.62,
        "amplitude": 0.22,
    }

    assert _pipeline()._is_valid_crop_signal(cycle, "raw") is False


def test_manual_low_ndvi_cycle_returns_no_plant_found():
    pipeline = _pipeline()
    pipeline.signal_processor = SimpleNamespace(estimate_noise=lambda raw, smooth: 0.01)
    pipeline.candidate_generator = CandidateGenerator.__new__(CandidateGenerator)
    pipeline.plot_generator = SimpleNamespace(
        raw_vs_smoothed=lambda record, query_date: {"dates": [], "raw_ndvi": [], "smoothed_ndvi": []},
        signal_with_markers=lambda record, query_date, feature_set="smooth": {
            "dates": [],
            "raw_ndvi": [],
            "smoothed_ndvi": [],
            "feature_set": feature_set,
        },
        selected_lifecycle=lambda record, cycle, query_date, feature_set="smooth": {
            "dates": [],
            "raw_ndvi": [],
            "smoothed_ndvi": [],
            "query_date": str(query_date),
            "lifecycle": None,
        },
    )

    result = pipeline.rerun_manual_lifecycle(
        dates=["2024-01-01", "2024-02-01", "2024-03-01"],
        raw_ndvi=[0.18, 0.25, 0.20],
        smoothed_ndvi=[0.18, 0.24, 0.20],
        sos_date=dt.date(2024, 1, 1),
        eos_date=dt.date(2024, 3, 1),
        query_date=dt.date(2024, 2, 1),
        feature_set="smooth",
    )

    assert result.predicted_crop == "No plant found"
    assert result.is_other_crop is True
    assert result.lifecycle is None
