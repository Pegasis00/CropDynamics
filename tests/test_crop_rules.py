from types import SimpleNamespace

from backend.models.prediction_service import PredictionResult
from backend.pipelines.inference_pipeline import InferencePipeline


def _pipeline_with_rules():
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.settings = SimpleNamespace(
        crop_rules=SimpleNamespace(
            enabled=True,
            fallback_label="Other Crop",
            crops={
                "Sugarcane": {
                    "duration_days": {
                        "min_exclusive": 280,
                    }
                },
                "Wheat": {
                    "start_month": {
                        "allowed": [10, 11, 12],
                    },
                    "end_month": {
                        "allowed": [2, 3, 4, 5],
                    },
                },
                "Paddy": {
                    "peak_month": {
                        "min": 8,
                        "max": 10,
                    }
                }
            },
        )
    )
    return pipeline


def _prediction(crop="Sugarcane"):
    return PredictionResult(
        predicted_crop=crop,
        confidence=0.91,
        class_probabilities={"Sugarcane": 0.91},
        is_other_crop=False,
    )


def test_sugarcane_duration_must_be_greater_than_280():
    result = _pipeline_with_rules()._apply_post_prediction_rules(
        _prediction(),
        {"duration": 280},
    )

    assert result.predicted_crop == "Other Crop"
    assert result.is_other_crop is True


def test_sugarcane_duration_above_280_is_kept():
    result = _pipeline_with_rules()._apply_post_prediction_rules(
        _prediction(),
        {"duration": 281},
    )

    assert result.predicted_crop == "Sugarcane"
    assert result.is_other_crop is False


def test_rules_do_not_affect_unconfigured_crops():
    result = _pipeline_with_rules()._apply_post_prediction_rules(
        _prediction("Cotton"),
        {"duration": 120},
    )

    assert result.predicted_crop == "Cotton"
    assert result.is_other_crop is False


def test_start_end_month_rules_reject_bad_season():
    result = _pipeline_with_rules()._apply_post_prediction_rules(
        _prediction("Wheat"),
        {
            "sos_date": "2024-07-01",
            "peak_date": "2024-09-01",
            "eos_date": "2024-11-01",
        },
    )

    assert result.predicted_crop == "Other Crop"
    assert result.is_other_crop is True


def test_start_end_month_rules_keep_good_season():
    result = _pipeline_with_rules()._apply_post_prediction_rules(
        _prediction("Wheat"),
        {
            "sos_date": "2024-11-01",
            "peak_date": "2025-02-01",
            "eos_date": "2025-04-01",
        },
    )

    assert result.predicted_crop == "Wheat"
    assert result.is_other_crop is False


def test_peak_month_min_max_rules():
    result = _pipeline_with_rules()._apply_post_prediction_rules(
        _prediction("Paddy"),
        {
            "sos_date": "2024-06-01",
            "peak_date": "2024-07-01",
            "eos_date": "2024-10-01",
        },
    )

    assert result.predicted_crop == "Other Crop"
    assert result.is_other_crop is True
