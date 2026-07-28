"""
PredictionService — runs ONLY the single selected model (per
configs/model.yaml) for live inference, exactly as the spec requires:
"Run the selected best model (not all 8) for inference in production."

Returns one final prediction + confidence. Never exposes per-algorithm
outputs. If the top class's confidence is below
`unknown_crop_confidence_floor`, returns "Other Crop" instead of forcing
one of the trained classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.config.loader import ModelConfig
from backend.models.registry import ModelRegistry

OTHER_CROP_LABEL = "Other Crop"


@dataclass(frozen=True)
class PredictionResult:
    predicted_crop: str
    confidence: float
    class_probabilities: dict[str, float]
    is_other_crop: bool


class PredictionService:
    def __init__(self, config: ModelConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry

    def predict(
        self,
        raw_features: dict[str, Any],
        smooth_features: dict[str, Any],
        feature_set: str | None = None,
    ) -> PredictionResult:
        feature_set = feature_set or self.config.selected_feature_set
        model = self.registry.get_selected_model(feature_set)
        label_encoder = self.registry.label_encoder

        features = smooth_features if feature_set == "smooth" else raw_features

        # Column order must match what the pipeline's imputer/scaler were
        # fit on. sklearn Pipelines store input feature names on the first
        # step when fit with a DataFrame, so we reuse that order here
        # rather than relying on dict insertion order.
        try:
            feature_names = list(model.named_steps["imputer"].feature_names_in_)
        except AttributeError:
            feature_names = list(features.keys())

        row = pd.DataFrame([{name: features.get(name, np.nan) for name in feature_names}])

        probabilities = self._predict_proba(model, row)[0]
        class_names = list(label_encoder.classes_)

        top_idx = int(np.argmax(probabilities))
        top_class = class_names[top_idx]
        top_confidence = float(probabilities[top_idx])

        is_other_crop = top_confidence < self.config.unknown_crop_confidence_floor

        return PredictionResult(
            predicted_crop=OTHER_CROP_LABEL if is_other_crop else top_class,
            confidence=top_confidence,
            class_probabilities=dict(zip(class_names, (float(p) for p in probabilities))),
            is_other_crop=is_other_crop,
        )

    @staticmethod
    def _predict_proba(model: Any, row: pd.DataFrame) -> np.ndarray:
        if not isinstance(model, Pipeline):
            return model.predict_proba(row)

        transformed: Any = row
        for _, step in model.steps[:-1]:
            transformed = step.transform(transformed)

        final_estimator = model.steps[-1][1]
        final_feature_names = getattr(final_estimator, "feature_names_in_", None)
        if final_feature_names is not None and len(final_feature_names) == transformed.shape[1]:
            transformed = pd.DataFrame(transformed, columns=list(final_feature_names), index=row.index)

        return final_estimator.predict_proba(transformed)
