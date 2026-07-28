"""
ModelRegistry — loads the 8 pre-trained artifacts (4 algorithms x
{raw, smooth}) plus the label encoder from model_weights/. It does NOT
train anything; training already happened in the reference notebook and
the resulting .joblib files are expected to already exist on disk (see
model_weights/README.md).

ModelSelector picks which of the 8 loaded models is "production" purely
from configs/model.yaml, so swapping the selected model never requires a
code change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from backend.config.loader import ModelConfig
from backend.utils.exceptions import ModelArtifactNotFoundError
from backend.utils.logging import get_logger

logger = get_logger(__name__)

ALGORITHMS = ["RandomForest", "XGBoost", "LightGBM", "SVM"]


class ModelRegistry:
    def __init__(self, config: ModelConfig, base_dir: Path):
        self.config = config
        self.weights_dir = base_dir / config.model_weights_dir
        self._smooth_models: dict[str, Any] = {}
        self._raw_models: dict[str, Any] = {}
        self._label_encoder: Any = None

    def load_all(self) -> None:
        """Load model artifacts needed by the current configuration."""
        if not self.config.enable_agreement_check:
            self._load_model(self.config.selected_algorithm, self.config.selected_feature_set)
            other_feature_set = "raw" if self.config.selected_feature_set == "smooth" else "smooth"
            self._load_model(self.config.selected_algorithm, other_feature_set, required=False)
            self._load_label_encoder()
            return

        for algo in ALGORITHMS:
            self._load_model(algo, "smooth", required=False)
            self._load_model(algo, "raw", required=False)

        self._load_label_encoder()

    def _load_model(self, algorithm: str, feature_set: str, required: bool = True) -> None:
        template = (
            self.config.smooth_model_file_tpl
            if feature_set == "smooth"
            else self.config.raw_model_file_tpl
        )
        path = self.weights_dir / template.format(algorithm=algorithm)

        if path.exists():
            store = self._smooth_models if feature_set == "smooth" else self._raw_models
            store[algorithm] = joblib.load(path)
            return

        if required:
            raise ModelArtifactNotFoundError(
                f"Selected model '{algorithm}' ({feature_set}) not loaded. "
                f"Expected file '{path.name}' in {self.weights_dir}"
            )
        logger.info("Optional model artifact not found, skipping: %s", path)

    def _load_label_encoder(self) -> None:
        encoder_path = self.weights_dir / self.config.label_encoder_file
        if encoder_path.exists():
            self._label_encoder = joblib.load(encoder_path)
            return

        raise ModelArtifactNotFoundError(
            f"label_encoder.joblib not found in {self.weights_dir}"
        )

    @property
    def label_encoder(self) -> Any:
        if self._label_encoder is None:
            raise ModelArtifactNotFoundError(
                f"label_encoder.joblib not found in {self.weights_dir}"
            )
        return self._label_encoder

    def get_selected_model(self, feature_set: str | None = None) -> Any:
        """Returns the sklearn Pipeline for whichever
        (algorithm, feature_set) is configured as production in
        configs/model.yaml."""
        algo = self.config.selected_algorithm
        feature_set = feature_set or self.config.selected_feature_set

        store = self._smooth_models if feature_set == "smooth" else self._raw_models
        model = store.get(algo)

        if model is None:
            expected_name = (
                self.config.smooth_model_file_tpl
                if feature_set == "smooth"
                else self.config.raw_model_file_tpl
            ).format(algorithm=algo)
            raise ModelArtifactNotFoundError(
                f"Selected model '{algo}' ({feature_set}) not loaded. "
                f"Expected file '{expected_name}' in {self.weights_dir}"
            )
        return model

    def get_model(self, algorithm: str, feature_set: str) -> Any:
        """Escape hatch for the optional agreement-check module — fetch any
        of the 8 loaded models explicitly, rather than only the selected
        production one."""
        store = self._smooth_models if feature_set == "smooth" else self._raw_models
        model = store.get(algorithm)
        if model is None:
            raise ModelArtifactNotFoundError(f"Model '{algorithm}' ({feature_set}) not loaded.")
        return model
