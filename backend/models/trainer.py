"""
ModelTrainer — offline training, ported feature-for-feature from the
notebook's "MODEL BUILDING" section.

This module is NOT imported or run by the FastAPI app. The app only reads
already-trained artifacts from model_weights/ via ModelRegistry. Run this
script by hand only when you need to retrain on new data:

    python -m backend.models.trainer \\
        --smooth-features lifecycle_features_smooth.csv \\
        --raw-features lifecycle_features_raw.csv \\
        --output-dir model_weights

It reproduces exactly:
  * 4 classifiers (RandomForest, XGBoost, LightGBM, SVM) x 2 feature sets
    (raw, smooth) = 8 model runs, NOT a raw+smooth fusion.
  * SimpleImputer(median) -> (StandardScaler for SVM only) -> classifier
  * GridSearchCV, StratifiedKFold(5), scoring="f1_macro"
  * 80/20 stratified train/test split
  * Best-model selection: sort by Smooth_Test_Macro_F1, then
    Smooth_Test_Acc; top row becomes configs/model.yaml's
    `selected_algorithm` (you update that file by hand after reviewing the
    comparison table this script prints).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from backend.utils.logging import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42
DROP_COLS = ["Farm_ID", "Crop", "GroundTruth_Month", "SOS_Date", "Peak_Date", "EOS_Date", "Status"]

MODEL_CONFIGS = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "pipeline_type": "tree",
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="mlogloss", objective="multi:softprob", n_jobs=-1
        ),
        "pipeline_type": "tree",
        "param_grid": {
            "model__n_estimators": [150, 300],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
    },
    "LightGBM": {
        "estimator": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1),
        "pipeline_type": "tree",
        "param_grid": {
            "model__n_estimators": [150, 300],
            "model__num_leaves": [15, 31],
            "model__learning_rate": [0.05, 0.1],
        },
    },
    "SVM": {
        "estimator": SVC(probability=True, random_state=RANDOM_STATE),
        "pipeline_type": "svm",
        "param_grid": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale"],
        },
    },
}


def make_pipeline(model, pipeline_type: str) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if pipeline_type == "svm":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


class ModelTrainer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _train_one(self, X_train, y_train, X_test, y_test, name, config, cv, class_names, dataset_label):
        logger.info("Training %s on %s features...", name, dataset_label)
        pipeline = make_pipeline(config["estimator"], config["pipeline_type"])
        grid_search = GridSearchCV(
            pipeline, param_grid=config["param_grid"], cv=cv, scoring="f1_macro", n_jobs=-1, refit=True
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_idx = grid_search.best_index_
        cv_mean = grid_search.cv_results_["mean_test_score"][best_idx]
        cv_std = grid_search.cv_results_["std_test_score"][best_idx]

        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average="macro")
        test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0
        )

        logger.info("%s | %s | CV macroF1=%.3f test macroF1=%.3f\n%s", name, dataset_label, cv_mean, test_f1_macro, report)

        return {
            "dataset": dataset_label,
            "model_name": name,
            "best_model": best_model,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "test_acc": test_acc,
            "test_f1_macro": test_f1_macro,
            "test_f1_weighted": test_f1_weighted,
        }

    def run(self, smooth_features_csv: Path, raw_features_csv: Path) -> pd.DataFrame:
        smooth_data = pd.read_csv(smooth_features_csv)
        raw_data = pd.read_csv(raw_features_csv)

        label_encoder = LabelEncoder()
        label_encoder.fit(smooth_data["Crop"])
        class_names = list(label_encoder.classes_)

        X_smooth = smooth_data.drop(columns=DROP_COLS, errors="ignore")
        y_smooth = label_encoder.transform(smooth_data["Crop"])
        X_raw = raw_data.drop(columns=DROP_COLS, errors="ignore")
        y_raw = label_encoder.transform(raw_data["Crop"])

        X_smooth = X_smooth.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_raw = X_raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

        X_train_smooth, X_test_smooth, y_train_smooth, y_test_smooth = train_test_split(
            X_smooth, y_smooth, test_size=0.2, random_state=RANDOM_STATE, stratify=y_smooth
        )
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_raw
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        results_smooth, results_raw = {}, {}
        for name, config in MODEL_CONFIGS.items():
            results_smooth[name] = self._train_one(
                X_train_smooth, y_train_smooth, X_test_smooth, y_test_smooth, name, config, cv, class_names, "Smooth"
            )
            results_raw[name] = self._train_one(
                X_train_raw, y_train_raw, X_test_raw, y_test_raw, name, config, cv, class_names, "Raw"
            )

        for name in MODEL_CONFIGS:
            joblib.dump(results_smooth[name]["best_model"], self.output_dir / f"model_smooth_{name}.joblib")
            joblib.dump(results_raw[name]["best_model"], self.output_dir / f"model_raw_{name}.joblib")
        joblib.dump(label_encoder, self.output_dir / "label_encoder.joblib")

        comparison_df = (
            pd.DataFrame(
                [
                    {
                        "Model": name,
                        "Smooth_Test_Macro_F1": results_smooth[name]["test_f1_macro"],
                        "Smooth_Test_Acc": results_smooth[name]["test_acc"],
                        "Raw_Test_Macro_F1": results_raw[name]["test_f1_macro"],
                        "Raw_Test_Acc": results_raw[name]["test_acc"],
                    }
                    for name in MODEL_CONFIGS
                ]
            )
            .sort_values(["Smooth_Test_Macro_F1", "Smooth_Test_Acc"], ascending=False)
            .reset_index(drop=True)
        )
        logger.info("Comparison table:\n%s", comparison_df.to_string())
        logger.info(
            "Update configs/model.yaml's selected_algorithm to '%s' if it isn't already.",
            comparison_df.iloc[0]["Model"],
        )
        return comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-features", required=True, type=Path)
    parser.add_argument("--raw-features", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("model_weights"), type=Path)
    args = parser.parse_args()

    trainer = ModelTrainer(args.output_dir)
    trainer.run(args.smooth_features, args.raw_features)
