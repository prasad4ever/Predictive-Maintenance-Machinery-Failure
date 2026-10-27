"""Compare three classifiers and keep the one that catches urgent failures."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.config import get_settings
from data.catalog import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PRIORITIES,
    TRAIN_DAYS,
    VALIDATION_DAYS,
)
from data.prepare import TELEMETRY_PATH

CLASSES = list(PRIORITIES)


def main() -> None:
    if not TELEMETRY_PATH.exists():
        from data.prepare import main as prepare

        prepare()
    frame = pd.read_csv(TELEMETRY_PATH)
    train_end = TRAIN_DAYS
    validation_end = TRAIN_DAYS + VALIDATION_DAYS
    train = frame[frame["day_index"] < train_end]
    validation = frame[(frame["day_index"] >= train_end) & (frame["day_index"] < validation_end)]
    test = frame[frame["day_index"] >= validation_end]
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
    }
    reports = []
    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None
    for name, estimator in candidates.items():
        pipeline = _pipeline(estimator)
        pipeline.fit(train[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)], train["priority"])
        validation_pred = pipeline.predict(validation[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)])
        score = _selection_score(validation["priority"], validation_pred)
        test_pred = pipeline.predict(test[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)])
        report = {
            "model": name,
            "validation_score": round(score, 4),
            "validation": _class_metrics(validation["priority"], validation_pred),
            "test": _class_metrics(test["priority"], test_pred),
        }
        reports.append(report)
        print(
            f"{name}: validation score {score:.3f} | "
            f"test P1 recall {report['test']['P1']['recall']:.3f} | "
            f"test macro F1 {report['test']['macro_f1']:.3f}"
        )
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipeline
    if best_pipeline is None:
        raise RuntimeError("no model was trained")
    test_pred = best_pipeline.predict(test[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)])
    text_report = classification_report(
        test["priority"],
        test_pred,
        labels=CLASSES,
        zero_division=0,
    )
    numeric = train[list(NUMERIC_FEATURES)].to_numpy(dtype=np.float64)
    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "numeric_features": list(NUMERIC_FEATURES),
        "numeric_mean": numeric.mean(axis=0),
        "numeric_scale": np.maximum(numeric.std(axis=0), 1e-6),
        "selection_score": best_score,
    }
    settings = get_settings()
    destination = settings.resolve(settings.model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    comparison = {
        "chosen_model": best_name,
        "selection_rule": "0.55*P1 F1 + 0.25*P2 F1 + 0.20*macro F1, measured on the time-held validation window",
        "validation_score": round(best_score, 4),
        "models": reports,
        "test_report": text_report,
        "rows": {"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
    }
    comparison_path = settings.resolve(settings.comparison_path)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Chose {best_name}")
    print(text_report)
    print(f"Wrote {destination}")


def _pipeline(estimator: LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier) -> Pipeline:
    transform = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline([("transform", transform), ("model", estimator)])


def _selection_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    per_class = f1_score(y_true, y_pred, labels=["P1", "P2"], average=None, zero_division=0)
    macro = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    return 0.55 * float(per_class[0]) + 0.25 * float(per_class[1]) + 0.20 * float(macro)


def _class_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    recall = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    precision = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    per_class = {}
    for index, name in enumerate(CLASSES):
        per_class[name] = {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "support": int((y_true == name).sum()),
        }
    per_class["macro_f1"] = round(float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)), 4)
    return per_class


if __name__ == "__main__":
    main()
