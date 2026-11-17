"""Model pipeline, evaluation metrics and versioned persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from antifraude.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_COLUMNS, engineer

TARGET = "is_fraud"


@dataclass(frozen=True)
class Metrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    n_test: int
    fraud_rate: float


def build_pipeline(seed: int = 42) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        class_weight="balanced",  # fraud is rare: ~3% of transactions
        random_state=seed,
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer)),
            ("preprocess", preprocess),
            ("model", clf),
        ]
    )


def train(df: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, Metrics]:
    """Stratified split, fit, and evaluate on the hold-out set."""
    x_train, x_test, y_train, y_test = train_test_split(
        df[RAW_COLUMNS], df[TARGET], test_size=0.25, stratify=df[TARGET], random_state=seed
    )
    pipe = build_pipeline(seed).fit(x_train, y_train)
    return pipe, evaluate(pipe, x_test, y_test)


def evaluate(pipe: Pipeline, x: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Metrics:
    proba = pipe.predict_proba(x)[:, 1]
    pred = (proba >= threshold).astype(int)
    return Metrics(
        roc_auc=round(float(roc_auc_score(y, proba)), 4),
        pr_auc=round(float(average_precision_score(y, proba)), 4),
        precision=round(float(precision_score(y, pred, zero_division=0)), 4),
        recall=round(float(recall_score(y, pred)), 4),
        f1=round(float(f1_score(y, pred)), 4),
        threshold=threshold,
        n_test=len(y),
        fraud_rate=round(float(np.mean(y)), 4),
    )


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: Pipeline
    version: str
    metrics: Metrics

    def score(self, frame: pd.DataFrame) -> float:
        return float(self.pipeline.predict_proba(frame[RAW_COLUMNS])[0, 1])


def save(pipe: Pipeline, metrics: Metrics, model_dir: Path) -> ModelArtifact:
    """Persist a new immutable version and point `current.json` at it."""
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / f"model-{version}.joblib")
    meta: dict[str, Any] = {"version": version, "metrics": asdict(metrics)}
    (model_dir / f"model-{version}.json").write_text(json.dumps(meta, indent=2))
    (model_dir / "current.json").write_text(json.dumps(meta, indent=2))
    return ModelArtifact(pipe, version, metrics)


def load_current(model_dir: Path) -> ModelArtifact | None:
    pointer = model_dir / "current.json"
    if not pointer.exists():
        return None
    meta = json.loads(pointer.read_text())
    pipe = joblib.load(model_dir / f"model-{meta['version']}.joblib")
    return ModelArtifact(pipe, meta["version"], Metrics(**meta["metrics"]))
