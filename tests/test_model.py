from pathlib import Path

from antifraude import data, model
from antifraude.features import NUMERIC_FEATURES, engineer


def test_synthetic_data_has_expected_shape_and_fraud_rate() -> None:
    df = data.generate(n=5_000, seed=1)
    assert len(df) == 5_000
    assert 0.02 < df["is_fraud"].mean() < 0.06
    assert (df["amount"] > 0).all()


def test_engineer_handles_payers_without_history() -> None:
    df = data.generate(n=100, seed=1)
    df.loc[:, "avg_amount_90d"] = 0.0
    feats = engineer(df)
    assert (feats["amount_to_avg_ratio"] == 1.0).all()
    assert feats[NUMERIC_FEATURES].notna().all().all()


def test_model_beats_quality_bar(trained: tuple[Path, model.Metrics]) -> None:
    _, metrics = trained
    assert metrics.roc_auc >= 0.85
    assert metrics.recall >= 0.6
    assert metrics.pr_auc > metrics.fraud_rate * 5  # far better than random


def test_save_and_load_roundtrip(trained: tuple[Path, model.Metrics]) -> None:
    model_dir, metrics = trained
    artifact = model.load_current(model_dir)
    assert artifact is not None
    assert artifact.metrics == metrics
    row = data.generate(n=1, seed=3)
    assert 0.0 <= artifact.score(row) <= 1.0


def test_load_current_returns_none_when_empty(tmp_path: Path) -> None:
    assert model.load_current(tmp_path) is None
