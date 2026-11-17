from pathlib import Path

import pytest

from antifraude import model, pipeline


def metrics(roc_auc: float = 0.95, recall: float = 0.8, pr_auc: float = 0.7) -> model.Metrics:
    return model.Metrics(roc_auc, pr_auc, 0.5, recall, 0.6, 0.5, 1000, 0.03)


@pytest.mark.parametrize(
    ("candidate", "current", "expected"),
    [
        (metrics(), None, True),
        (metrics(roc_auc=0.8), None, False),
        (metrics(recall=0.4), None, False),
        (metrics(pr_auc=0.60), metrics(pr_auc=0.70), False),  # regression
        (metrics(pr_auc=0.695), metrics(pr_auc=0.70), True),  # within tolerance
    ],
)
def test_should_promote(candidate: model.Metrics, current: model.Metrics | None, expected: bool) -> None:
    ok, reason = pipeline.should_promote(candidate, current, min_roc_auc=0.85, min_recall=0.6)
    assert ok is expected, reason


def test_end_to_end_retraining(tmp_path: Path) -> None:
    dataset = pipeline.extract(tmp_path / "train.csv", n=6_000, seed=11)
    result = pipeline.train_and_maybe_promote(dataset, tmp_path / "models")
    assert result["promoted"] is True
    assert (tmp_path / "models" / "current.json").exists()
