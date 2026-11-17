"""Retraining pipeline steps. Pure functions so they can run in Airflow,
in a cron job or in tests without any orchestrator installed."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from antifraude import data, model
from antifraude.config import get_settings

log = logging.getLogger(__name__)


def extract(path: Path, n: int = 20_000, seed: int = 42) -> Path:
    """Write the training dataset. Production would query the labelled
    chargeback/MED (Mecanismo Especial de Devolução) table instead."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = data.generate(n=n, seed=seed)
    if path.suffix == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)
    return path


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def should_promote(
    candidate: model.Metrics, current: model.Metrics | None, min_roc_auc: float, min_recall: float
) -> tuple[bool, str]:
    """Quality gate: absolute minimums plus no regression against production."""
    if candidate.roc_auc < min_roc_auc:
        return False, f"roc_auc {candidate.roc_auc} below minimum {min_roc_auc}"
    if candidate.recall < min_recall:
        return False, f"recall {candidate.recall} below minimum {min_recall}"
    if current is not None and candidate.pr_auc < current.pr_auc - 0.01:
        return False, f"pr_auc regressed from {current.pr_auc} to {candidate.pr_auc}"
    return True, "candidate approved"


def train_and_maybe_promote(dataset: Path, model_dir: Path) -> dict[str, object]:
    settings = get_settings()
    pipe, metrics = model.train(load(dataset))
    current = model.load_current(model_dir)
    ok, reason = should_promote(
        metrics, current.metrics if current else None, settings.min_roc_auc, settings.min_recall
    )
    result: dict[str, object] = {"promoted": ok, "reason": reason, "metrics": asdict(metrics)}
    if ok:
        result["version"] = model.save(pipe, metrics, model_dir).version
    log.info("training finished: %s", json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud model and promote it if it passes the gate.")
    parser.add_argument("--dataset", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--model-dir", type=Path, default=get_settings().model_dir)
    parser.add_argument("--rows", type=int, default=20_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    extract(args.dataset, n=args.rows)
    print(json.dumps(train_and_maybe_promote(args.dataset, args.model_dir), indent=2))


if __name__ == "__main__":
    main()
