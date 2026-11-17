"""Daily retraining DAG (Apache Airflow 2.x, TaskFlow API).

extract -> train/evaluate -> quality gate -> promote. The heavy lifting lives in
`antifraude.pipeline`, so the same code runs (and is unit tested) without Airflow.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

DATA_DIR = Path("/opt/airflow/data")
MODEL_DIR = Path("/opt/airflow/models")


@dag(
    schedule="0 3 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    tags=["antifraude", "ml"],
)
def antifraude_retrain() -> None:
    @task
    def extract(ds: str | None = None) -> str:
        from antifraude.pipeline import extract as do_extract

        return str(do_extract(DATA_DIR / f"train-{ds}.parquet"))

    @task
    def train_and_promote(dataset: str) -> dict[str, object]:
        from antifraude.pipeline import train_and_maybe_promote

        return train_and_maybe_promote(Path(dataset), MODEL_DIR)

    train_and_promote(extract())


antifraude_retrain()
