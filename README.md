# antifraude-pix-python

Real-time **Pix fraud scoring API** in **Python**: **FastAPI** + **scikit-learn** model + deterministic risk rules, persisted with **SQLAlchemy 2.0** (PostgreSQL) and retrained daily by an **Apache Airflow** DAG with an automatic quality gate.

Portfolio project showing production-style Python backend and applied ML engineering (MLOps basics): typed APIs, train/serve feature parity, model versioning, evaluation metrics for imbalanced data, CI with linting, strict typing and tests.

## Highlights

| Area | What is implemented |
|---|---|
| API | FastAPI with dependency injection, lifespan startup, **Pydantic v2** validation (timezone-aware datetimes, `Decimal` money), OpenAPI docs at `/docs` |
| ML | scikit-learn `Pipeline` (`FunctionTransformer` feature engineering → `ColumnTransformer` with `StandardScaler` + `OneHotEncoder` → `HistGradientBoostingClassifier` with class weighting for a ~3% fraud rate) |
| Evaluation | Stratified split; **ROC-AUC, PR-AUC (average precision), precision, recall, F1** |
| MLOps | Immutable model versions (`joblib` + JSON metadata), `current.json` pointer, promotion gate (minimum ROC-AUC/recall and no PR-AUC regression vs production) |
| Orchestration | **Airflow 2 TaskFlow** DAG (`dags/antifraude_retrain.py`) — extract → train/evaluate → gate → promote; the logic lives in plain functions so it is unit tested without Airflow |
| Features | Velocity features from history (transfers in 24h, 90-day average amount), amount-to-average ratio, night time, new device, first transfer to payee, Pix key type, channel |
| Rules | Banco Central night-time limit, explainable `reasons` returned to analysts, `approve` / `review` / `block` decisions |
| Data | **SQLAlchemy 2.0** typed ORM (`Mapped`, `mapped_column`), PostgreSQL via psycopg 3 (SQLite for local runs) |
| Reliability | Idempotent scoring by `transaction_id` (replays get the same answer; payload mismatch → `409`) |
| Observability | Prometheus metrics (`/metrics`: decisions counter, latency histogram), `/health` |
| Quality | **pytest** (+ coverage gate), **ruff** (lint + format), **mypy --strict**, GitHub Actions matrix (3.11, 3.12), Docker multi-stage non-root image, Docker Compose |

## Run

```bash
python -m venv .venv && . .venv/bin/activate
make install
make run            # http://localhost:8000/docs (trains a bootstrap model on first start)
make train          # train on 50k synthetic rows and promote if it passes the gate
```

```bash
curl -X POST localhost:8000/v1/score -H 'Content-Type: application/json' -d '{
  "transaction_id": "tx-123", "payer_id": "payer-1", "payee_key_type": "random",
  "amount": "2900.00", "channel": "app", "created_at": "2026-11-16T02:10:00-03:00",
  "payer_account_age_days": 12, "new_device": true, "first_transfer_to_payee": true }'
```

```json
{"transaction_id":"tx-123","score":0.995,"decision":"block",
 "reasons":["night_time_transfer","new_device","first_transfer_to_payee","random_key_payee","young_account"],
 "model_version":"20261117T041541699282Z"}
```

With Docker: `docker compose up --build` (API + PostgreSQL) and `docker compose run --rm trainer` to retrain.

## Model results (50k synthetic transactions, 3.9% fraud, hold-out 25%)

| ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.88 | 0.78 | 0.92 | 0.77 | 0.84 |

The synthetic generator (`antifraude/data.py`) simulates account takeover, mule accounts and social engineering, plus 1% label noise, so metrics are realistic rather than perfect.

## Project layout

```
src/antifraude/
  api.py        FastAPI app and endpoints
  schemas.py    Pydantic contracts
  features.py   feature engineering shared by training and serving
  model.py      pipeline, metrics, versioned persistence
  rules.py      deterministic rules and explanations
  db.py         SQLAlchemy models and repository (velocity queries)
  pipeline.py   retraining steps + CLI (antifraude-train)
dags/           Airflow DAG
tests/          pytest suite
```
