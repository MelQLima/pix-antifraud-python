"""FastAPI application: real-time Pix fraud scoring."""

import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy.orm import Session

from antifraude import __version__, data, model, rules
from antifraude.config import Settings, get_settings
from antifraude.db import ScoredTransaction, TransactionRepository, make_engine, session_factory
from antifraude.features import BRT, to_frame
from antifraude.schemas import PixTransaction, ScoreResponse

log = logging.getLogger("antifraude")

DECISIONS = Counter("antifraude_decisions_total", "Scoring decisions", ["decision"])
LATENCY = Histogram("antifraude_score_seconds", "Scoring latency in seconds")


def load_or_bootstrap_model(settings: Settings) -> model.ModelArtifact:
    artifact = model.load_current(settings.model_dir)
    if artifact is None:
        # First start (e.g. local demo): train on synthetic data so the API works out of the box.
        log.warning("no promoted model found in %s, training a bootstrap model", settings.model_dir)
        pipe, metrics = model.train(data.generate(n=8_000))
        artifact = model.save(pipe, metrics, settings.model_dir)
    return artifact


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        app.state.sessions = session_factory(engine)
        app.state.model = load_or_bootstrap_model(settings)
        log.info("model %s loaded", app.state.model.version)
        yield
        engine.dispose()

    app = FastAPI(title="Antifraude Pix", version=__version__, lifespan=lifespan)

    def get_session(request: Request) -> Iterator[Session]:
        with request.app.state.sessions() as session:
            yield session

    def get_model(request: Request) -> model.ModelArtifact:
        artifact: model.ModelArtifact = request.app.state.model
        return artifact

    @app.post("/v1/score", response_model=ScoreResponse, status_code=status.HTTP_200_OK)
    def score(
        tx: PixTransaction,
        session: Annotated[Session, Depends(get_session)],
        artifact: Annotated[model.ModelArtifact, Depends(get_model)],
    ) -> ScoreResponse:
        started = time.perf_counter()
        repo = TransactionRepository(session)

        # Idempotency: the same transaction id always gets the same answer.
        if (existing := repo.get(tx.transaction_id)) is not None:
            if existing.payer_id != tx.payer_id or existing.amount != tx.amount:
                raise HTTPException(status.HTTP_409_CONFLICT, "transaction_id reused with a different payload")
            return ScoreResponse(
                transaction_id=existing.transaction_id,
                score=existing.score,
                decision=existing.decision,
                reasons=["replayed"],
                model_version=existing.model_version,
            )

        history = repo.history(tx.payer_id, tx.created_at)
        hour = tx.created_at.astimezone(BRT).hour
        prob = round(artifact.score(to_frame(tx, history)), 4)
        decision = rules.decide(prob, tx, hour, settings.review_threshold, settings.block_threshold)
        repo.add(
            ScoredTransaction(
                transaction_id=tx.transaction_id,
                payer_id=tx.payer_id,
                amount=tx.amount,
                created_at=tx.created_at,
                score=prob,
                decision=decision.value,
                model_version=artifact.version,
            )
        )
        DECISIONS.labels(decision.value).inc()
        LATENCY.observe(time.perf_counter() - started)
        return ScoreResponse(
            transaction_id=tx.transaction_id,
            score=prob,
            decision=decision,
            reasons=rules.explain(tx, history, hour),
            model_version=artifact.version,
        )

    @app.get("/v1/model")
    def model_info(artifact: Annotated[model.ModelArtifact, Depends(get_model)]) -> dict[str, object]:
        return {"version": artifact.version, "metrics": artifact.metrics.__dict__}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
