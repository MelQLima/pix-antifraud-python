"""Persistence with SQLAlchemy 2.0 (typed ORM, PostgreSQL in production, SQLite locally)."""

from collections.abc import Iterator
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import DateTime, Engine, Numeric, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from antifraude.features import PayerHistory


class Base(DeclarativeBase):
    pass


class ScoredTransaction(Base):
    __tablename__ = "scored_transactions"

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    payer_id: Mapped[str] = mapped_column(String(64), index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    score: Mapped[float]
    decision: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(32))


def make_engine(url: str) -> Engine:
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


class TransactionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def history(self, payer_id: str, now: datetime) -> PayerHistory:
        """Velocity features: transfers in the last 24h and 90-day average amount."""
        count_24h = self.session.scalar(
            select(func.count())
            .select_from(ScoredTransaction)
            .where(
                ScoredTransaction.payer_id == payer_id,
                ScoredTransaction.created_at >= now - timedelta(hours=24),
            )
        )
        avg_90d = self.session.scalar(
            select(func.avg(ScoredTransaction.amount)).where(
                ScoredTransaction.payer_id == payer_id,
                ScoredTransaction.created_at >= now - timedelta(days=90),
                ScoredTransaction.decision != "block",
            )
        )
        return PayerHistory(tx_count_24h=int(count_24h or 0), avg_amount_90d=float(avg_90d or 0.0))

    def get(self, transaction_id: str) -> ScoredTransaction | None:
        return self.session.get(ScoredTransaction, transaction_id)

    def add(self, row: ScoredTransaction) -> None:
        self.session.add(row)
        self.session.commit()


def iter_session(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as session:
        yield session
