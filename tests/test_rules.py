from datetime import UTC, datetime
from decimal import Decimal

from antifraude.features import PayerHistory
from antifraude.rules import decide, explain
from antifraude.schemas import Channel, Decision, PixKeyType, PixTransaction


def tx(amount: str = "100.00", **kw: object) -> PixTransaction:
    base: dict[str, object] = {
        "transaction_id": "t1",
        "payer_id": "p1",
        "payee_key_type": PixKeyType.CPF,
        "amount": Decimal(amount),
        "channel": Channel.APP,
        "created_at": datetime(2026, 11, 16, 15, tzinfo=UTC),
        "payer_account_age_days": 400,
    }
    base.update(kw)
    return PixTransaction.model_validate(base)


def test_night_limit_blocks_even_with_low_score() -> None:
    assert decide(0.01, tx("1500.00"), hour=2, review=0.5, block=0.85) is Decision.BLOCK
    assert decide(0.01, tx("500.00"), hour=2, review=0.5, block=0.85) is Decision.APPROVE


def test_thresholds() -> None:
    assert decide(0.2, tx(), 14, 0.5, 0.85) is Decision.APPROVE
    assert decide(0.6, tx(), 14, 0.5, 0.85) is Decision.REVIEW
    assert decide(0.9, tx(), 14, 0.5, 0.85) is Decision.BLOCK


def test_explain_lists_risk_signals() -> None:
    risky = tx(
        "5000.00",
        new_device=True,
        first_transfer_to_payee=True,
        payee_key_type=PixKeyType.RANDOM,
        payer_account_age_days=3,
    )
    reasons = explain(risky, PayerHistory(tx_count_24h=12, avg_amount_90d=100.0), hour=1)
    assert set(reasons) == {
        "night_time_transfer",
        "new_device",
        "first_transfer_to_payee",
        "random_key_payee",
        "young_account",
        "amount_far_above_average",
        "burst_of_transfers",
    }
    assert explain(tx(), PayerHistory(), hour=14) == []
