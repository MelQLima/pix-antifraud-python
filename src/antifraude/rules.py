"""Deterministic rules combined with the model score.

Rules encode regulatory/business constraints that must hold regardless of the
model (e.g. Banco Central night-time Pix limits) and produce human-readable
reasons for analysts.
"""

from decimal import Decimal

from antifraude.features import PayerHistory
from antifraude.schemas import Decision, PixKeyType, PixTransaction

NIGHT_LIMIT = Decimal("1000.00")  # BCB: default night-time limit for individuals
BURST_LIMIT = 10


def explain(tx: PixTransaction, history: PayerHistory, hour: int) -> list[str]:
    reasons: list[str] = []
    if 0 <= hour < 6 or hour >= 20:
        reasons.append("night_time_transfer")
    if tx.new_device:
        reasons.append("new_device")
    if tx.first_transfer_to_payee:
        reasons.append("first_transfer_to_payee")
    if tx.payee_key_type is PixKeyType.RANDOM:
        reasons.append("random_key_payee")
    if tx.payer_account_age_days < 30:
        reasons.append("young_account")
    if history.avg_amount_90d > 0 and float(tx.amount) > 5 * history.avg_amount_90d:
        reasons.append("amount_far_above_average")
    if history.tx_count_24h >= BURST_LIMIT:
        reasons.append("burst_of_transfers")
    return reasons


def decide(score: float, tx: PixTransaction, hour: int, review: float, block: float) -> Decision:
    night = 0 <= hour < 6 or hour >= 20
    if night and tx.amount > NIGHT_LIMIT:
        return Decision.BLOCK
    if score >= block:
        return Decision.BLOCK
    if score >= review:
        return Decision.REVIEW
    return Decision.APPROVE
