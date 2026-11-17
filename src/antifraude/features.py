"""Feature engineering shared by training and online scoring (no train/serve skew)."""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from antifraude.schemas import PixTransaction

BRT = ZoneInfo("America/Sao_Paulo")

RAW_COLUMNS = [
    "amount",
    "hour",
    "account_age_days",
    "tx_count_24h",
    "avg_amount_90d",
    "new_device",
    "first_transfer_to_payee",
    "key_type",
    "channel",
]
NUMERIC_FEATURES = [
    "log_amount",
    "amount_to_avg_ratio",
    "account_age_days",
    "tx_count_24h",
    "is_night",
    "new_device",
    "first_transfer_to_payee",
]
CATEGORICAL_FEATURES = ["key_type", "channel"]


@dataclass(frozen=True)
class PayerHistory:
    """Velocity aggregates for the payer, computed from previous transactions."""

    tx_count_24h: int = 0
    avg_amount_90d: float = 0.0


def to_frame(tx: PixTransaction, history: PayerHistory) -> pd.DataFrame:
    """Build the raw single-row frame expected by the model pipeline."""
    local_hour = tx.created_at.astimezone(BRT).hour
    return pd.DataFrame(
        [
            {
                "amount": float(tx.amount),
                "hour": local_hour,
                "account_age_days": tx.payer_account_age_days,
                "tx_count_24h": history.tx_count_24h,
                "avg_amount_90d": history.avg_amount_90d,
                "new_device": int(tx.new_device),
                "first_transfer_to_payee": int(tx.first_transfer_to_payee),
                "key_type": tx.payee_key_type.value,
                "channel": tx.channel.value,
            }
        ],
        columns=RAW_COLUMNS,
    )


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw columns. Pure and vectorised."""
    out = pd.DataFrame(index=df.index)
    amount = df["amount"].astype(float)
    avg = df["avg_amount_90d"].astype(float)
    out["log_amount"] = np.log1p(amount)
    # Payers without history get ratio 1 (neutral) instead of infinity.
    out["amount_to_avg_ratio"] = np.where(avg > 0, amount / avg.where(avg > 0, 1.0), 1.0).clip(0, 50)
    out["account_age_days"] = df["account_age_days"].astype(float)
    out["tx_count_24h"] = df["tx_count_24h"].astype(float)
    out["is_night"] = df["hour"].astype(int).between(0, 5).astype(int)
    out["new_device"] = df["new_device"].astype(int)
    out["first_transfer_to_payee"] = df["first_transfer_to_payee"].astype(int)
    out["key_type"] = df["key_type"].astype(str)
    out["channel"] = df["channel"].astype(str)
    return out
