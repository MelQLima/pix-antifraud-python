"""Synthetic labelled Pix transactions with realistic fraud patterns.

Patterns: account-takeover (new device, night, amount far above the payer's
average), mule accounts (young accounts, burst of transfers) and social
engineering (first transfer to a random key, high amount).
"""

import numpy as np
import pandas as pd

from antifraude.features import RAW_COLUMNS


def generate(n: int = 20_000, fraud_rate: float = 0.03, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate

    avg = rng.lognormal(mean=5.0, sigma=0.8, size=n)  # ~R$150 typical average
    legit_amount = avg * rng.lognormal(0, 0.5, n)
    fraud_amount = avg * rng.uniform(3, 25, n)
    amount = np.where(is_fraud, fraud_amount, legit_amount).round(2)

    legit_hour = rng.choice(24, n, p=_daytime_distribution())
    fraud_hour = np.where(rng.random(n) < 0.6, rng.integers(0, 6, n), rng.integers(0, 24, n))
    hour = np.where(is_fraud, fraud_hour, legit_hour)

    age = np.where(is_fraud & (rng.random(n) < 0.5), rng.integers(0, 30, n), rng.integers(30, 3000, n))
    tx_count = np.where(is_fraud, rng.poisson(6, n), rng.poisson(1.5, n))
    new_device = np.where(is_fraud, rng.random(n) < 0.7, rng.random(n) < 0.05)
    first_payee = np.where(is_fraud, rng.random(n) < 0.85, rng.random(n) < 0.25)

    key_types = np.array(["cpf", "cnpj", "email", "phone", "random"])
    key_type = np.where(
        is_fraud,
        rng.choice(key_types, n, p=[0.15, 0.05, 0.1, 0.2, 0.5]),
        rng.choice(key_types, n, p=[0.35, 0.15, 0.15, 0.25, 0.1]),
    )
    channel = rng.choice(np.array(["app", "web", "api"]), n, p=[0.8, 0.15, 0.05])

    df = pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "account_age_days": age,
            "tx_count_24h": tx_count,
            "avg_amount_90d": np.where(rng.random(n) < 0.05, 0.0, avg.round(2)),
            "new_device": new_device.astype(int),
            "first_transfer_to_payee": first_payee.astype(int),
            "key_type": key_type,
            "channel": channel,
        },
        columns=RAW_COLUMNS,
    )
    # Label noise: some fraud looks legitimate and vice versa.
    flip = rng.random(n) < 0.01
    df["is_fraud"] = np.where(flip, ~is_fraud, is_fraud).astype(int)
    return df


def _daytime_distribution() -> np.ndarray:
    weights = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 9, 10, 10, 9, 9, 9, 9, 9, 8, 7, 5, 3, 2], dtype=float)
    return weights / weights.sum()
