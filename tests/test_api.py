from typing import Any

from fastapi.testclient import TestClient


def payload(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "transaction_id": "tx-1",
        "payer_id": "payer-1",
        "payee_key_type": "cpf",
        "amount": "120.50",
        "channel": "app",
        "created_at": "2026-11-16T14:30:00-03:00",
        "payer_account_age_days": 900,
        "new_device": False,
        "first_transfer_to_payee": False,
    }
    body.update(kw)
    return body


def test_low_risk_transfer_is_approved(client: TestClient) -> None:
    r = client.post("/v1/score", json=payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == "approve"
    assert body["reasons"] == []
    assert 0 <= body["score"] < 0.5


def test_account_takeover_pattern_is_not_approved(client: TestClient) -> None:
    for i in range(3):  # build some normal history for the payer
        client.post("/v1/score", json=payload(transaction_id=f"h{i}", amount="80.00"))
    r = client.post(
        "/v1/score",
        json=payload(
            transaction_id="ato",
            amount="2900.00",
            created_at="2026-11-16T02:10:00-03:00",
            new_device=True,
            first_transfer_to_payee=True,
            payee_key_type="random",
        ),
    )
    body = r.json()
    assert body["decision"] == "block"
    assert {"night_time_transfer", "new_device", "amount_far_above_average"} <= set(body["reasons"])


def test_scoring_is_idempotent(client: TestClient) -> None:
    first = client.post("/v1/score", json=payload(transaction_id="same")).json()
    second = client.post("/v1/score", json=payload(transaction_id="same")).json()
    assert second["score"] == first["score"]
    assert second["decision"] == first["decision"]
    assert second["reasons"] == ["replayed"]

    conflict = client.post("/v1/score", json=payload(transaction_id="same", amount="999.00"))
    assert conflict.status_code == 409


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/v1/score", json=payload(amount="-1")).status_code == 422
    assert client.post("/v1/score", json=payload(created_at="2026-11-16T14:30:00")).status_code == 422
    assert client.post("/v1/score", json=payload(payee_key_type="boleto")).status_code == 422


def test_ops_endpoints(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert "roc_auc" in client.get("/v1/model").json()["metrics"]
    client.post("/v1/score", json=payload(transaction_id="m1"))
    assert "antifraude_decisions_total" in client.get("/metrics").text
