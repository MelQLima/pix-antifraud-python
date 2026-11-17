"""Request/response contracts (Pydantic v2)."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PixKeyType(StrEnum):
    CPF = "cpf"
    CNPJ = "cnpj"
    EMAIL = "email"
    PHONE = "phone"
    RANDOM = "random"


class Channel(StrEnum):
    APP = "app"
    WEB = "web"
    API = "api"


class Decision(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"


class PixTransaction(BaseModel):
    """A Pix transfer to be scored before it is sent to the SPI."""

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    transaction_id: str = Field(min_length=1, max_length=64)
    payer_id: str = Field(min_length=1, max_length=64)
    payee_key_type: PixKeyType
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2, description="Amount in BRL")
    channel: Channel
    created_at: datetime
    payer_account_age_days: int = Field(ge=0)
    new_device: bool = False
    first_transfer_to_payee: bool = False

    @field_validator("created_at")
    @classmethod
    def must_be_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must include a timezone offset")
        return v


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float = Field(ge=0, le=1)
    decision: Decision
    reasons: list[str]
    model_version: str
