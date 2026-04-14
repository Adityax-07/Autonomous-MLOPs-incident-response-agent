# mlops-agent/app/schemas.py
"""
Pydantic v2 request / response schemas for the UPI fraud prediction API.
All monetary values are in INR.
"""

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """Single UPI transaction feature vector sent by the client."""

    amount: float = Field(
        ...,
        gt=0,
        le=200_000,
        description="Transaction amount in INR (1 – 200 000)",
        examples=[4999.0],
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day the transaction was initiated (0–23)",
        examples=[14],
    )
    merchant_cat: int = Field(
        ...,
        ge=0,
        le=9,
        description="Merchant category code (0 = groceries … 9 = crypto)",
        examples=[3],
    )
    device_type: int = Field(
        ...,
        ge=0,
        le=2,
        description="Device type: 0 = mobile UPI, 1 = web, 2 = POS terminal",
        examples=[0],
    )
    sender_age_days: int = Field(
        ...,
        ge=0,
        description="Age of sender VPA account in days",
        examples=[365],
    )
    receiver_age_days: int = Field(
        ...,
        ge=0,
        description="Age of receiver VPA account in days",
        examples=[200],
    )
    txn_count_1h: int = Field(
        ...,
        ge=0,
        description="Number of UPI transactions by this sender in the last 1 hour",
        examples=[2],
    )
    same_device: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if current device matches sender's registered device, else 0",
        examples=[1],
    )

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be greater than 0")
        return round(v, 2)


class PredictionResponse(BaseModel):
    """Model's fraud prediction for a single transaction."""

    transaction_id: str = Field(
        ...,
        description="Echo of the client-supplied transaction ID (or server-generated UUID)",
    )
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the transaction is fraudulent (0–1)",
    )
    is_fraud: bool = Field(
        ...,
        description="True if fraud_probability ≥ decision_threshold",
    )
    decision_threshold: float = Field(
        ...,
        description="Threshold used for the binary decision",
    )
    model_version: str = Field(
        ...,
        description="Identifier of the model artefact that produced this prediction",
    )


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    """Exposes model metadata for observability."""

    model_version: str
    feature_columns: list[str]
    decision_threshold: float
    trained_at: str
    metrics: dict[str, float]
