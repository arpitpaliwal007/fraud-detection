from fastapi import FastAPI
from pydantic import BaseModel, Field

from .core import LogisticFraudModel, generate_transactions

model = LogisticFraudModel().fit(generate_transactions(5000))
threshold = 0.62
app = FastAPI(title="Fraud Scoring API", version="0.1.0")


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    log_amount: float
    hour_risk: int = Field(ge=0, le=1)
    distance: float = Field(ge=0)
    velocity_1h: int = Field(ge=0)
    new_device: int = Field(ge=0, le=1)
    foreign: int = Field(ge=0, le=1)
    merchant_risk: float = Field(ge=0, le=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score")
def score(transaction: Transaction) -> dict:
    row = transaction.model_dump()
    probability = model.predict_proba(row)
    return {"fraud_probability": round(probability, 6), "decision": "review" if probability >= threshold else "approve",
            "threshold": threshold, "explanation": model.explain(row)}

