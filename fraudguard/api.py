"""REST API for real-time fraud scoring.

    uvicorn fraudguard.api:app --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from fraudguard.scorer import Scorer

scorer: Scorer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer
    scorer = Scorer()
    yield
    scorer.con.close()


app = FastAPI(title="FraudGuard", version="0.1.0", lifespan=lifespan)


class Transaction(BaseModel):
    """Raw IEEE-CIS transaction fields. Extra columns (C1..C14, D*, M*) are accepted as-is."""

    model_config = ConfigDict(extra="allow", json_schema_extra={"examples": [{
        "TransactionID": 4000001,
        "TransactionDT": 15811200,
        "TransactionAmt": 249.0,
        "ProductCD": "W",
        "card1": 13926,
        "card4": "visa",
        "card6": "debit",
        "addr1": 315,
        "P_emaildomain": "gmail.com",
        "C1": 3,
        "C13": 5,
        "D1": 14,
        "D15": 14,
    }]})

    TransactionID: int
    TransactionDT: int = Field(description="Seconds from the dataset's reference time")
    TransactionAmt: float = Field(gt=0)
    ProductCD: str | None = None
    card1: int
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None


class Reason(BaseModel):
    feature: str
    value: float | int | str | None
    contribution: float


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    risk_score: int = Field(description="0-1000, higher is riskier")
    is_alert: bool
    threshold: float
    reasons: list[Reason]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": scorer is not None}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    if scorer is None:
        raise HTTPException(503, "Model not loaded")
    return scorer.score(txn.model_dump())
