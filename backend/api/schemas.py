from pydantic import BaseModel


class ModelPrediction(BaseModel):
    model: str
    prediction: str
    confidence: float
    up_probability: float
    backtest_accuracy: float


class PredictionResponse(BaseModel):
    ticker: str
    as_of_date: str
    current_price: float
    change_pct: float
    per_model: list[ModelPrediction]
    up_votes: int
    down_votes: int
    majority_direction: str
    majority_confidence: float
    avg_up_probability: float
    trained_fresh: bool


class InsightResponse(BaseModel):
    ticker: str
    provider_used: str
    insight: str
    disclaimer: str = "Educational project only. Not financial advice."


class ErrorResponse(BaseModel):
    detail: str
