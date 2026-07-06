from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.core import cache
from backend.core.data_fetch import TickerNotFoundError, fetch_ohlcv
from backend.core.feature_engineering import build_features
from backend.core.llm.router import NoLLMConfiguredError, get_market_insight
from backend.core.model_pipeline import predict_latest, train_all_models

router = APIRouter(prefix="/insight", tags=["llm-insight"])


@router.get("/{ticker}")
def insight(
    ticker: str,
    provider: str | None = Query(None, description="openai | gemini"),
):
    """
    Runs the same ML pipeline as /predict, then hands the ensemble result
    plus key indicators to an LLM for a plain-language read. Provider can
    be forced via ?provider=..., otherwise falls back to whichever is
    configured (see DEFAULT_LLM_PROVIDER in .env).
    """
    ticker = ticker.upper()

    try:
        raw = fetch_ohlcv(ticker)
    except TickerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    features_df, feature_cols = build_features(raw)

    bundle = cache.load_cached_bundle(ticker)
    if bundle is None:
        bundle = train_all_models(features_df, feature_cols)
        cache.save_bundle(ticker, bundle)

    prediction = predict_latest(bundle, features_df)
    latest_row = raw.iloc[-1]
    prev_row = raw.iloc[-2]
    change_pct = float((latest_row["Close"] - prev_row["Close"]) / prev_row["Close"] * 100)
    latest_features = features_df.iloc[-1]

    context = {
        "ticker": ticker,
        "close": round(float(latest_row["Close"]), 2),
        "change_pct": round(change_pct, 2),
        "rsi": round(float(latest_features["rsi_14"]), 2),
        "macd_diff": round(float(latest_features["macd_diff"]), 4),
        "bb_pct": round(float(latest_features["bb_pct"]), 2),
        "majority_direction": prediction["majority_direction"],
        "up_votes": prediction["up_votes"],
        "down_votes": prediction["down_votes"],
        "avg_up_probability": prediction["avg_up_probability"],
    }

    try:
        result = get_market_insight(context, requested_provider=provider)
    except NoLLMConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ticker": ticker,
        "disclaimer": "Educational project only. Not financial advice.",
        **result,
    }
