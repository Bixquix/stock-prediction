from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.core import cache
from backend.core.data_fetch import TickerNotFoundError, fetch_ohlcv
from backend.core.feature_engineering import build_features
from backend.core.model_pipeline import predict_latest, train_all_models

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.get("/{ticker}")
def predict(ticker: str, force_retrain: bool = Query(False)):
    """
    Predict next-trading-day direction for any ticker Yahoo Finance supports.
    e.g. /predict/RELIANCE.NS  /predict/AAPL  /predict/^NSEI  /predict/BTC-USD

    On first request for a ticker (or after the cache TTL expires) this
    trains all 8 models fresh on that ticker's own price history, which
    takes a few seconds. Subsequent requests within the TTL window reuse
    the cached models.
    """
    ticker = ticker.upper()
    trained_fresh = False

    bundle = None if force_retrain else cache.load_cached_bundle(ticker)

    try:
        raw = fetch_ohlcv(ticker)
    except TickerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    features_df, feature_cols = build_features(raw)

    if features_df.empty or len(features_df) < 200:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough clean data to train on '{ticker}' after feature engineering.",
        )

    if bundle is None:
        bundle = train_all_models(features_df, feature_cols)
        cache.save_bundle(ticker, bundle)
        trained_fresh = True

    prediction = predict_latest(bundle, features_df)

    latest_row = raw.iloc[-1]
    prev_row = raw.iloc[-2]
    change_pct = float((latest_row["Close"] - prev_row["Close"]) / prev_row["Close"] * 100)

    return {
        "ticker": ticker,
        "as_of_date": raw.index[-1].strftime("%Y-%m-%d"),
        "current_price": round(float(latest_row["Close"]), 2),
        "change_pct": round(change_pct, 2),
        "trained_fresh": trained_fresh,
        **prediction,
    }
