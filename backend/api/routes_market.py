from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.core.data_fetch import (
    TickerNotFoundError,
    fetch_latest_price,
    fetch_recent_ohlcv,
)

router = APIRouter(prefix="/market", tags=["market-data"])


@router.get("/{ticker}/quote")
def quote(ticker: str):
    """Return the latest close, daily change, and volume for a ticker."""
    ticker = ticker.upper()
    try:
        return fetch_latest_price(ticker)
    except TickerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{ticker}/history")
def history(
    ticker: str,
    days: int = Query(180, ge=30, le=1000),
):
    """Return recent daily closes and volume for frontend charting."""
    ticker = ticker.upper()
    try:
        raw = fetch_recent_ohlcv(ticker, period="5y")
    except TickerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    view = raw.tail(days)
    series = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        }
        for idx, row in view.iterrows()
    ]

    return {"ticker": ticker, "days": len(series), "series": series}
