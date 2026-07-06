"""
Fetches historical OHLCV data for any ticker Yahoo Finance supports.

Ticker format notes:
    - Indian NSE stocks -> "RELIANCE.NS", "TCS.NS", "INFY.NS"
    - Indian BSE stocks -> "500325.BO"
    - US stocks -> "AAPL", "MSFT", "TSLA"
    - Indices -> "^NSEI" (Nifty 50), "^GSPC" (S&P 500), "^DJI"
    - Crypto -> "BTC-USD"
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf

from backend.config import get_settings

settings = get_settings()


class TickerNotFoundError(Exception):
    pass


def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise TickerNotFoundError(
            f"No data returned for ticker '{ticker}'. "
            f"Check the symbol - Indian stocks usually need a '.NS' "
            f"(NSE) or '.BO' (BSE) suffix, e.g. 'RELIANCE.NS'."
        )

    # yfinance sometimes returns MultiIndex columns for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_ohlcv(ticker: str, years: int | None = None) -> pd.DataFrame:
    """
    Download daily OHLCV history for a ticker.

    Returns a DataFrame indexed by Date with columns:
    Open, High, Low, Close, Volume
    """
    years = years or settings.MIN_HISTORY_YEARS
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,
    )
    df = _normalize_ohlcv(df, ticker)

    if len(df) < 300:
        raise TickerNotFoundError(
            f"Only {len(df)} rows of history found for '{ticker}'. "
            f"Need at least ~300 trading days to train a reliable model."
        )

    return df


def fetch_recent_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download recent daily OHLCV history for charts and quick UI context."""
    df = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=True,
    )
    df = _normalize_ohlcv(df, ticker)

    if len(df) < 2:
        raise TickerNotFoundError(
            f"Only {len(df)} rows of recent history found for '{ticker}'."
        )

    return df


def fetch_latest_price(ticker: str) -> dict:
    """Lightweight fetch used for quick current-price display."""
    df = fetch_recent_ohlcv(ticker, period="5d")
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    change_pct = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100
    return {
        "ticker": ticker,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": round(float(last["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "volume": int(last["Volume"]),
    }
