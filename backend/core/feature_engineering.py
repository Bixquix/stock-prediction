"""
Turns raw OHLCV into a feature matrix for ANY ticker.

Your original project's features were NIFTY50-specific (merged with Gold,
Crude, USD/INR, etc. from a fixed set of Yahoo tickers). That doesn't
generalize — a random NASDAQ stock has no meaningful relationship to
USD/INR. So this version builds features purely from the stock's own
price/volume action using standard technical indicators via `ta`, which
works identically for any ticker.

Target: 1 if next trading day's Close > today's Close, else 0.
"""
from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DataFrame with columns Open, High, Low, Close, Volume (indexed by Date)
    returns: DataFrame of engineered features + 'target' column, NaNs dropped
    """
    out = df.copy()

    close, high, low, volume = out["Close"], out["High"], out["Low"], out["Volume"]

    # --- Returns & lagged returns (momentum) ---
    out["return_1d"] = close.pct_change(1)
    out["return_3d"] = close.pct_change(3)
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)

    # --- Moving averages (trend, as ratio to price so it's scale-free) ---
    out["sma_10_ratio"] = SMAIndicator(close, window=10).sma_indicator() / close
    out["sma_20_ratio"] = SMAIndicator(close, window=20).sma_indicator() / close
    out["sma_50_ratio"] = SMAIndicator(close, window=50).sma_indicator() / close
    out["ema_10_ratio"] = EMAIndicator(close, window=10).ema_indicator() / close
    out["ema_20_ratio"] = EMAIndicator(close, window=20).ema_indicator() / close

    # --- Momentum oscillators ---
    out["rsi_14"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()
    stoch = StochasticOscillator(high, low, close)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    # --- Trend strength ---
    out["adx_14"] = ADXIndicator(high, low, close, window=14).adx()

    # --- Volatility ---
    bb = BollingerBands(close, window=20)
    out["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    out["bb_pct"] = bb.bollinger_pband()
    out["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range() / close
    out["volatility_10d"] = close.pct_change().rolling(10).std()

    # --- Volume ---
    out["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    out["obv_change"] = out["obv"].pct_change(5)
    out["volume_change_5d"] = volume.pct_change(5)
    out["volume_zscore_20d"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

    # --- Candle shape (today's own action) ---
    out["high_low_range"] = (high - low) / close
    out["close_open_ratio"] = (close - out["Open"]) / out["Open"]

    # --- Target: next day direction ---
    out["target"] = (close.shift(-1) > close).astype(int)

    feature_cols = [c for c in out.columns if c not in
                    ["Open", "High", "Low", "Close", "Volume", "target"]]

    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna()

    return out, feature_cols
