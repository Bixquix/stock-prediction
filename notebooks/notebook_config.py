"""
Single source of truth for which ticker the notebooks analyze.

Change TICKER here, then re-run the notebooks in order:
  1. data_collection.ipynb
  2. feature_engineering.ipynb
  3. model_training.ipynb
  4. rolling_backtest.ipynb   (optional)

Every notebook imports this file, so you only ever change the ticker
in ONE place.
"""
import re

TICKER = "RELIANCE.NS"   # <-- CHANGE THIS to analyze any other stock/index/crypto
                          #     Indian NSE: "TCS.NS" | Indian BSE: "500325.BO"
                          #     US stocks:  "AAPL"   | Indices: "^NSEI", "^GSPC"
                          #     Crypto:     "BTC-USD"

YEARS_OF_HISTORY = 8       # how much daily history to pull
CORR_THRESHOLD = 0.10      # minimum |correlation| for a candidate asset to be kept
CORR_ROLLING_WINDOW = 60   # rolling window (trading days) for cross-asset correlation feature
TEST_SIZE = 0.15           # fraction of data held out, chronologically, for testing


def safe_name(ticker: str) -> str:
    """Filesystem-safe version of a ticker, used for file/folder naming
    (e.g. 'RELIANCE.NS' -> 'RELIANCE_NS', '^NSEI' -> 'NSEI')."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
