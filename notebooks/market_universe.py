"""
Decides which "macro/exogenous" candidate assets are worth checking for
correlation against a given ticker.

This replaces the original notebook's approach of hand-picking Gold, Crude,
INDIA VIX, USD/INR, etc. specifically for NIFTY50. Those were reasonable
choices for an Indian index, but meaningless for, say, a US tech stock.
Instead, we pick a plausible candidate universe based on what *kind* of
ticker is being analyzed, fetch all of them, and let the feature engineering
step keep only the ones that actually turn out to be correlated.
"""

IN_UNIVERSE = {
    "NIFTY_50": "^NSEI",
    "SENSEX": "^BSESN",
    "INDIA_VIX": "^INDIAVIX",
    "USD_INR": "INR=X",
    "GOLD": "GC=F",
    "CRUDE_OIL": "CL=F",
    "SP500": "^GSPC",
    "US10Y_YIELD": "^TNX",
    "DXY": "DX-Y.NYB",
}

US_UNIVERSE = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "VIX": "^VIX",
    "US10Y_YIELD": "^TNX",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
    "CRUDE_OIL": "CL=F",
}

CRYPTO_UNIVERSE = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SP500": "^GSPC",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
}

INDIAN_INDEX_TICKERS = {"^NSEI", "^BSESN", "^NSEBANK", "^CNXIT"}


def detect_market(ticker: str) -> str:
    """Rough classification used only to pick a sensible candidate universe."""
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO") or t in INDIAN_INDEX_TICKERS:
        return "IN"
    if t.endswith("-USD") or t.endswith("-USDT"):
        return "CRYPTO"
    return "US"


def get_candidate_universe(ticker: str) -> dict:
    """Returns {name: yahoo_symbol} candidates to test for correlation,
    excluding the ticker itself if it happens to appear in its own universe."""
    market = detect_market(ticker)
    universe = {"IN": IN_UNIVERSE, "US": US_UNIVERSE, "CRYPTO": CRYPTO_UNIVERSE}[market]
    return {k: v for k, v in universe.items() if v.upper() != ticker.upper()}
