# 📈 Stock Direction Predictor — ML Ensemble + Multi-LLM Insight API

Predicts next-trading-day direction (UP/DOWN) for **any ticker Yahoo Finance
covers** — Indian stocks (NSE/BSE), US stocks, indices, crypto — using an
8-model machine learning ensemble trained fresh on that ticker's own price
history, plus optional plain-language commentary from your choice of LLM
(OpenAI, Anthropic, or Gemini).

> ⚠️ Educational project. Not financial advice.

## What changed from the original version

The original project trained 8 models once, offline, on NIFTY50 + a fixed
set of macro tickers (Gold, Crude, USD/INR, etc.), then served predictions
from static `.pkl` files via a Streamlit app. That Streamlit app has been
**removed** — it only worked against those old NIFTY50-only pickles and a
static `web_feed.csv`, neither of which exist anymore now that models are
trained per-ticker on demand. The frontend is now a standalone `frontend/`
app talking to the FastAPI backend below. This version:

- Generalizes to **any ticker** — features are built purely from the
  stock's own price/volume action (RSI, MACD, Bollinger Bands, ADX, OBV,
  moving averages, volatility, returns) instead of index-specific macro
  correlations, so the same pipeline works identically for `RELIANCE.NS`,
  `AAPL`, `^NSEI`, or `BTC-USD`.
- **The research notebooks themselves are now generalized too** (see
  below) — they used to be hardcoded to NIFTY50 + a fixed set of macro
  tickers. Now they detect which market a ticker belongs to, fetch a
  plausible set of candidate macro/market assets for that market, measure
  *actual* correlation with the target, and keep only what's genuinely
  correlated — instead of assuming Gold/Crude/VIX matter just because they
  mattered for Nifty.
- **Trains on demand** per ticker (cached for `CACHE_TTL_HOURS`, default
  24h) instead of relying on pre-baked pickles.
- Exposes everything through a **FastAPI backend** (JSON API + auto docs
  at `/docs`) so you can build any frontend you want on top of it.
- Adds a **multi-provider LLM layer** — an abstract `LLMProvider`
  interface with OpenAI/Anthropic/Gemini implementations, selectable per
  request (`?provider=anthropic`) or via a configured default.

## Notebooks — generalized research pipeline

The four notebooks in `notebooks/` now work for **any ticker**, not just
NIFTY50. They share one config file so you only change the ticker once:

```python
# notebooks/notebook_config.py
TICKER = "RELIANCE.NS"   # <-- change this, then re-run the notebooks in order
```

Run them in order:

1. **`data_collection.ipynb`** — detects which market the ticker belongs to
   (`market_universe.py`), downloads the target's OHLCV plus a plausible
   set of candidate macro/market assets for that market (e.g. Nifty/Sensex/
   India VIX/USD-INR for Indian stocks; S&P500/Nasdaq/VIX/DXY for US
   stocks), aligns everything to the target's trading calendar, and saves
   raw CSVs to `data/raw/`.
2. **`feature_engineering.ipynb`** — measures **real correlation** between
   each candidate's returns and the target's returns, keeps only the ones
   that clear `CORR_THRESHOLD` (falls back to the top 3 if nothing clears
   it), builds cross-asset features from those, adds technical indicator
   features from the target's own price/volume, defines the next-day
   direction target, and saves the feature matrix to `data/processed/`.
3. **`model_training.ipynb`** — chronological train/test split (never
   shuffled — this is time series data), trains the same 8-model ensemble
   as the original project, and saves the trained bundle to
   `models/{TICKER}/model_bundle.pkl`.
4. **`rolling_backtest.ipynb`** — simulates the ensemble's majority-vote
   strategy over the held-out test period vs. buy-and-hold, with a
   cumulative-return chart and summary stats (final return, max drawdown,
   hit rate).

The live API (`backend/`) uses its own equivalent, request-driven version
of this same pipeline (`backend/core/`) so a user hitting `/predict/AAPL`
doesn't have to run notebooks by hand — the notebooks are for research,
experimentation, and demonstrating the methodology in an interview.


## Project structure

```
backend/
├── main.py                  # FastAPI app + routes registration
├── config.py                 # env-driven settings
├── api/
│   ├── routes_predict.py     # GET /predict/{ticker}
│   ├── routes_llm.py         # GET /insight/{ticker}
│   └── schemas.py
└── core/
    ├── data_fetch.py         # any-ticker OHLCV via yfinance
    ├── feature_engineering.py# generic technical-indicator features
    ├── model_pipeline.py     # trains the 8-model ensemble
    ├── cache.py               # per-ticker trained-model cache (TTL)
    └── llm/
        ├── base.py            # LLMProvider interface + prompt builder
        ├── openai_provider.py
        ├── anthropic_provider.py
        ├── gemini_provider.py
        └── router.py          # picks provider at runtime
notebooks/                    # generalized research pipeline (any ticker)
├── notebook_config.py         # <-- change TICKER here, only place you need to
├── market_universe.py         # picks candidate macro assets per market
├── data_collection.ipynb
├── feature_engineering.ipynb
├── model_training.ipynb
└── rolling_backtest.ipynb
frontend/                     # your custom frontend, talks to the FastAPI backend
└── app.js
data/cache/                   # trained model bundles land here (gitignored)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add at least ONE LLM API key (OpenAI, Anthropic, or Google)
```

## Run locally

```bash
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive API docs.

### Example calls

```bash
# Next-day direction prediction for any ticker
curl http://127.0.0.1:8000/predict/RELIANCE.NS
curl http://127.0.0.1:8000/predict/AAPL
curl http://127.0.0.1:8000/predict/^NSEI

# Force a fresh retrain instead of using the cache
curl http://127.0.0.1:8000/predict/AAPL?force_retrain=true

# LLM-generated buy/sell/hold-style commentary
curl http://127.0.0.1:8000/insight/AAPL?provider=anthropic
curl http://127.0.0.1:8000/insight/AAPL?provider=openai
curl http://127.0.0.1:8000/insight/AAPL              # uses default/whatever key is set
```

**Ticker format cheat sheet:** NSE stocks need `.NS` (`TCS.NS`), BSE needs
`.BO`, US stocks are plain (`MSFT`), indices use `^` (`^NSEI`, `^GSPC`),
crypto is `BTC-USD` style.

## Deployment

**Docker (recommended — works on Render, Railway, Fly.io, AWS ECS, etc.):**

```bash
docker build -t stock-predictor .
docker run -p 8000:8000 --env-file .env stock-predictor
```

**Bare metal / VM:**

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Set `ALLOWED_ORIGINS` in `.env` to your actual frontend domain before
deploying (it defaults to `*` for local dev).

## How the ML pipeline works

1. Fetch ~8 years of daily OHLCV for the ticker.
2. Engineer ~20 technical-indicator features (momentum, trend, volatility,
   volume, candle shape) — no lookahead: every feature is computable using
   only data up to and including "today."
3. Chronological train/test split (never random shuffle — this is time
   series, shuffling would leak future data into training).
4. Train 8 classifiers (Logistic Regression, SVM, Random Forest, Extra
   Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM) and record each
   one's holdout accuracy.
5. Predict on the latest row; report per-model vote + probability plus an
   ensemble majority vote and average UP probability.
6. Optionally, hand that structured result to an LLM for a natural-language
   read — the LLM reasons over the numbers you give it, it isn't asked to
   pull its own market data.

## Disclaimer

Model accuracies for next-day direction prediction on liquid markets are
realistically in the 50–58% range — this is a portfolio/learning project
demonstrating an ML + LLM engineering pipeline, not a trading system.
