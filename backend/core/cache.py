"""
Simple disk cache so that repeated requests for the same ticker within
CACHE_TTL_HOURS reuse the already-trained models instead of retraining
(training 8 models can take anywhere from a few seconds to ~30s depending
on ticker history length — not something you want on every page refresh).
"""
from __future__ import annotations

import datetime as dt
import os
import re

import joblib

from backend.config import get_settings

settings = get_settings()


def _safe_filename(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", ticker) + ".joblib"


def _cache_path(ticker: str) -> str:
    os.makedirs(settings.CACHE_DIR, exist_ok=True)
    return os.path.join(settings.CACHE_DIR, _safe_filename(ticker))


def load_cached_bundle(ticker: str) -> dict | None:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None

    bundle = joblib.load(path)
    trained_at = bundle.get("trained_at")
    if trained_at is None:
        return None

    age_hours = (dt.datetime.utcnow() - trained_at).total_seconds() / 3600
    if age_hours > settings.CACHE_TTL_HOURS:
        return None

    return bundle


def save_bundle(ticker: str, bundle: dict) -> None:
    bundle["trained_at"] = dt.datetime.utcnow()
    joblib.dump(bundle, _cache_path(ticker))
