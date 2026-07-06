"""
Central configuration for the app.
All secrets/keys are read from environment variables (.env file locally,
real environment variables in production/deployment) — nothing is hardcoded.
"""
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


def optional_secret(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None

    normalized = value.strip()
    if normalized.lower() in {"none", "null", "changeme"}:
        return None
    if normalized.startswith("your_") and normalized.endswith("_here"):
        return None

    return normalized


class Settings:
    # ---- General ----
    APP_NAME: str = "Stock Direction Predictor API"
    ENV: str = os.getenv("ENV", "development")

    # ---- Model training / caching ----
    CACHE_DIR: str = os.getenv("CACHE_DIR", "data/cache")
    CACHE_TTL_HOURS: float = float(os.getenv("CACHE_TTL_HOURS", "24"))
    MIN_HISTORY_YEARS: int = int(os.getenv("MIN_HISTORY_YEARS", "8"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.15"))

    # ---- LLM provider keys (only the ones you set will be usable) ----
    OPENAI_API_KEY: str | None = optional_secret("OPENAI_API_KEY")
    GOOGLE_API_KEY: str | None = optional_secret("GOOGLE_API_KEY")

    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # ---- CORS (so your own frontend, on a different origin, can call this) ----
    ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")


@lru_cache
def get_settings() -> Settings:
    return Settings()
