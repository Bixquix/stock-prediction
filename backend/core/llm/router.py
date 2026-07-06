"""
Picks which LLM provider actually handles a request. A frontend can ask
for a specific provider (?provider=openai or ?provider=gemini), or leave it
unset and get whichever is configured (checking DEFAULT_LLM_PROVIDER first,
then falling back to any provider that has an API key set).
"""
from __future__ import annotations

from backend.config import get_settings
from backend.core.llm.base import LLMProvider, build_market_prompt
from backend.core.llm.gemini_provider import GeminiProvider
from backend.core.llm.openai_provider import OpenAIProvider

settings = get_settings()

_PROVIDERS: dict[str, LLMProvider] = {
    "openai": OpenAIProvider(),
    "gemini": GeminiProvider(),
}


class NoLLMConfiguredError(Exception):
    pass


def get_provider(requested: str | None = None) -> LLMProvider:
    if requested:
        provider = _PROVIDERS.get(requested.lower())
        if provider is None:
            raise ValueError(
                f"Unknown provider '{requested}'. Choose from: {list(_PROVIDERS)}"
            )
        if not provider.is_configured():
            raise NoLLMConfiguredError(
                f"Provider '{requested}' has no API key set in the environment."
            )
        return provider

    # No specific provider requested: try the configured default, then
    # fall back to whichever provider actually has a key set.
    default = _PROVIDERS.get(settings.DEFAULT_LLM_PROVIDER.lower())
    if default and default.is_configured():
        return default

    for provider in _PROVIDERS.values():
        if provider.is_configured():
            return provider

    raise NoLLMConfiguredError(
        "No LLM provider is configured. Set OPENAI_API_KEY or GOOGLE_API_KEY "
        "in your .env file."
    )


def get_market_insight(context: dict, requested_provider: str | None = None) -> dict:
    provider = get_provider(requested_provider)
    prompt = build_market_prompt(context)
    text = provider.generate_insight(prompt)
    return {"provider_used": provider.name, "insight": text}
