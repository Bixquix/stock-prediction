from __future__ import annotations

from backend.config import get_settings
from backend.core.llm.base import LLMProvider

settings = get_settings()


class OpenAIProvider(LLMProvider):
    name = "openai"

    def is_configured(self) -> bool:
        return bool(settings.OPENAI_API_KEY)

    def generate_insight(self, prompt: str) -> str:
        from openai import OpenAI  # imported lazily so the package is optional

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
