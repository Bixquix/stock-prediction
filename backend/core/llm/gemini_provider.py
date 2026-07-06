from __future__ import annotations

from backend.config import get_settings
from backend.core.llm.base import LLMProvider

settings = get_settings()


class GeminiProvider(LLMProvider):
    name = "gemini"

    def is_configured(self) -> bool:
        return bool(settings.GOOGLE_API_KEY)

    def generate_insight(self, prompt: str) -> str:
        import google.generativeai as genai  # imported lazily so the package is optional

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
