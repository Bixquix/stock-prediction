"""
Abstract interface every LLM provider must implement. This is what makes
"support multiple, pick at runtime" possible without the rest of the app
caring which provider is actually behind the call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate_insight(self, prompt: str) -> str:
        """Send the prompt to the provider's chat/completion API and return text."""
        raise NotImplementedError

    @abstractmethod
    def is_configured(self) -> bool:
        """True if the required API key is present."""
        raise NotImplementedError


def build_market_prompt(context: dict) -> str:
    """
    Shared prompt builder so all providers get an identical, apples-to-apples
    prompt. Keep this factual/structured — the model is asked to reason over
    numbers we give it, not to invent data.
    """
    return f"""You are a markets analyst assistant embedded in a stock analysis
dashboard. Analyze the data below for {context['ticker']} and respond in
under 120 words with plain-language reasoning.

Latest close: {context['close']} ({context['change_pct']}% vs prior close)
RSI(14): {context['rsi']}
MACD histogram: {context['macd_diff']}
20-day Bollinger %B: {context['bb_pct']}

ML ensemble of 8 models predicts: {context['majority_direction']} \
({context['up_votes']} models say UP, {context['down_votes']} say DOWN, \
average UP probability {context['avg_up_probability']}%)

Give:
1. A one-line read on the technical setup (trend/momentum/volatility).
2. Whether this setup leans toward Buy / Sell / Hold bias for a short-term
   (next 1-2 trading days) trader — and say explicitly that this is a
   probabilistic technical read, not investment advice.
3. One key risk that could invalidate this read.

End with: "Not financial advice — for educational purposes only."
"""
