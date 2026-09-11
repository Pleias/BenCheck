"""Gemini provider stub used for local development."""

from __future__ import annotations

from textwrap import shorten

from ..base import LLMProvider, LLMRequest, LLMResponse


class GeminiProvider(LLMProvider):
    """Basic Gemini provider that currently returns deterministic stub responses."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash") -> None:
        super().__init__(api_key=api_key, model=model)

    def complete(self, request: LLMRequest) -> LLMResponse:
        preview = shorten(request.prompt.replace("\n", " ").strip(), width=80, placeholder="...")
        label = self.model or self.name
        text = f"[{label}] {preview}"
        payload = {
            "provider": self.name,
            "model": self.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "system_prompt": request.system_prompt,
        }
        return LLMResponse(text=text, raw=payload)
