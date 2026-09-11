"""Base classes for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMRequest:
    """Normalized request payload for a language model completion."""

    prompt: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None


@dataclass
class LLMResponse:
    """Normalized response returned by a language model."""

    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract provider exposing completion semantics."""

    name: str

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for ``request`` and return an :class:`LLMResponse`."""
