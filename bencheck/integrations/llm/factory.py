"""LLM provider factory utilities."""

from __future__ import annotations

from typing import Dict, Type

from .base import LLMProvider
from .providers import GeminiProvider

_PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {
    "gemini": GeminiProvider,
}


def create_provider(name: str, **kwargs) -> LLMProvider:
    """Instantiate an LLM provider by ``name`` using the configured registry."""

    key = name.lower()
    if key not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown LLM provider: {name}")
    provider_cls = _PROVIDER_REGISTRY[key]
    return provider_cls(**kwargs)
