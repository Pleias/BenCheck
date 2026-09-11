"""Configuration helpers for LLM providers."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMSettings:
    """Collection of configuration parameters for LLM access."""

    provider: str = "gemini"
    model: str = "gemini-1.5-flash"
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> LLMSettings:
        """Create settings using environment variables."""

        return cls(
            provider=os.getenv("BENCHECK_LLM_PROVIDER", "gemini"),
            model=os.getenv("BENCHECK_LLM_MODEL", "gemini-1.5-flash"),
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    def create_provider(self):
        """Instantiate the configured provider using stored settings."""

        from .factory import create_provider

        return create_provider(self.provider, api_key=self.api_key, model=self.model)
