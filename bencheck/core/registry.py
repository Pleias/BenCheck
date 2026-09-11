"""Simple registries for bencheck components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BencheckRegistry:
    """Stores component factories by name."""

    _registry: Dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, factory: Any) -> None:
        """Register ``factory`` under ``name``."""
        self._registry[name] = factory

    def resolve(self, name: str) -> Any:
        """Retrieve a component factory by ``name``."""
        return self._registry[name]


model_registry = BencheckRegistry()
adapter_registry = BencheckRegistry()
check_registry = BencheckRegistry()
