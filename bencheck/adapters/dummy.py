"""Minimal adapter returning hard-coded questions."""

from __future__ import annotations

from typing import Any, List

from ..types import BencheckQuestion
from .base import BaseDatasetAdapter


class DummyAdapter(BaseDatasetAdapter):
    """Returns a fixed set of hard-coded questions for testing."""

    def load(self, source: Any | None = None) -> List[BencheckQuestion]:
        """Return two hard-coded questions."""
        return [
            BencheckQuestion(
                id="q1",
                question="What is the capital of France?",
                choices=["London", "Paris", "Berlin", "Madrid"],
                correct=[1],
                metadata={},
            ),
            BencheckQuestion(
                id="q2",
                question="Which planet is closest to the Sun?",
                choices=["Mercury", "Venus", "Earth", "Mars"],
                correct=[0],
                metadata={"cheatable": True},
            ),
        ]
