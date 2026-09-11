"""Benchmark distance metric placeholder."""

from __future__ import annotations

from typing import Iterable


def compute_distance(values: Iterable[float]) -> float:
    """Return the spread of ``values`` as a naive distance proxy."""

    collected = list(values)
    if not collected:
        return 0.0
    return max(collected) - min(collected)
