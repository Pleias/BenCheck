"""Scoring metrics placeholders."""

from __future__ import annotations

from typing import Iterable


def compute_accuracy(flags: Iterable[bool]) -> float:
    """Compute the accuracy (share of ``True`` values) for ``flags``."""

    values = list(flags)
    if not values:
        return 0.0
    return round((sum(1 for flag in values if flag) / len(values)) * 100.0, 2)
