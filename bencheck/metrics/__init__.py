"""Metrics helpers for bencheck."""

from .benchmark_distance import compute_distance
from .scoring import compute_accuracy

__all__ = ["compute_accuracy", "compute_distance"]
