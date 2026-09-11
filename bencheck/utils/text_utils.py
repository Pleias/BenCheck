"""Text helpers for bencheck."""

from __future__ import annotations

from typing import List


def tokenize(text: str) -> List[str]:
    """Return a whitespace tokenization of ``text``."""

    return text.split()
