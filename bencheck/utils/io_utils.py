"""IO helpers."""

from __future__ import annotations

import os


def ensure_directory(path: str) -> None:
    """Ensure ``path`` exists."""

    os.makedirs(path, exist_ok=True)
