"""JSON-based benchmark adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from ..types import BencheckQuestion
from .base import BaseDatasetAdapter


class JsonAdapter(BaseDatasetAdapter):
    """Loads benchmark questions from JSON files or payloads."""

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, source: Any | None) -> List[BencheckQuestion]:
        if source is None:
            raise ValueError("JsonAdapter requires a source path or payload")

        payload: Iterable[Any]
        if isinstance(source, (str, Path)):
            path = Path(source)
            payload = json.loads(path.read_text(encoding=self.encoding))
        else:
            payload = source

        if isinstance(payload, dict):
            payload = payload.get("questions", [])

        if not isinstance(payload, Iterable):
            raise ValueError("Invalid payload for JsonAdapter")

        return self.coerce_questions(payload)  # type: ignore[arg-type]
