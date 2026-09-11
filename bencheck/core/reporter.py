"""Result reporting helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from ..types import BenchmarkEvaluation


class BencheckReporter:
    """Responsible for formatting or saving bencheck results."""

    def to_json(self, results: BenchmarkEvaluation | Dict[str, Any], path: str | Path) -> None:
        """Persist ``results`` as JSON to ``path``."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(results)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def to_dict(self, results: BenchmarkEvaluation | Dict[str, Any]) -> Dict[str, Any]:
        """Return a dictionary representation of ``results``."""

        if is_dataclass(results):
            return asdict(results)
        return json.loads(json.dumps(results))
