"""BenCheck: Benchmark quality auditing toolkit."""

from importlib.metadata import PackageNotFoundError, version

from .base import BencheckCheck, BencheckModel, DatasetAdapter
from .core.runner import BencheckRunner
from .types import BencheckQuestion, BenchmarkEvaluation, CheckResult, QuestionType

try:
    __version__ = version("bencheck")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "unknown"

__all__ = [
    "BencheckModel",
    "BencheckCheck",
    "DatasetAdapter",
    "BencheckQuestion",
    "CheckResult",
    "BenchmarkEvaluation",
    "QuestionType",
    "BencheckRunner",
]
