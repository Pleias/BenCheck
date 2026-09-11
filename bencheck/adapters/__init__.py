"""Dataset adapters bundled with bencheck."""

from .base import BaseDatasetAdapter
from .base_mcq import BaseMCQAdapter
from .dummy import DummyAdapter
from .hellaswag import HellaSwagAdapter
from .json import JsonAdapter
from .mcq_universal import UniversalMCQAdapter
from .specialized import (
    ARC_Adapter,
    BigBenchAdapter,
    CommonsenseQAAdapter,
    GlobalPIQAAdapter,
    MMLUAdapter,
    PIQAAdapter,
    TruthfulQAAdapter,
    WinograndeAdapter,
)

__all__ = [
    # Base classes
    "BaseDatasetAdapter",
    "BaseMCQAdapter",
    # Simple adapters
    "DummyAdapter",
    "JsonAdapter",
    # Universal MCQ adapter
    "UniversalMCQAdapter",
    # Specialized MCQ adapters
    "HellaSwagAdapter",
    "MMLUAdapter",
    "TruthfulQAAdapter",
    "BigBenchAdapter",
    "PIQAAdapter",
    "GlobalPIQAAdapter",
    "ARC_Adapter",
    "WinograndeAdapter",
    "CommonsenseQAAdapter",
]
