"""Universal adapter for multiple-choice question datasets from HuggingFace.

Supports automatic format detection for popular benchmarks like HellaSwag,
MMLU, TruthfulQA, HellaSwag, and more.

Based on analysis from feat/length-enum-bias:exp1.ipynb
"""

from __future__ import annotations

import string
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..types import BencheckQuestion, QuestionType
from .base import BaseDatasetAdapter

# Option key candidates (in priority order)
OPTION_KEYS = (
    "multiple_choice_targets",
    "mc1_targets",
    "mc2_targets",
    "endings",
    "options",
    "choices",
    "candidates",
    "answer_choices",
)

# Label key candidates (in priority order)
LABEL_KEYS = (
    "mc1_targets",
    "mc2_targets",
    "mc1_targets_scores",
    "mc2_targets_scores",
    "answer_index",
    "answer_idx",
    "target_index",
    "label",
    "gold",
    "answer",
    "correct",
    "correct_answer",
    "target",
    "targets",
    "mc1_labels",
)

# Context key candidates (in priority order)
CONTEXT_KEYS = (
    "inputs",
    "context",
    "ctx",
    "question",
    "prompt",
    "passage",
    "description",
)

# ID key candidates
ID_KEYS = ("id", "uid", "ind", "idx", "question_id")

# Letter to index mapping (A=0, B=1, C=2, D=3, ...)
LETTER_TO_INDEX = {c: i for i, c in enumerate(string.ascii_uppercase)}

# Known dataset-specific defaults (from exp1.ipynb)
DEFAULT_KEYS_BY_DATASET = {
    "Rowan/hellaswag": {
        "option_key": "endings",
        "label_key": "label",
        "context_key": "ctx",  # HellaSwag uses "ctx", not "context"
    },
    "cais/mmlu": {
        "option_key": "choices",
        "label_key": "answer",
        "context_key": "question",
    },
    "truthful_qa": {
        "option_key": "mc1_targets",
        "label_key": "mc1_targets",
        "context_key": "question",
    },
    "domenicrosati/TruthfulQA": {
        "option_key": "mc1_targets",
        "label_key": "mc1_labels",
        "context_key": "question",
    },
}


class UniversalMCQAdapter(BaseDatasetAdapter):
    """
    Universal adapter for multiple-choice question datasets.

    Automatically detects field names for options, labels, and context
    based on common patterns. Supports explicit override via parameters.

    Supported datasets (auto-configured):
    - HellaSwag
    - MMLU
    - TruthfulQA
    - Many others via automatic field detection

    Example:
        >>> adapter = UniversalMCQAdapter(dataset_name="Rowan/hellaswag", split="validation")
        >>> questions = adapter.load()  # Auto-loads from HuggingFace
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        split: str = "validation",
        config_name: Optional[str] = None,
        option_key: Optional[str] = None,
        label_key: Optional[str] = None,
        context_key: Optional[str] = None,
        id_key: Optional[str] = None,
    ):
        """
        Initialize the universal MCQ adapter.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "Rowan/hellaswag")
            split: Dataset split to load (default: "validation")
            config_name: Optional config/subset name (e.g., for BigBench tasks)
            option_key: Override field name for answer options (auto-detected if None)
            label_key: Override field name for correct label (auto-detected if None)
            context_key: Override field name for question context (auto-detected if None)
            id_key: Override field name for question ID (auto-detected if None)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.config_name = config_name
        self.option_key = option_key
        self.label_key = label_key
        self.context_key = context_key
        self.id_key = id_key

    def load(self, source: Any | None = None) -> List[BencheckQuestion]:
        """
        Load MCQ questions from HuggingFace or provided source.

        Args:
            source: If provided, overrides HuggingFace loading:
                - list/iterable: Use as records directly
                - dict with "records" key: Extract records
                - None: Load from HuggingFace using dataset_name

        Returns:
            List of BencheckQuestion objects
        """
        # Load records
        if source is None:
            # Load from HuggingFace
            records = self._load_from_huggingface()
        elif isinstance(source, dict) and "records" in source:
            records = source["records"]
        elif hasattr(source, "__iter__"):
            records = list(source)
        else:
            raise ValueError(
                f"Invalid source type: {type(source)}. Expected list, dict with 'records', or None"
            )

        if not records:
            return []

        # Infer keys from first record if not explicitly provided
        first_record = records[0]
        option_key, label_key, context_key, id_key = self._infer_keys(first_record)

        # Convert records to BencheckQuestion
        questions = []
        for idx, record in enumerate(records):
            try:
                question = self._convert_record(
                    record, idx, option_key, label_key, context_key, id_key
                )
                if question:
                    questions.append(question)
            except Exception:
                # Skip malformed records
                continue

        return questions

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace datasets library."""
        if self.dataset_name is None:
            raise ValueError("dataset_name must be provided to load from HuggingFace")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for HuggingFace loading. "
                "Install with: pip install datasets"
            )

        if self.config_name:
            dataset = load_dataset(self.dataset_name, self.config_name, split=self.split)
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)

        return list(dataset)

    def _infer_keys(
        self, record: Mapping[str, Any]
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Infer field names for options, label, context, and ID.

        Uses dataset-specific defaults if available, otherwise tries common patterns.
        """
        option_key = self.option_key
        label_key = self.label_key
        context_key = self.context_key
        id_key = self.id_key

        # Apply dataset-specific defaults
        if self.dataset_name and self.dataset_name in DEFAULT_KEYS_BY_DATASET:
            defaults = DEFAULT_KEYS_BY_DATASET[self.dataset_name]
            option_key = option_key or defaults.get("option_key")
            label_key = label_key or defaults.get("label_key")
            context_key = context_key or defaults.get("context_key")

        # Auto-detect option key
        if option_key is None:
            for key in OPTION_KEYS:
                if key in record:
                    option_key = key
                    break

        if option_key is None:
            raise ValueError(f"Could not infer option key. Available keys: {list(record.keys())}")

        # Auto-detect label key
        if label_key is None:
            for key in LABEL_KEYS:
                if key in record:
                    label_key = key
                    break

        # Auto-detect context key
        if context_key is None:
            for key in CONTEXT_KEYS:
                if key in record:
                    context_key = key
                    break

        # Auto-detect ID key
        if id_key is None:
            for key in ID_KEYS:
                if key in record:
                    id_key = key
                    break

        return option_key, label_key, context_key, id_key

    def _convert_record(
        self,
        record: Mapping[str, Any],
        idx: int,
        option_key: str,
        label_key: Optional[str],
        context_key: Optional[str],
        id_key: Optional[str],
    ) -> Optional[BencheckQuestion]:
        """Convert a single record to BencheckQuestion."""
        # Extract options
        options = self._extract_options(record.get(option_key))
        if not options:
            return None

        # Extract correct label
        if label_key:
            correct = self._extract_label(record, label_key, options)
        else:
            # No label available - use 0 as placeholder
            correct = 0

        # Extract context
        if context_key:
            context = str(record.get(context_key, ""))
        else:
            # Fallback: try to build from ctx/ctx_a+ctx_b (HellaSwag style)
            if "ctx" in record:
                context = record["ctx"]
            elif "ctx_a" in record:
                ctx_a = record.get("ctx_a", "")
                ctx_b = record.get("ctx_b", "")
                context = f"{ctx_a} {ctx_b}".strip()
            else:
                context = ""

        # Extract ID
        if id_key and id_key in record:
            question_id = str(record[id_key])
        else:
            question_id = str(idx)

        # Build metadata
        metadata = {"dataset": self.dataset_name or "unknown", "split": self.split}
        if self.config_name:
            metadata["config"] = self.config_name

        # Add source_id if available (for HellaSwag ActivityNet/WikiHow distinction)
        if "source_id" in record:
            metadata["source_id"] = record["source_id"]

        return BencheckQuestion(
            id=question_id,
            question=context,
            choices=options,
            correct=correct,
            question_type=QuestionType.SINGLE_CHOICE,
            metadata=metadata,
        )

    @staticmethod
    def _extract_options(raw_options: Any) -> List[str]:
        """Extract options from various formats."""
        if raw_options is None:
            return []

        # Dict with "choices" key (e.g., TruthfulQA mc1_targets)
        if isinstance(raw_options, Mapping):
            if "choices" in raw_options:
                return [str(o) for o in raw_options["choices"]]
            # Fallback: use all values
            return [str(v) for v in raw_options.values() if v]

        # List/sequence
        if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
            return [str(o) for o in raw_options]

        # Single value (shouldn't happen, but handle gracefully)
        return [str(raw_options)]

    @staticmethod
    def _extract_label(record: Mapping[str, Any], label_key: str, options: List[str]) -> int:
        """
        Extract correct label index from various formats.

        Handles:
        - Integer indices
        - String indices (convertible to int)
        - Letter indices (A, B, C, D)
        - Dict with "labels" array (TruthfulQA)
        - String matching against options
        """
        label_val = record.get(label_key)

        if label_val is None:
            return 0

        # Dict with "labels" array (e.g., TruthfulQA mc1_targets)
        if isinstance(label_val, Mapping) and "labels" in label_val:
            labels_seq = label_val.get("labels", [])
            if labels_seq:
                # Find index of first 1 (or max value)
                import numpy as np

                return int(np.argmax(labels_seq))

        # Integer or convertible to integer
        if isinstance(label_val, int):
            return label_val
        if isinstance(label_val, (float, bool)):
            return int(label_val)

        # String processing
        if isinstance(label_val, str):
            # Try direct match against options
            if label_val in options:
                return options.index(label_val)

            # Try letter index (A=0, B=1, etc.)
            if label_val.upper() in LETTER_TO_INDEX:
                return LETTER_TO_INDEX[label_val.upper()]

            # Try converting to integer
            try:
                return int(label_val)
            except ValueError:
                pass

        # List of labels (mc1_labels style - list of 0s and 1s)
        if isinstance(label_val, Sequence) and not isinstance(label_val, (str, bytes)):
            for j, v in enumerate(label_val):
                if int(v) == 1:
                    return j

        # Default fallback
        return 0
