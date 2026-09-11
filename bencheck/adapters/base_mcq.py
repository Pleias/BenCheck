"""Base class for multiple-choice question adapters.

Provides common functionality for loading MCQ datasets from HuggingFace.
Specialized adapters inherit from this and define their specific configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..types import BencheckQuestion, QuestionType
from ..utils import mcq_helpers
from .base import BaseDatasetAdapter


class BaseMCQAdapter(BaseDatasetAdapter):
    """
    Base class for MCQ adapters with common loading and conversion logic.

    Subclasses should define:
    - dataset_name: HuggingFace dataset identifier
    - option_key: Field name for answer options
    - label_key: Field name for correct label
    - context_key: Field name for question context
    - (optional) config_name: Dataset config/subset
    - (optional) extra_metadata_keys: Additional metadata to extract

    Example:
        class MMLUAdapter(BaseMCQAdapter):
            dataset_name = "cais/mmlu"
            option_key = "choices"
            label_key = "answer"
            context_key = "question"
    """

    # Dataset configuration (override in subclass)
    dataset_name: Optional[str] = None
    config_name: Optional[str] = None
    option_key: Optional[str] = None
    label_key: Optional[str] = None
    context_key: Optional[str] = None
    id_key: Optional[str] = None
    extra_metadata_keys: List[str] = []

    def __init__(
        self,
        split: str = "validation",
        dataset_name: Optional[str] = None,
        config_name: Optional[str] = None,
    ):
        """
        Initialize the adapter.

        Args:
            split: Dataset split to load (default: "validation")
            dataset_name: Override class-level dataset_name
            config_name: Override class-level config_name
        """
        self.split = split
        # Allow instance-level override of class attributes
        if dataset_name:
            self.dataset_name = dataset_name
        if config_name:
            self.config_name = config_name

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

        # Convert records to BencheckQuestion
        questions = []
        for idx, record in enumerate(records):
            try:
                question = self._convert_record(record, idx)
                if question:
                    questions.append(question)
            except Exception:
                # Skip malformed records
                continue

        return questions

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace datasets library."""
        if self.dataset_name is None:
            raise ValueError(f"{self.__class__.__name__} requires dataset_name to be set")

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

    def _convert_record(self, record: Mapping[str, Any], idx: int) -> Optional[BencheckQuestion]:
        """
        Convert a single record to BencheckQuestion.

        This method uses the class configuration (option_key, label_key, etc.)
        to extract fields. Can be overridden for custom conversion logic.
        """
        # Extract options
        option_key = self._get_option_key(record)
        if not option_key or option_key not in record:
            return None

        options = mcq_helpers.extract_options(record[option_key])
        if not options:
            return None

        # Extract correct label
        label_key = self._get_label_key(record)
        if label_key and label_key in record:
            # Get original label value to preserve MCQ format
            label_val = record[label_key]

            # Check for multiple correct answers in various formats:
            # 1. List with len>1 (e.g., [0, 2, 3])
            # 2. Dict with "labels" array containing multiple 1s (TruthfulQA mc2 format)
            is_mcq = False

            if isinstance(label_val, list) and len(label_val) > 1:
                is_mcq = True
            elif isinstance(label_val, dict) and "labels" in label_val:
                # Check if labels array has multiple 1s
                labels_array = label_val.get("labels", [])
                num_correct = sum(1 for x in labels_array if int(x) == 1)
                if num_correct > 1:
                    is_mcq = True

            if is_mcq:
                # Multiple correct answers - preserve as list
                correct = mcq_helpers.extract_label_indices(record, label_key, options)
            else:
                # Single correct answer - convert to index
                correct = mcq_helpers.extract_label_index(record, label_key, options)
        else:
            # No label available - use 0 as placeholder
            correct = 0

        # Extract context
        context_key = self._get_context_key(record)
        context = mcq_helpers.extract_context(record, context_key)

        # Extract ID
        id_key = self._get_id_key(record)
        question_id = mcq_helpers.extract_question_id(record, idx, id_key)

        # Build metadata
        metadata = mcq_helpers.build_metadata(
            record,
            dataset_name=self.dataset_name,
            split=self.split,
            config_name=self.config_name,
            extra_keys=self.extra_metadata_keys,
        )

        # Determine question type based on correct answer format
        if isinstance(correct, list) and len(correct) > 1:
            question_type = QuestionType.MULTIPLE_CHOICE
            metadata["question_type"] = "multiple_choice"
            metadata["num_correct_answers"] = len(correct)
        else:
            question_type = QuestionType.SINGLE_CHOICE
            metadata["question_type"] = "single_choice"
            metadata["num_correct_answers"] = 1

        return BencheckQuestion(
            id=question_id,
            question=context,
            choices=options,
            correct=correct,
            question_type=question_type,
            metadata=metadata,
        )

    def _get_option_key(self, record: Mapping[str, Any]) -> Optional[str]:
        """Get option key (class attribute or auto-detect)."""
        if self.option_key:
            return self.option_key
        return mcq_helpers.detect_field_key(record, mcq_helpers.OPTION_KEYS)

    def _get_label_key(self, record: Mapping[str, Any]) -> Optional[str]:
        """Get label key (class attribute or auto-detect)."""
        if self.label_key:
            return self.label_key
        return mcq_helpers.detect_field_key(record, mcq_helpers.LABEL_KEYS)

    def _get_context_key(self, record: Mapping[str, Any]) -> Optional[str]:
        """Get context key (class attribute or auto-detect)."""
        if self.context_key:
            return self.context_key
        return mcq_helpers.detect_field_key(record, mcq_helpers.CONTEXT_KEYS)

    def _get_id_key(self, record: Mapping[str, Any]) -> Optional[str]:
        """Get ID key (class attribute or auto-detect)."""
        if self.id_key:
            return self.id_key
        return mcq_helpers.detect_field_key(record, mcq_helpers.ID_KEYS)
