"""HellaSwag benchmark adapter."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..types import BencheckQuestion
from .base_mcq import BaseMCQAdapter


class HellaSwagAdapter(BaseMCQAdapter):
    """
    Adapter for HellaSwag benchmark from HuggingFace.

    HellaSwag format:
        - ctx / ctx_a + ctx_b: Context/prompt for the question
        - endings: List of 4 possible continuations
        - label: Correct answer index (0-3, integer or string)
        - source_id: Identifies source (ActivityNet vs WikiHow)
        - ind: Original dataset index (has duplicates - 433 duplicates in validation)

    Note:
        HellaSwag validation has 10,042 questions but only 9,609 unique 'ind' values.
        This adapter uses row index as question_id and preserves original 'ind'
        in metadata as 'original_ind' for reference.

    Reference:
        HellaSwag: https://arxiv.org/abs/1905.07830
        "Can a Machine Really Finish Your Sentence?" (Zellers et al., 2019)

    Example:
        >>> # Load HellaSwag
        >>> adapter = HellaSwagAdapter(dataset_name="Rowan/hellaswag")
        >>> questions = adapter.load()
        >>>
        >>> # Filter to only ActivityNet questions
        >>> adapter = HellaSwagAdapter(filter_source="activitynet")
        >>> questions = adapter.load()
    """

    # Default HellaSwag configuration
    dataset_name = "Rowan/hellaswag"
    option_key = "endings"
    label_key = "label"
    context_key = "ctx"  # Will also handle ctx_a + ctx_b in _convert_record
    # NOTE: id_key removed - using row index instead due to duplicate 'ind' values
    extra_metadata_keys = ["source_id", "activity_label"]

    def __init__(
        self,
        dataset_name: str = "Rowan/hellaswag",
        split: str = "validation",
        filter_source: Optional[str] = None,
    ):
        """
        Initialize the HellaSwag adapter.

        Args:
            dataset_name: HuggingFace dataset identifier (default: "Rowan/hellaswag")
            split: Dataset split to load (e.g., "train", "validation", "test")
            filter_source: Optional filter for source_id (e.g., "activitynet", "wikihow")
        """
        super().__init__(split=split, dataset_name=dataset_name)
        self.filter_source = filter_source

    def _convert_record(self, record: Mapping[str, Any], idx: int) -> Optional[BencheckQuestion]:
        """
        Convert HellaSwag record to BencheckQuestion.

        Handles both "ctx" and "ctx_a + ctx_b" formats.
        Applies source filtering if configured.
        Uses row index as question_id (not 'ind') due to duplicate 'ind' values.
        Preserves original 'ind' field in metadata as 'original_ind'.
        """
        from ..utils import mcq_helpers

        # Apply source filter if specified
        if self.filter_source is not None:
            source_id = record.get("source_id", "").lower()
            if self.filter_source.lower() not in source_id:
                return None

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
            label_val = record[label_key]

            # Check for MCQ format
            is_mcq = False
            if isinstance(label_val, list) and len(label_val) > 1:
                is_mcq = True
            elif isinstance(label_val, dict) and "labels" in label_val:
                labels_array = label_val.get("labels", [])
                num_correct = sum(1 for x in labels_array if int(x) == 1)
                if num_correct > 1:
                    is_mcq = True

            if is_mcq:
                correct = mcq_helpers.extract_label_indices(record, label_key, options)
            else:
                correct = mcq_helpers.extract_label_index(record, label_key, options)
        else:
            correct = 0

        # Extract context
        context_key = self._get_context_key(record)
        context = mcq_helpers.extract_context(record, context_key)

        # Use row index as question_id (NOT 'ind' field due to duplicates)
        question_id = str(idx)

        # Build metadata
        metadata = mcq_helpers.build_metadata(
            record,
            dataset_name=self.dataset_name,
            split=self.split,
            config_name=self.config_name,
            extra_keys=self.extra_metadata_keys,
        )

        # Preserve original 'ind' field for reference
        if "ind" in record:
            metadata["original_ind"] = record["ind"]

        # Determine question type
        from ..types import QuestionType

        if isinstance(correct, list):
            question_type = QuestionType.MULTIPLE_CHOICE
        else:
            question_type = QuestionType.SINGLE_CHOICE

        # Create BencheckQuestion
        from ..types import BencheckQuestion

        return BencheckQuestion(
            id=question_id,
            question=context,
            choices=options,
            correct=correct,
            question_type=question_type,
            metadata=metadata,
        )
