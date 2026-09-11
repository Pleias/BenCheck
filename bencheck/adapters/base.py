"""Base utilities for dataset adapters."""

from typing import Any, Iterable, List, Mapping, Union

from ..base import DatasetAdapter
from ..types import BencheckQuestion, QuestionType


class BaseDatasetAdapter(DatasetAdapter):
    """Helper base class providing common utilities for adapters."""

    def load(self, source: Any | None) -> List[BencheckQuestion]:
        raise NotImplementedError

    @staticmethod
    def normalize_correct(
        correct_raw: Any,
    ) -> Union[int, List[int], str, List[str]]:
        """
        Normalize the 'correct' field without changing indices.

        Adapters should use this helper to handle different input formats
        while preserving the original indexing convention (0-based or 1-based).

        Examples:
            normalize_correct(1) → 1
            normalize_correct([1]) → [1]
            normalize_correct("4") → "4"
            normalize_correct(["A", "C"]) → ["A", "C"]
        """
        if correct_raw is None:
            return []
        if isinstance(correct_raw, (int, str)):
            return correct_raw
        if isinstance(correct_raw, list):
            return correct_raw
        return correct_raw

    @staticmethod
    def coerce_questions(
        payload: Iterable[Mapping[str, Any]],
        question_type: QuestionType = QuestionType.SINGLE_CHOICE,
    ) -> List[BencheckQuestion]:
        """
        Convert an iterable of mappings to BencheckQuestion objects.

        Args:
            payload: Iterable of dicts with keys: id, question, choices, correct
            question_type: Type of questions (default: SINGLE_CHOICE)

        Returns:
            List of BencheckQuestion objects
        """
        questions = []
        for record in payload:
            correct = BaseDatasetAdapter.normalize_correct(record.get("correct"))

            questions.append(
                BencheckQuestion(
                    id=str(record["id"]),
                    question=record["question"],
                    choices=list(record.get("choices", [])),
                    correct=correct,
                    question_type=question_type,
                    metadata=dict(record.get("metadata", {})),
                )
            )
        return questions
