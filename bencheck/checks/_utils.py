"""
Common utilities for diagnostic checks.

This module contains shared helper functions used across multiple checks
to avoid code duplication.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def normalize_correct_index(correct: Any) -> Optional[int]:
    """
    Extract a single correct index from various formats.

    IMPORTANT: This function ALWAYS returns a SINGLE index. For multiple-choice
    questions with multiple correct answers (e.g., [1, 3]), it returns ONLY THE
    FIRST correct index. This is by design for checks that operate on single-choice
    questions. Use with caution for true MCQ datasets.

    Handles multiple input formats used by different benchmarks:
    - int: Direct index (0-based or 1-based depending on dataset)
    - List[int]: Multiple correct answers (returns FIRST ONLY, logs warning)
    - str: String digit or letter (A/B/C/D)
    - List[str]: List of strings (returns FIRST ONLY, logs warning)

    Args:
        correct: The correct field from BencheckQuestion (flexible format)

    Returns:
        Single integer index, or None if invalid

    Examples:
        >>> normalize_correct_index(2)
        2
        >>> normalize_correct_index([1, 3])  # WARNING: Returns only first!
        1
        >>> normalize_correct_index("A")
        0
        >>> normalize_correct_index("2")
        2

    Warnings:
        - Logs warning when multiple correct answers are provided
        - Logs warning when correct index is invalid or cannot be parsed
    """
    if isinstance(correct, int):
        return correct
    elif isinstance(correct, list):
        if len(correct) == 0:
            logger.warning("normalize_correct_index: Empty list provided for correct answer")
            return None

        # IMPORTANT: For MCQ with multiple correct answers, we take ONLY THE FIRST
        if len(correct) > 1:
            logger.warning(
                f"normalize_correct_index: Multiple correct answers {correct} provided. "
                f"Returning ONLY FIRST index {correct[0]}. This may be incorrect for true MCQ datasets!"
            )

        first = correct[0]
        if isinstance(first, int):
            return first
        elif isinstance(first, str):
            # Try to convert string to int
            if first.isdigit():
                return int(first)
            # Try letter to index (A=0, B=1, etc.)
            letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
            result = letter_to_idx.get(first.upper())
            if result is None:
                logger.warning(f"normalize_correct_index: Could not map letter '{first}' to index")
            return result
        return None
    elif isinstance(correct, str):
        # Try direct digit conversion
        if correct.isdigit():
            return int(correct)
        # Try letter to index
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
        result = letter_to_idx.get(correct.upper())
        if result is None:
            logger.warning(
                f"normalize_correct_index: Could not parse string '{correct}' as index or letter"
            )
        return result

    logger.warning(
        f"normalize_correct_index: Unexpected type {type(correct)} for correct answer: {correct}"
    )
    return None


def normalize_scores(scores: List[Any]) -> List[float]:
    """
    Normalize ScoreOutput list to list of floats.

    IMPORTANT: For list and dict scores, this function returns ONLY THE FIRST
    element/value. If you have per-token scores [0.1, 0.2, 0.3], it returns 0.1
    and discards [0.2, 0.3]. This is intentional for aggregate scoring but may
    lose information.

    Different models return scores in different formats:
    - float/int: Direct score
    - List[float]: Per-token scores (returns FIRST ONLY, logs warning if >1)
    - Dict[str, Any]: Structured output (returns FIRST VALUE, logs warning)

    Args:
        scores: List of ScoreOutput values from model

    Returns:
        List of floats (one per score)

    Examples:
        >>> normalize_scores([0.5, -2.3, 1.1])
        [0.5, -2.3, 1.1]
        >>> normalize_scores([[0.5, 0.3], [-2.3], [1.1, 0.9]])  # WARNING: Loses [0.3] and [0.9]!
        [0.5, -2.3, 1.1]
        >>> normalize_scores([{"score": 0.5}, {"score": -2.3}])
        [0.5, -2.3]

    Warnings:
        - Logs warning when list scores have >1 element (data loss)
        - Logs warning when dict scores have >1 key (data loss)
    """
    scores_float = []
    for idx, s in enumerate(scores):
        if isinstance(s, (int, float)):
            scores_float.append(float(s))
        elif isinstance(s, list):
            if not s:
                logger.warning(
                    f"normalize_scores: Empty list at index {idx}, using 0.0 as fallback"
                )
                scores_float.append(0.0)
            else:
                if len(s) > 1:
                    logger.warning(
                        f"normalize_scores: List score at index {idx} has {len(s)} elements {s}. "
                        f"Returning ONLY FIRST value {s[0]}, discarding {s[1:]}. Data loss!"
                    )
                scores_float.append(float(s[0]))
        elif isinstance(s, dict):
            if not s:
                logger.warning(
                    f"normalize_scores: Empty dict at index {idx}, using 0.0 as fallback"
                )
                scores_float.append(0.0)
            else:
                if len(s) > 1:
                    logger.warning(
                        f"normalize_scores: Dict score at index {idx} has {len(s)} keys {list(s.keys())}. "
                        f"Returning ONLY FIRST value. Data loss!"
                    )
                scores_float.append(float(list(s.values())[0]))
        else:
            logger.warning(
                f"normalize_scores: Unexpected type {type(s)} at index {idx}: {s}. Using 0.0 as fallback"
            )
            scores_float.append(0.0)
    return scores_float
