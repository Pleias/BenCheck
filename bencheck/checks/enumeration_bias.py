"""
Enumeration Bias Check - analyzes whether answer position affects selection.

Checks if models exhibit positional bias (preferring option A, B, C, or D regardless
of content), which can indicate dataset construction artifacts or model shortcuts.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..base import BencheckCheck, ScoringModel
from ..types import BencheckQuestion, QuestionType
from ._utils import normalize_correct_index
from ._utils import normalize_scores as normalize_scores_util

# Optional numpy dependency for chi-squared test
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class EnumerationBiasCheck(BencheckCheck):
    """
    Check for enumeration (positional) bias in multiple-choice evaluation.

    This check requires a model to compute predictions and analyzes whether:
    1. The dataset has uniform distribution of correct answers across positions
    2. The model's predictions are uniformly distributed (or biased toward certain positions)
    3. Accuracy varies by position of the correct answer

    A well-designed benchmark should have correct answers uniformly distributed
    across positions, and a well-calibrated model should not favor specific positions.

    Reference:
        Based on analysis from "HellaSwag or HellaBad? 36% of this popular
        LLM benchmark contains errors" (ACL 2026 submission), Section 3.2
    """

    name = "enumeration_bias"

    def __init__(self, model: Optional[ScoringModel] = None):
        """
        Initialize the enumeration bias check.

        Args:
            model: ScoringModel to use for making predictions. Required for this check.
        """
        self.model = model

    def run_on_question(
        self, question: BencheckQuestion, model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single question with model predictions.

        Returns:
            Dict containing:
                - gold_position: Position (index) of the correct answer
                - predicted_position: Position selected by the model
                - is_correct: Whether prediction matches gold
                - scores: Scores for each option (from model)
        """
        if self.model is None:
            raise ValueError("EnumerationBiasCheck requires a model. Pass model in __init__.")

        # Warn if this is a MULTIPLE_CHOICE question (designed for single-choice)
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            logger.warning(
                f"EnumerationBiasCheck: Question {question.id} is MULTIPLE_CHOICE but this check "
                f"assumes single-choice questions. Using first correct answer only."
            )

        # Score all options
        scores = self.model.score_continuations(question.question, question.choices)

        # Convert to floats (handle flexible ScoreOutput)
        scores_float = normalize_scores_util(scores)

        # Model prediction = argmax of scores
        if HAS_NUMPY:
            predicted_position = int(np.argmax(scores_float))
        else:
            predicted_position = max(range(len(scores_float)), key=lambda i: scores_float[i])

        # Get gold position
        gold_position = normalize_correct_index(question.correct)
        if gold_position is None or gold_position >= len(question.choices):
            logger.warning(
                f"EnumerationBiasCheck: Invalid gold position {gold_position} for question {question.id}. "
                f"Expected index < {len(question.choices)}. Skipping analysis."
            )
            return {
                "gold_position": gold_position,
                "predicted_position": predicted_position,
                "is_correct": False,
                "scores": scores_float,
                "error": "invalid_gold_position",
            }

        is_correct = predicted_position == gold_position

        return {
            "gold_position": gold_position,
            "predicted_position": predicted_position,
            "is_correct": is_correct,
            "scores": [round(s, 4) for s in scores_float],
        }

    def aggregate_metrics(self, question_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute dataset-level enumeration bias metrics.

        Returns:
            Dict containing:
                - n_questions: Number of questions analyzed
                - n_with_predictions: Number with valid predictions
                - gold_position_dist: Distribution of correct answers by position
                - pred_position_dist: Distribution of model predictions by position
                - accuracy_by_gold_position: Accuracy broken down by position of correct answer
                - gold_chi2_stat: Chi-squared test for uniformity of gold distribution
                - pred_chi2_stat: Chi-squared test for uniformity of prediction distribution
        """
        # Filter valid results
        valid_results = {qid: res for qid, res in question_results.items() if "error" not in res}

        n_questions = len(valid_results)
        if n_questions == 0:
            return {"n_questions": 0, "error": "no_valid_questions"}

        # Collect position statistics
        gold_positions = [res["gold_position"] for res in valid_results.values()]
        pred_positions = [res["predicted_position"] for res in valid_results.values()]

        max_positions = max(max(gold_positions, default=0), max(pred_positions, default=0)) + 1

        # Distribution of gold answers
        gold_counter = Counter(gold_positions)
        gold_dist = [gold_counter.get(i, 0) for i in range(max_positions)]

        # Distribution of predictions
        pred_counter = Counter(pred_positions)
        pred_dist = [pred_counter.get(i, 0) for i in range(max_positions)]

        # Accuracy by gold position
        accuracy_by_position = {}
        for pos in range(max_positions):
            questions_at_pos = [
                res for res in valid_results.values() if res["gold_position"] == pos
            ]
            if questions_at_pos:
                correct_at_pos = sum(1 for res in questions_at_pos if res["is_correct"])
                accuracy_by_position[f"position_{pos}"] = round(
                    (correct_at_pos / len(questions_at_pos)) * 100, 2
                )
            else:
                accuracy_by_position[f"position_{pos}"] = 0.0

        # Chi-squared test for uniformity
        gold_chi2 = self._chi_squared_uniform(gold_dist)
        pred_chi2 = self._chi_squared_uniform(pred_dist)

        return {
            "n_questions": n_questions,
            "n_positions": max_positions,
            "gold_position_dist": gold_dist,
            "pred_position_dist": pred_dist,
            "accuracy_by_gold_position": accuracy_by_position,
            "overall_accuracy": round(
                (sum(1 for res in valid_results.values() if res["is_correct"]) / n_questions) * 100,
                2,
            ),
            "gold_chi2_stat": round(gold_chi2[0], 4),
            "gold_chi2_p_value": round(gold_chi2[1], 6) if gold_chi2[1] is not None else None,
            "pred_chi2_stat": round(pred_chi2[0], 4),
            "pred_chi2_p_value": round(pred_chi2[1], 6) if pred_chi2[1] is not None else None,
        }

    @staticmethod
    def _chi_squared_uniform(counts: List[int]) -> tuple[float, Optional[float]]:
        """
        Chi-squared test for uniformity of distribution.

        Tests whether the observed counts differ significantly from a uniform
        distribution across positions.

        Returns:
            (chi2_statistic, p_value) where p_value is None if scipy not available

        Note:
            If numpy is not available, uses pure Python implementation (slower).
        """
        total = sum(counts)

        if total == 0 or len(counts) == 0:
            return 0.0, None

        expected = total / len(counts)

        # Calculate chi-squared statistic
        if HAS_NUMPY:
            counts_array = np.array(counts)
            chi2_stat = float(np.sum((counts_array - expected) ** 2 / expected))
        else:
            # Pure Python fallback
            chi2_stat = sum((count - expected) ** 2 / expected for count in counts)

        # Try to compute p-value with scipy
        try:
            from scipy.stats import chi2 as scipy_chi2

            df = len(counts) - 1
            p_value = float(scipy_chi2.sf(chi2_stat, df))
            return chi2_stat, p_value
        except ImportError:
            return chi2_stat, None
