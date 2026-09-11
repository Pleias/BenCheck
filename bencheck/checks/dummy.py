"""Diagnostic check used for exercising the bencheck stack."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..base import BencheckCheck, ScoringModel
from ..types import BencheckQuestion


class DummyCheck(BencheckCheck):
    """
    Minimal placeholder diagnostic check with light-weight heuristics.

    Can work in two modes:
    1. Model-free (default): Uses simple lexical overlap heuristic
    2. Model-based: Uses provided ScoringModel for scoring

    This allows DummyCheck to work out-of-the-box for quick testing.
    """

    name = "dummy_check"

    def __init__(self, model: Optional[ScoringModel] = None, spread_threshold: float = 0.5) -> None:
        """
        Initialize DummyCheck.

        Args:
            model: Optional ScoringModel. If None, uses built-in lexical heuristic.
            spread_threshold: Threshold for "challenging" classification
        """
        self.model = model
        self.spread_threshold = spread_threshold

    def run_on_question(
        self, question: BencheckQuestion, model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Run check on a single question."""
        # Get scores (either from model or built-in heuristic)
        if self.model is not None:
            # Use provided model
            scores = [
                self.model.score_continuation(question.question, option)
                for option in question.choices
            ]
            # Normalize ScoreOutput to floats
            scores = self._normalize_scores(scores)
        else:
            # Use built-in lexical overlap heuristic (model-free)
            scores = [
                self._lexical_overlap_score(question.question, option)
                for option in question.choices
            ]

        predicted_choice = max(range(len(scores)), key=lambda idx: scores[idx])

        # Handle different correct formats
        correct_indices = self._normalize_correct_to_indices(question.correct)
        is_correct = predicted_choice in correct_indices

        spread = max(scores) - min(scores) if scores else 0.0
        challenging = spread < self.spread_threshold
        cheatable = bool(question.metadata.get("cheatable"))

        return {
            "predicted_choice": predicted_choice,
            "is_correct": is_correct,
            "challenging": challenging,
            "cheatable": cheatable,
            "score_spread": round(spread, 4),
            "scores": [round(score, 4) for score in scores],
        }

    def aggregate_metrics(self, question_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Compute dataset-level metrics from per-question results."""
        total_questions = len(question_results)
        if total_questions == 0:
            return {
                "accuracy": 0.0,
                "challenging_questions_pct": 0.0,
                "cheatable_questions_pct": 0.0,
            }

        correct_predictions = sum(
            1 for result in question_results.values() if result.get("is_correct", False)
        )
        challenging_questions = sum(
            1 for result in question_results.values() if result.get("challenging", False)
        )
        cheatable_questions = sum(
            1 for result in question_results.values() if result.get("cheatable", False)
        )

        return {
            "accuracy": round((correct_predictions / total_questions) * 100, 2),
            "challenging_questions_pct": round((challenging_questions / total_questions) * 100, 2),
            "cheatable_questions_pct": round((cheatable_questions / total_questions) * 100, 2),
        }

    @staticmethod
    def _normalize_scores(scores: list) -> list[float]:
        """
        Normalize ScoreOutput values to floats.

        Handles:
        - float: return as-is
        - List[float]: take mean (or first element if single)
        - Dict[str, Any]: take 'score' key, or first numeric value

        Raises:
            ValueError: If score format is unrecognized
        """
        result = []
        for s in scores:
            if isinstance(s, (int, float)):
                result.append(float(s))
            elif isinstance(s, list):
                # Take mean of list scores
                if s:
                    result.append(float(sum(s) / len(s)))
                else:
                    result.append(0.0)
            elif isinstance(s, dict):
                # Try 'score' key first, then first numeric value
                if "score" in s:
                    result.append(float(s["score"]))
                else:
                    # Take first numeric value
                    for v in s.values():
                        if isinstance(v, (int, float)):
                            result.append(float(v))
                            break
                    else:
                        result.append(0.0)
            else:
                raise ValueError(
                    f"Cannot normalize score of type {type(s)}. "
                    f"Expected float, List[float], or Dict[str, Any]"
                )
        return result

    @staticmethod
    def _lexical_overlap_score(context: str, option: str) -> float:
        """
        Simple lexical overlap heuristic (model-free scoring).

        Score = (number of overlapping words) + 0.1 * (option length)
        """
        context_tokens = set(token.lower().strip(".,?!") for token in context.split())
        option_tokens = [token.lower().strip(".,?!") for token in option.split()]
        overlap = sum(1 for token in option_tokens if token in context_tokens)
        return float(overlap) + len(option_tokens) * 0.1

    @staticmethod
    def _normalize_correct_to_indices(correct: Any) -> list[int]:
        """Convert correct field to list of indices."""
        if isinstance(correct, int):
            return [correct]
        elif isinstance(correct, list):
            # Assume all elements are ints
            if all(isinstance(x, int) for x in correct):
                return correct
            # If strings, try to convert
            return [int(x) for x in correct if str(x).isdigit()]
        elif isinstance(correct, str) and correct.isdigit():
            return [int(correct)]
        return []
