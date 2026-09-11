"""
None-of-the-Above Check - evaluates how distinguishable wrong answers are.

This check tests whether a model can identify when the correct answer is missing
by removing it and adding a "None of the above" option. If the model selects
the placeholder, it suggests the incorrect options were already obviously wrong,
making the question too easy.

This check ONLY works with generation-based models (not log-likelihood).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..base import BencheckCheck
from ..types import BencheckQuestion
from ._utils import normalize_correct_index


class NoneOfTheAboveCheck(BencheckCheck):
    """
    Check for question quality by removing correct answer and adding "None of the above".

    This diagnostic runs two passes:
    1. Baseline: Generate answer with original options
    2. Placeholder: Remove correct answer, add "None of the above" at the end, generate again

    If the model selects "None of the above", it indicates the incorrect options
    were obviously wrong (low quality distractors).

    IMPORTANT: This check requires a generative model (with .generate() method).
    It does NOT work with log-likelihood scoring.

    Reference:
        Inspired by similar diagnostics in benchmark quality research
    """

    name = "none_of_the_above"

    def __init__(
        self,
        model: Optional[Any] = None,
        placeholder_text: str = "None of the above",
    ):
        """
        Initialize the check.

        Args:
            model: Generative model with .generate() method
            placeholder_text: Text to use as placeholder (default: "None of the above")
        """
        self.model = model
        self.placeholder_text = placeholder_text

    def _score_question(self, question_text: str, choices: List[str]) -> int:
        """
        Score a question using generation and return predicted index.

        Args:
            question_text: Question text
            choices: Answer choices

        Returns:
            Index of predicted answer
        """
        if not hasattr(self.model, "generate"):
            raise ValueError(
                "NoneOfTheAboveCheck requires a model with .generate() method. "
                "This check does not support log-likelihood scoring."
            )

        # Format as multiple choice prompt
        prompt = self._format_mcq_prompt(question_text, choices)
        output = self.model.generate(prompt, max_tokens=10)

        # Parse output (extract A, B, C, D, etc.)
        prediction = self._parse_generation(output, len(choices))
        return prediction

    def _format_mcq_prompt(self, question: str, choices: List[str]) -> str:
        """
        Format question as multiple choice prompt.

        Args:
            question: Question text
            choices: List of answer choices

        Returns:
            Formatted prompt string
        """
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        formatted = []
        for i, choice in enumerate(choices):
            label = labels[i] if i < len(labels) else str(i + 1)
            formatted.append(f"{label}) {choice}")

        return (
            f"Question: {question}\n\n"
            f"Choices:\n" + "\n".join(formatted) + "\n\n"
            f"Provide only the letter of the correct answer.\n"
            f"Answer:"
        )

    def _parse_generation(self, output: str, num_choices: int) -> int:
        """
        Parse generated output to extract answer index.

        Args:
            output: Generated text from model
            num_choices: Number of available choices

        Returns:
            Index of predicted answer (0-based)
        """
        import re

        output = output.strip()
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        valid_labels = labels[:num_choices]

        # Pattern 1: Look for "Answer: X" style responses
        answer_patterns = [
            r"(?:answer|correct|choice|option)(?:\s+is)?:\s*([A-H])",
            r"(?:the|my)?\s*(?:answer|correct|choice)\s+is\s+([A-H])",
            r"\b([A-H])\s*(?:is|would be|appears to be)\s+(?:the\s+)?(?:correct|right|answer)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in valid_labels:
                    return valid_labels.index(letter)

        # Pattern 2: Formatted answers like "(A)", "A)", "A."
        formatted_patterns = [
            r"[\(\[]([A-H])[\)\]]",
            r"\b([A-H])[\.:\)]",
        ]

        for pattern in formatted_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in valid_labels:
                    return valid_labels.index(letter)

        # Pattern 3: Standalone letter at start
        standalone_match = re.match(r"^\s*([A-H])\b", output, re.IGNORECASE)
        if standalone_match:
            letter = standalone_match.group(1).upper()
            if letter in valid_labels:
                return valid_labels.index(letter)

        # Pattern 4: Any valid letter
        for letter in valid_labels:
            if re.search(r"\b" + letter + r"\b", output, re.IGNORECASE):
                return valid_labels.index(letter)

        # Fallback: return first choice (arbitrary)
        logger.warning(f"Could not parse generation output: '{output}'. Defaulting to index 0.")
        return 0

    def run_on_question(
        self, question: BencheckQuestion, model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run the check on a single question.

        Returns:
            Dict containing:
                - baseline_prediction: Model's original prediction (index)
                - baseline_correct: Whether baseline was correct
                - placeholder_prediction: Model's prediction with removed correct answer
                - placeholder_selected: Whether model chose "None of the above"
                - prediction_changed: Whether prediction changed
                - original_correct_idx: Original correct answer index
                - placeholder_idx: Index of "None of the above" in mutated choices
        """
        if self.model is None:
            raise ValueError("NoneOfTheAboveCheck requires a model. Pass model in __init__.")

        # Get correct index
        correct_idx = normalize_correct_index(question.correct)
        if correct_idx is None or correct_idx >= len(question.choices):
            logger.warning(
                f"NoneOfTheAboveCheck: Invalid correct index {correct_idx} for question {question.id}. "
                f"Expected index < {len(question.choices)}. Skipping analysis."
            )
            return {
                "error": "invalid_correct_index",
                "correct_idx": correct_idx,
            }

        # Baseline: score original options
        baseline_pred = self._score_question(question.question, question.choices)
        baseline_correct = baseline_pred == correct_idx

        # Placeholder: remove correct answer, add "None of the above" at the end
        mutated_choices = [choice for i, choice in enumerate(question.choices) if i != correct_idx]
        mutated_choices.append(self.placeholder_text)

        # The placeholder is now at the last index
        placeholder_idx = len(mutated_choices) - 1

        # Score with mutated choices
        placeholder_pred = self._score_question(question.question, mutated_choices)

        # Check if model selected "None of the above"
        placeholder_selected = placeholder_pred == placeholder_idx

        # Map placeholder prediction back to original indices for comparison
        # If model selected position before deletion point, index stays same
        # If model selected position after deletion point, add 1 to account for removed item
        # If model selected placeholder (last position), it's a new choice
        if placeholder_pred < correct_idx:
            mapped_pred = placeholder_pred
        elif placeholder_pred == placeholder_idx:
            mapped_pred = None  # Selected the new option
        else:
            mapped_pred = placeholder_pred + 1

        prediction_changed = baseline_pred != mapped_pred

        return {
            "baseline_prediction": baseline_pred,
            "baseline_correct": baseline_correct,
            "placeholder_prediction": placeholder_pred,
            "placeholder_selected": placeholder_selected,
            "prediction_changed": prediction_changed,
            "original_correct_idx": correct_idx,
            "placeholder_idx": placeholder_idx,
            "num_original_choices": len(question.choices),
            "num_mutated_choices": len(mutated_choices),
        }

    def aggregate_metrics(self, question_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute dataset-level metrics.

        Returns:
            Dict containing:
                - n_questions: Number of valid questions
                - baseline_accuracy: Accuracy on original questions
                - placeholder_selected_rate: How often model chose "None of the above" (GOOD)
                - prediction_changed_rate: How often prediction changed when correct answer removed
                - correct_to_placeholder_rate: baseline correct → selected NOTA (GOOD - recognizes absence)
                - incorrect_to_placeholder_rate: baseline wrong → selected NOTA (GOOD - rejects wrong)
                - correct_to_wrong_rate: baseline correct → selected wrong distractor (BAD!)
                - incorrect_to_wrong_rate: baseline wrong → stuck with wrong (BAD!)
        """
        # Filter valid results
        valid_results = {qid: res for qid, res in question_results.items() if "error" not in res}

        n_questions = len(valid_results)
        if n_questions == 0:
            return {"n_questions": 0, "error": "no_valid_questions"}

        baseline_correct = sum(1 for res in valid_results.values() if res["baseline_correct"])
        placeholder_selected = sum(
            1 for res in valid_results.values() if res["placeholder_selected"]
        )
        prediction_changed = sum(1 for res in valid_results.values() if res["prediction_changed"])

        # Transition metrics (what happens when correct answer is removed)
        # GOOD: Model selects NOTA (recognizes correct answer is missing)
        correct_to_placeholder = sum(
            1 for res in valid_results.values()
            if res["baseline_correct"] and res["placeholder_selected"]
        )
        incorrect_to_placeholder = sum(
            1 for res in valid_results.values()
            if not res["baseline_correct"] and res["placeholder_selected"]
        )

        # BAD: Model doesn't select NOTA (fails to recognize absence of correct answer)
        correct_to_wrong = sum(
            1 for res in valid_results.values()
            if res["baseline_correct"] and not res["placeholder_selected"]
        )
        incorrect_to_wrong = sum(
            1 for res in valid_results.values()
            if not res["baseline_correct"] and not res["placeholder_selected"]
        )

        # Calculate percentages
        baseline_accuracy_pct = round((baseline_correct / n_questions) * 100, 2)

        # In mutated version, correct answer is ALWAYS "None of the above" (since we removed the actual correct answer)
        placeholder_accuracy_pct = round((placeholder_selected / n_questions) * 100, 2)
        accuracy_delta_pct = round(placeholder_accuracy_pct - baseline_accuracy_pct, 2)

        return {
            "n_questions": n_questions,
            "baseline_accuracy": baseline_accuracy_pct,
            "baseline_correct_count": baseline_correct,
            "placeholder_accuracy": placeholder_accuracy_pct,
            "accuracy_delta_pct": accuracy_delta_pct,
            "placeholder_selected_count": placeholder_selected,
            "placeholder_selected_rate": round((placeholder_selected / n_questions) * 100, 2),
            "prediction_changed_count": prediction_changed,
            "prediction_changed_rate": round((prediction_changed / n_questions) * 100, 2),
            # Transition metrics (GOOD - selects NOTA)
            "correct_to_placeholder_count": correct_to_placeholder,
            "correct_to_placeholder_rate": round((correct_to_placeholder / n_questions) * 100, 2),
            "incorrect_to_placeholder_count": incorrect_to_placeholder,
            "incorrect_to_placeholder_rate": round((incorrect_to_placeholder / n_questions) * 100, 2),
            # Error metrics (BAD - doesn't select NOTA)
            "correct_to_wrong_count": correct_to_wrong,
            "correct_to_wrong_rate": round((correct_to_wrong / n_questions) * 100, 2),
            "incorrect_to_wrong_count": incorrect_to_wrong,
            "incorrect_to_wrong_rate": round((incorrect_to_wrong / n_questions) * 100, 2),
        }
