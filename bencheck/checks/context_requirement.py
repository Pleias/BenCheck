"""
Context Requirement Check - Tests if questions need context to be answered.

This check evaluates whether models can identify correct answers even when the
question context is removed or replaced with meaningless text (Lorem Ipsum).
High accuracy on empty/Lorem Ipsum questions indicates the dataset contains
inherent bias or that wrong answers are obviously incorrect.
"""

from __future__ import annotations

# Import question transforms
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

from ..base import BencheckCheck
from ..types import BencheckQuestion
from ._utils import normalize_correct_index
from ._utils import normalize_scores as normalize_scores_util

try:
    from ..utils.question_transforms import (
        generate_lorem_ipsum,
        remove_question_text,
        replace_with_lorem_ipsum,
    )

    HAS_TRANSFORMS = True
except ImportError:
    HAS_TRANSFORMS = False


class ContextRequirementCheck(BencheckCheck):
    """
    Check whether questions require context to answer correctly.

    This diagnostic runs multiple passes:
    1. Baseline: Score with original question
    2. Empty: Replace question with empty string
    3. Lorem Ipsum: Replace question with Lorem Ipsum (matching length)

    If model accuracy remains high without context, it suggests:
    - Incorrect options are obviously wrong (low quality distractors)
    - Dataset contains inherent position/length biases
    - Questions are answerable without reading the context

    This is inspired by similar tests in:
    - "Do Language Models Need Context?" (Various papers)
    - Dataset quality research showing context-independent bias

    Statistical Analysis:
    The check always computes comprehensive statistical analysis including:
    - Binomial hypothesis tests (H0: performance = random guessing)
    - Wilson score confidence intervals
    - Cohen's h effect sizes
    - Statistical interpretation of whether questions require context

    Scoring Modes:
    - log_likelihood: Score based on continuation probabilities (requires ScoringModel)
    - generation: Generate answer and compare (requires GenerativeModel)

    Example:
        # With log-likelihood scoring
        check = ContextRequirementCheck(
            model=my_scoring_model,
            scoring_mode="log_likelihood"
        )

        # With generation scoring
        check = ContextRequirementCheck(
            model=my_generative_model,
            scoring_mode="generation",
            significance_level=0.05,
            min_effect_size=0.2
        )
    """

    name = "context_requirement"

    def __init__(
        self,
        model: Optional[Any] = None,
        transform_type: Literal["empty", "lorem_ipsum", "both"] = "both",
        lorem_match_by: Literal["characters", "words"] = "characters",
        scoring_mode: Literal["log_likelihood", "generation"] = "log_likelihood",
        significance_level: float = 0.05,
        confidence_level: float = 0.95,
        min_effect_size: float = 0.2,
    ):
        """
        Initialize the check.

        Args:
            model: Model to use (ScoringModel for log_likelihood, GenerativeModel for generation)
            transform_type: Type of transformation to apply
                - "empty": Replace question with empty string
                - "lorem_ipsum": Replace with Lorem Ipsum
                - "both": Test both transformations
            lorem_match_by: For lorem_ipsum, how to match length
                - "characters": Match character count
                - "words": Match word count
            scoring_mode: How to score answers
                - "log_likelihood": Score continuations with logprobs (ScoringModel)
                - "generation": Generate answer and parse (GenerativeModel)
            significance_level: Alpha level for hypothesis tests (default: 0.05)
            confidence_level: Confidence level for intervals (default: 0.95)
            min_effect_size: Minimum Cohen's h to consider meaningful (default: 0.2)

        Note:
            Statistical analysis (binomial tests, confidence intervals, effect sizes)
            is always enabled and computed automatically. If scipy is not available,
            graceful fallback to basic analysis is used.
        """
        if not HAS_TRANSFORMS:
            raise ImportError(
                "question_transforms module not available. "
                "This check requires bencheck.utils.question_transforms"
            )

        self.model = model
        self.transform_type = transform_type
        self.lorem_match_by = lorem_match_by
        self.scoring_mode = scoring_mode
        self.significance_level = significance_level
        self.confidence_level = confidence_level
        self.min_effect_size = min_effect_size

    def run_on_question(
        self, question: BencheckQuestion, model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run the check on a single question.

        Returns:
            Dict containing:
                - baseline_prediction: Model's prediction with original question
                - baseline_correct: Whether baseline was correct
                - empty_prediction: Prediction with empty question (if applicable)
                - empty_correct: Whether empty was correct
                - lorem_prediction: Prediction with Lorem Ipsum (if applicable)
                - lorem_correct: Whether Lorem was correct
                - context_dependency: Score indicating need for context (0-1)
        """
        if self.model is None:
            raise ValueError("ContextRequirementCheck requires a model. Pass model in __init__.")

        # Get correct index
        correct_idx = normalize_correct_index(question.correct)
        if correct_idx is None or correct_idx >= len(question.choices):
            logger.warning(
                f"ContextRequirementCheck: Invalid correct index {correct_idx} for question {question.id}. "
                f"Expected index < {len(question.choices)}. Skipping analysis."
            )
            return {"error": "invalid_correct_index"}

        results = {
            "correct_idx": correct_idx,
            "scoring_mode": self.scoring_mode,
            "n_choices": len(question.choices),  # Store for random baseline calculation
        }

        # Baseline: Original question
        baseline_pred = self._score_question(question.question, question.choices)
        results["baseline_prediction"] = baseline_pred
        results["baseline_correct"] = baseline_pred == correct_idx

        # Empty question
        if self.transform_type in ("empty", "both"):
            empty_pred = self._score_question("", question.choices)
            results["empty_prediction"] = empty_pred
            results["empty_correct"] = empty_pred == correct_idx
            results["empty_changed"] = empty_pred != baseline_pred

        # Lorem Ipsum
        if self.transform_type in ("lorem_ipsum", "both"):
            lorem_text = generate_lorem_ipsum(
                len(question.question.split())
                if self.lorem_match_by == "words"
                else len(question.question),
                match_type=self.lorem_match_by,
            )
            lorem_pred = self._score_question(lorem_text, question.choices)
            results["lorem_prediction"] = lorem_pred
            results["lorem_correct"] = lorem_pred == correct_idx
            results["lorem_changed"] = lorem_pred != baseline_pred

        # Compute context dependency score (0 = no context needed, 1 = context required)
        # If predictions change when context is removed, context is required
        dependency_score = 0.0
        if "empty_changed" in results:
            dependency_score += 0.5 if results["empty_changed"] else 0.0
        if "lorem_changed" in results:
            dependency_score += 0.5 if results["lorem_changed"] else 0.0

        results["context_dependency"] = dependency_score

        return results

    def _score_question(self, question_text: str, choices: List[str]) -> int:
        """
        Score a question and return predicted index.

        Args:
            question_text: Question text (may be empty or Lorem Ipsum)
            choices: Answer choices

        Returns:
            Index of predicted answer
        """
        if self.scoring_mode == "log_likelihood":
            # Use ScoringModel protocol
            if not hasattr(self.model, "score_continuations"):
                raise ValueError(
                    "Model does not implement score_continuations(). "
                    "Use scoring_mode='generation' or provide a ScoringModel."
                )

            scores = self.model.score_continuations(question_text, choices)
            scores_float = normalize_scores_util(scores)
            return int(scores_float.index(max(scores_float)))

        elif self.scoring_mode == "generation":
            # Use GenerativeModel protocol
            if not hasattr(self.model, "generate"):
                raise ValueError(
                    "Model does not implement generate(). "
                    "Use scoring_mode='log_likelihood' or provide a GenerativeModel."
                )

            # Format as multiple choice prompt
            prompt = self._format_mcq_prompt(question_text, choices)
            output = self.model.generate(prompt, max_tokens=10)

            # Parse output (extract A, B, C, D, etc.)
            prediction = self._parse_generation(output, len(choices))
            return prediction

        else:
            raise ValueError(f"Invalid scoring_mode: {self.scoring_mode}")

    def _format_mcq_prompt(self, question: str, choices: List[str]) -> str:
        """
        Format question as multiple choice prompt.

        Creates a structured prompt that encourages models to respond
        with just the letter, improving parsing reliability.

        Args:
            question: Question text (may be empty for context removal tests)
            choices: List of answer choices

        Returns:
            Formatted prompt string
        """
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        formatted = []
        for i, choice in enumerate(choices):
            label = labels[i] if i < len(labels) else str(i + 1)
            formatted.append(f"{label}) {choice}")

        if question:
            return (
                f"Question: {question}\n\n"
                f"Choices:\n" + "\n".join(formatted) + "\n\n"
                f"Provide only the letter of the correct answer (A, B, C, or D).\n"
                f"Answer:"
            )
        else:
            return (
                f"Choose the correct answer from the options below.\n\n"
                f"Choices:\n" + "\n".join(formatted) + "\n\n"
                f"Provide only the letter of the correct answer (A, B, C, or D).\n"
                f"Answer:"
            )

    def _parse_generation(self, output: str, num_choices: int) -> int:
        """
        Parse generated output to extract answer index.

        Handles various response formats:
        - "Answer: A", "The answer is B", "Correct: C"
        - "(A)", "A)", "A."
        - Standalone letters: "A", "B"
        - Number formats: "1", "2" (1-based), "0" (0-based)
        - Multi-line responses (takes first valid answer)

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

        # Pattern 1: Look for "Answer: X", "Correct: X", "The answer is X", etc.
        # This is the most reliable pattern
        answer_patterns = [
            r"(?:answer|correct|choice|option)(?:\s+is)?:\s*([A-H])",  # "Answer: A"
            r"(?:the|my)?\s*(?:answer|correct|choice)\s+is\s+([A-H])",  # "The answer is B"
            r"\b([A-H])\s*(?:is|would be|appears to be)\s+(?:the\s+)?(?:correct|right|answer)",  # "A is correct"
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in valid_labels:
                    idx = valid_labels.index(letter)
                    logger.debug(f"Parsed answer '{letter}' (index {idx}) from pattern: {pattern}")
                    return idx

        # Pattern 2: Look for formatted answers like "(A)", "A)", "A.", "[A]"
        formatted_patterns = [
            r"[\(\[]([A-H])[\)\]]",  # (A), [A]
            r"\b([A-H])[\.:\)]",  # A., A:, A)
        ]

        for pattern in formatted_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in valid_labels:
                    idx = valid_labels.index(letter)
                    logger.debug(f"Parsed formatted answer '{letter}' (index {idx})")
                    return idx

        # Pattern 3: Standalone letter at start of output (common with instruction-tuned models)
        standalone_match = re.match(r"^\s*([A-H])\b", output, re.IGNORECASE)
        if standalone_match:
            letter = standalone_match.group(1).upper()
            if letter in valid_labels:
                idx = valid_labels.index(letter)
                logger.debug(f"Parsed standalone answer '{letter}' (index {idx})")
                return idx

        # Pattern 4: Try to find any letter in the valid range (less reliable)
        for letter in valid_labels:
            # Use word boundary to avoid matching letters in words
            if re.search(r"\b" + letter + r"\b", output, re.IGNORECASE):
                idx = valid_labels.index(letter)
                logger.debug(f"Parsed letter '{letter}' (index {idx}) from word boundary search")
                return idx

        # Pattern 5: Try number formats (1-based: "1", "2", etc.)
        number_pattern = r"(?:answer|correct|choice|option)(?:\s+is)?:\s*(\d+)"
        match = re.search(number_pattern, output, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            # Convert 1-based to 0-based
            if 1 <= num <= num_choices:
                idx = num - 1
                logger.debug(f"Parsed 1-based number '{num}' (index {idx})")
                return idx

        # Pattern 6: Fallback - first digit in output (1-based assumed)
        digit_match = re.search(r"\b([1-9])\b", output)
        if digit_match:
            num = int(digit_match.group(1))
            if num <= num_choices:
                idx = num - 1
                logger.debug(f"Parsed fallback number '{num}' (index {idx})")
                return idx

        # Pattern 7: Detect refusal to answer (e.g., "Neither", "Cannot answer", "Not relevant")
        # These indicate model recognizes the question doesn't make sense with transformed context
        refusal_patterns = [
            r"\b(?:neither|none)(?:\s+of\s+(?:the|these))?\s+(?:choice|option|answer)s?\b",
            r"\b(?:cannot|can't|unable to)\s+(?:answer|determine|choose)\b",
            r"\b(?:not|isn't|aren't)\s+(?:relevant|related|applicable)\b",
            r"\b(?:doesn't|does not)\s+(?:make sense|relate)\b",
            r"^(?:neither|none)\b",  # Leading "Neither" or "None"
            # Nonsensical/gibberish detection
            r"\b(?:is|are|seems?)\s+(?:nonsensical|gibberish|meaningless|incoherent)\b",
            r"(?:question|text|prompt)\s+(?:is|appears?|seems?)\s+(?:nonsensical|gibberish|meaningless)\b",
            # No correct answer responses
            r"\b(?:no|there is no)\s+correct\s+answer\b",
            r"\bnone\s+of\s+(?:these|the\s+(?:choices|options|answers))\s+(?:is|are)\s+correct\b",
            r"\b(?:there are|there is)\s+no\s+(?:choices?|options?)\b",
            # Model explains task instead of answering
            r"^(?:The|This)\s+question\s+(?:is asking|asks|requests)\b",
            # Inappropriate/policy violation
            r"\b(?:inappropriate|concerning|violates?\s+(?:the\s+)?policy)\b",
        ]

        for pattern in refusal_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                # Model refused to answer - return random choice to simulate guessing
                # This is more honest than always defaulting to 0
                idx = random.randrange(num_choices)
                logger.info(
                    f"Model refused to answer (matched pattern: {pattern}). "
                    f"Output: '{output[:100]}...'. Using random choice (index {idx})."
                )
                return idx

        # No valid answer found - log warning and default to first choice
        logger.warning(
            f"Failed to parse answer from output: '{output[:100]}...'. "
            f"Defaulting to index 0. Consider improving prompt or parsing."
        )
        return 0

    def aggregate_metrics(self, question_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute dataset-level metrics.

        Returns:
            Dict with metrics including:
                - baseline_accuracy: Accuracy with original questions
                - empty_accuracy: Accuracy with empty questions (if tested)
                - lorem_accuracy: Accuracy with Lorem Ipsum (if tested)
                - context_dependency_score: Mean dependency score (0-1)
                - questions_requiring_context_pct: % of questions needing context
        """
        if not question_results:
            return {}

        # Filter out errors
        valid_results = [r for r in question_results.values() if "error" not in r]
        if not valid_results:
            return {"error": "no_valid_results"}

        n_questions = len(valid_results)

        # Baseline accuracy
        baseline_correct = sum(1 for r in valid_results if r.get("baseline_correct", False))
        baseline_acc = (baseline_correct / n_questions) * 100

        metrics = {
            "n_questions": n_questions,
            "scoring_mode": valid_results[0].get("scoring_mode", "unknown"),
            "baseline_accuracy": baseline_acc,
        }

        # Empty question metrics
        if "empty_correct" in valid_results[0]:
            empty_correct = sum(1 for r in valid_results if r.get("empty_correct", False))
            empty_acc = (empty_correct / n_questions) * 100
            metrics["empty_accuracy"] = empty_acc
            metrics["empty_vs_baseline_drop"] = baseline_acc - empty_acc

        # Lorem Ipsum metrics
        if "lorem_correct" in valid_results[0]:
            lorem_correct = sum(1 for r in valid_results if r.get("lorem_correct", False))
            lorem_acc = (lorem_correct / n_questions) * 100
            metrics["lorem_accuracy"] = lorem_acc
            metrics["lorem_vs_baseline_drop"] = baseline_acc - lorem_acc

        # Context dependency
        dependency_scores = [r.get("context_dependency", 0.0) for r in valid_results]
        mean_dependency = sum(dependency_scores) / len(dependency_scores)
        metrics["mean_context_dependency"] = mean_dependency

        # % of questions that require context (dependency > 0.5)
        requiring_context = sum(1 for score in dependency_scores if score > 0.5)
        metrics["questions_requiring_context_pct"] = (requiring_context / n_questions) * 100

        # Prediction agreement with full-prompt baseline (regardless of correctness)
        for changed_key, label in [("empty_changed", "empty"), ("lorem_changed", "lorem")]:
            if changed_key not in valid_results[0]:
                continue
            same = sum(1 for r in valid_results if not r.get(changed_key))
            metrics[f"{label}_prediction_agreement_pct"] = round(same / n_questions * 100, 2)

        # Add statistical analysis if enabled
        # Always run statistical analysis (graceful fallback if scipy unavailable)
        stats = self._run_statistical_analysis(valid_results)
        if stats:
            metrics["statistical_analysis"] = stats

        return metrics

    def _run_statistical_analysis(self, valid_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run advanced statistical analysis on results.

        Performs hypothesis testing to determine if model performance on transformed
        questions is significantly better than random chance.

        Args:
            valid_results: List of valid question results

        Returns:
            Dictionary with statistical test results for each transformation type
        """
        baseline_scores = []
        empty_scores = []
        lorem_scores = []

        for res in valid_results:
            baseline_scores.append(1.0 if res.get("baseline_correct") else 0.0)
            if "empty_correct" in res:
                empty_scores.append(1.0 if res.get("empty_correct") else 0.0)
            if "lorem_correct" in res:
                lorem_scores.append(1.0 if res.get("lorem_correct") else 0.0)

        # Determine n_choices (assume uniform across dataset)
        n_choices = valid_results[0].get("n_choices", 4) if valid_results else 4

        results = {}

        # Analyze based on transform type
        if self.transform_type in ("empty", "both") and empty_scores:
            results["empty"] = self._analyze_transform(
                baseline_scores, empty_scores, "empty", n_choices
            )

        if self.transform_type in ("lorem_ipsum", "both") and lorem_scores:
            results["lorem_ipsum"] = self._analyze_transform(
                baseline_scores, lorem_scores, "lorem_ipsum", n_choices
            )

        return results

    def _analyze_transform(
        self,
        baseline_scores: List[float],
        transformed_scores: List[float],
        transform_name: str,
        n_choices: int = 4,
    ) -> Dict[str, Any]:
        """Analyze single transformation with statistical tests.

        Tests null hypothesis: Performance on transformed questions = random guessing

        Args:
            baseline_scores: Scores on original questions (for reference)
            transformed_scores: Scores on transformed questions
            transform_name: Name of transformation (for interpretation)
            n_choices: Number of answer choices (used to calculate random baseline)

        Returns:
            Dictionary with test results, effect sizes, and interpretation
        """
        n = len(transformed_scores)
        n_correct = sum(1 for s in transformed_scores if s > 0.5)
        transformed_accuracy = n_correct / n if n > 0 else 0.0

        # Calculate random baseline based on number of choices
        random_baseline = 1.0 / n_choices

        # Binomial test against random baseline
        test_stat, pvalue = self._binomial_test_vs_baseline(n_correct, n, random_baseline)

        # Wilson score confidence interval
        ci_lower, ci_upper = self._binomial_ci(n_correct, n, self.confidence_level)

        # Cohen's h effect size
        cohens_h = self._cohens_h(transformed_accuracy, random_baseline)

        # Interpretation
        beats_random = (pvalue < self.significance_level) and (
            transformed_accuracy > random_baseline
        )
        is_meaningful = cohens_h >= self.min_effect_size

        return {
            "n_questions": n,
            "transformed_accuracy": round(transformed_accuracy, 4),
            "transformed_ci_lower": round(ci_lower, 4),
            "transformed_ci_upper": round(ci_upper, 4),
            "random_baseline": random_baseline,
            "binomial_test_statistic": round(test_stat, 4),
            "binomial_test_pvalue": round(pvalue, 6),
            "cohens_h_vs_random": round(cohens_h, 4),
            "beats_random": beats_random,
            "is_meaningful_effect": is_meaningful,
            "requires_context": not beats_random,
            "interpretation": self._interpret_results(
                transformed_accuracy,
                random_baseline,
                pvalue,
                cohens_h,
                beats_random,
                is_meaningful,
                transform_name,
            ),
        }

    def _binomial_test_vs_baseline(
        self, successes: int, n: int, baseline: float
    ) -> tuple[float, float]:
        """One-sample binomial test against baseline probability.

        Tests if observed success rate is significantly different from baseline.
        Uses normal approximation for large samples.

        Args:
            successes: Number of successes observed
            n: Total number of trials
            baseline: Expected probability under null hypothesis

        Returns:
            Tuple of (z_statistic, one_sided_pvalue)
            one_sided_pvalue tests if successes > baseline (upper tail)
        """
        if n == 0:
            return 0.0, 1.0

        observed_p = successes / n

        # Use normal approximation for n*p >= 5 and n*(1-p) >= 5
        if n * baseline >= 5 and n * (1 - baseline) >= 5:
            # Z-test
            se = math.sqrt(baseline * (1 - baseline) / n)
            if se == 0:
                return 0.0, 1.0 if observed_p <= baseline else 0.0

            z = (observed_p - baseline) / se

            # One-sided p-value (upper tail: H1: p > baseline)
            pvalue = 0.5 * (1 - math.erf(z / math.sqrt(2)))

            return z, pvalue
        else:
            # For small samples, approximate p-value
            z = 0.0
            if observed_p > baseline:
                se = math.sqrt(baseline * (1 - baseline) / n)
                if se > 0:
                    pvalue = 0.5 * (1 - math.erf((observed_p - baseline) / se / math.sqrt(2)))
                else:
                    pvalue = 0.0
            else:
                pvalue = 1.0

            return z, pvalue

    def _binomial_ci(self, successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
        """Wilson score confidence interval for binomial proportion.

        More accurate than normal approximation for small samples.

        Args:
            successes: Number of successes
            n: Total number of trials
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n == 0:
            return 0.0, 0.0

        p = successes / n
        z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

        return max(0.0, center - margin), min(1.0, center + margin)

    def _cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h effect size for difference between proportions.

        Args:
            p1: First proportion
            p2: Second proportion

        Returns:
            Cohen's h effect size (0.2=small, 0.5=medium, 0.8=large)
        """
        # Prevent division by zero at boundaries
        p1 = max(0.001, min(0.999, p1))
        p2 = max(0.001, min(0.999, p2))

        phi1 = 2 * math.asin(math.sqrt(p1))
        phi2 = 2 * math.asin(math.sqrt(p2))

        return abs(phi1 - phi2)

    def _interpret_results(
        self,
        acc: float,
        baseline: float,
        pvalue: float,
        cohens_h: float,
        beats_random: bool,
        is_meaningful: bool,
        transform_name: str,
    ) -> str:
        """Generate human-readable interpretation of statistical results.

        Args:
            acc: Transformed accuracy
            baseline: Random baseline
            pvalue: P-value from binomial test
            cohens_h: Cohen's h effect size
            beats_random: Whether performance beats random
            is_meaningful: Whether effect size is meaningful
            transform_name: Name of transformation

        Returns:
            Interpretation string
        """
        baseline_pct = baseline * 100
        acc_pct = acc * 100

        if beats_random and is_meaningful:
            return (
                f"Model performs significantly above random chance on {transform_name} questions "
                f"({acc_pct:.1f}% vs {baseline_pct:.1f}% baseline, p={pvalue:.4f}, h={cohens_h:.3f}). "
                f"Questions can be answered without context using answer choices alone."
            )
        elif beats_random and not is_meaningful:
            return (
                f"Model performs statistically above random (p={pvalue:.4f}) "
                f"but with small effect size (h={cohens_h:.3f}). "
                f"Some signal available in answer choices, but context likely still helpful."
            )
        elif not beats_random and acc_pct > baseline_pct:
            return (
                f"Model performs at {acc_pct:.1f}% (random baseline: {baseline_pct:.1f}%), "
                f"but not significantly better (p={pvalue:.4f}). "
                f"Questions likely require context - no reliable signal from answer choices alone."
            )
        else:
            return (
                f"Model performs at or below random chance ({acc_pct:.1f}% vs {baseline_pct:.1f}% baseline). "
                f"Questions clearly require context to be answered correctly."
            )
