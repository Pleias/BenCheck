"""
Length Bias Check - analyzes correlation between answer length and selection.

Based on findings from research on HellaSwag and other benchmarks that models may
exhibit length bias, preferring longer or shorter answers regardless of correctness.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from ..base import BencheckCheck
from ..types import BencheckQuestion, QuestionType
from ._utils import normalize_correct_index

# Optional numpy dependency for statistics
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional scipy dependency for statistical tests
try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

_DEFAULT_TOKENIZER = "google/gemma-3-12b-it"
_tokenizer_cache: dict = {}


def _get_tokenizer(name_or_path: str):
    if name_or_path in _tokenizer_cache:
        return _tokenizer_cache[name_or_path]
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name_or_path)
        _tokenizer_cache[name_or_path] = tok
        return tok
    except Exception as e:
        logger.warning("length_bias: could not load tokenizer %r (%s); falling back to words", name_or_path, e)
        _tokenizer_cache[name_or_path] = None
        return None


def _length_bytes(text: str) -> int:
    """Length in bytes (UTF-8), as used in the paper."""
    return len(text.encode("utf-8"))


def _length_words(text: str) -> int:
    """Length in words (simple space split)."""
    return len(text.split())


class LengthBiasCheck(BencheckCheck):
    """
    Check for length bias in multiple-choice questions.

    This check analyzes whether:
    1. Correct answers tend to be longer/shorter than incorrect ones
    2. Answer length variability is high (relative length difference)
    3. The dataset exhibits patterns that could lead to length-based shortcuts

    This is a model-free check - it analyzes dataset properties without
    requiring model predictions.

    Reference:
        Based on analysis from "HellaSwag or HellaBad? 36% of this popular
        LLM benchmark contains errors" (ACL 2026 submission)
    """

    name = "length_bias"

    def __init__(
        self,
        length_metric: str = "tokens",
        tokenizer_name_or_path: str = _DEFAULT_TOKENIZER,
    ):
        """
        Initialize the length bias check.

        Args:
            length_metric: How to measure length. Options:
                - "tokens": Subword token count via HuggingFace tokenizer (default)
                - "words": Whitespace-separated word count
                - "bytes": UTF-8 byte count (more precise for multilingual text)
            tokenizer_name_or_path: HuggingFace model name or local path used when
                length_metric="tokens". Falls back to word count if the tokenizer
                cannot be loaded.
        """
        if length_metric not in ("tokens", "words", "bytes"):
            raise ValueError(f"length_metric must be 'tokens', 'words', or 'bytes', got {length_metric}")
        self.length_metric = length_metric
        self.tokenizer_name_or_path = tokenizer_name_or_path

        if length_metric == "tokens":
            tok = _get_tokenizer(tokenizer_name_or_path)
            if tok is not None:
                self._length_fn = lambda text: len(tok.encode(text))
            else:
                logger.warning("length_bias: falling back to word count")
                # Report the metric actually used, so results are not labelled "tokens"
                self.length_metric = "words"
                self._length_fn = _length_words
        elif length_metric == "words":
            self._length_fn = _length_words
        else:
            self._length_fn = _length_bytes

    def run_on_question(
        self, question: BencheckQuestion, model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze length distribution for a single question.

        Returns:
            Dict containing:
                - option_lengths: List of lengths for each option
                - correct_length: Length of the correct answer
                - incorrect_lengths: Lengths of incorrect answers
                - longest_is_correct: Whether the longest option is correct
                - shortest_is_correct: Whether the shortest option is correct
                - relative_length_diff: (max_len - min_len) / max_len
                - length_rank_of_correct: Rank of correct answer by length (0=shortest)
        """
        # Warn if this is a MULTIPLE_CHOICE question (designed for single-choice)
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            logger.warning(
                f"LengthBiasCheck: Question {question.id} is MULTIPLE_CHOICE but this check "
                f"assumes single-choice questions. Using first correct answer only."
            )

        lengths = [self._length_fn(opt) for opt in question.choices]
        correct_idx = normalize_correct_index(question.correct)

        if correct_idx is None or correct_idx >= len(lengths):
            # Invalid correct index - skip analysis
            logger.warning(
                f"LengthBiasCheck: Invalid correct index {correct_idx} for question {question.id}. "
                f"Expected index < {len(lengths)}. Skipping analysis."
            )
            return {"option_lengths": lengths, "error": "invalid_correct_index"}

        correct_length = lengths[correct_idx]
        incorrect_lengths = [l for i, l in enumerate(lengths) if i != correct_idx]

        max_len = max(lengths)
        min_len = min(lengths)
        relative_diff = (max_len - min_len) / max_len if max_len > 0 else 0.0

        # Compute rank of correct answer by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        length_rank = sorted_indices.index(correct_idx)

        # Strictly compare against distractors so the two flags are mutually exclusive.
        # Using == max_len / == min_len causes double-counting when options tie in length
        # (common in 2-option benchmarks like PIQA and WinoGrande).
        max_incorrect = max(incorrect_lengths) if incorrect_lengths else correct_length
        min_incorrect = min(incorrect_lengths) if incorrect_lengths else correct_length

        return {
            "option_lengths": lengths,
            "correct_length": correct_length,
            "incorrect_lengths": incorrect_lengths,
            "mean_incorrect_length": (
                float(np.mean(incorrect_lengths))
                if HAS_NUMPY
                else (sum(incorrect_lengths) / len(incorrect_lengths))
            )
            if incorrect_lengths
            else 0.0,
            "longest_is_correct": correct_length > max_incorrect,
            "shortest_is_correct": correct_length < min_incorrect,
            "relative_length_diff": round(relative_diff, 4),
            "length_rank_of_correct": length_rank,
            "max_length": max_len,
            "min_length": min_len,
        }

    def aggregate_metrics(self, question_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute dataset-level length bias metrics with statistical tests.

        Returns:
            Dict containing:
                - n_questions: Number of questions analyzed
                - median_relative_length_diff: Median of relative length differences
                - mean_correct_length: Average length of correct answers
                - mean_incorrect_length: Average length of incorrect answers
                - correct_is_longest_pct: % of questions where correct is longest
                - correct_is_shortest_pct: % of questions where correct is shortest
                - length_rank_distribution: Distribution of correct answer ranks by length

                Statistical tests (if scipy available):
                - mannwhitneyu_p_value: P-value testing if correct/incorrect lengths differ
                - cohen_d: Effect size of length difference
                - rank_chi2_p_value: P-value testing if rank distribution is uniform
        """
        # Filter out questions with errors
        valid_results = {qid: res for qid, res in question_results.items() if "error" not in res}

        n_questions = len(valid_results)
        if n_questions == 0:
            return {"n_questions": 0, "error": "no_valid_questions"}

        # Collect statistics
        relative_diffs = [res["relative_length_diff"] for res in valid_results.values()]
        correct_lengths = [res["correct_length"] for res in valid_results.values()]
        incorrect_lengths = []
        for res in valid_results.values():
            incorrect_lengths.extend(res["incorrect_lengths"])

        longest_correct_count = sum(
            1 for res in valid_results.values() if res["longest_is_correct"]
        )
        shortest_correct_count = sum(
            1 for res in valid_results.values() if res["shortest_is_correct"]
        )

        # Length rank distribution (how often is correct answer the shortest, 2nd shortest, etc.)
        from collections import Counter

        rank_counts = Counter(res["length_rank_of_correct"] for res in valid_results.values())
        max_rank = max(rank_counts.keys()) if rank_counts else 0
        rank_distribution = [rank_counts.get(i, 0) for i in range(max_rank + 1)]

        # Calculate statistics (with numpy fallback)
        if HAS_NUMPY:
            median_rel_diff = round(float(np.median(relative_diffs)), 4)
            mean_rel_diff = round(float(np.mean(relative_diffs)), 4)
            mean_correct = round(float(np.mean(correct_lengths)), 2)
            mean_incorrect = (
                round(float(np.mean(incorrect_lengths)), 2) if incorrect_lengths else 0.0
            )
            std_correct = round(float(np.std(correct_lengths)), 2)
            std_incorrect = round(float(np.std(incorrect_lengths)), 2) if incorrect_lengths else 0.0
        else:
            # Pure Python fallback
            sorted_diffs = sorted(relative_diffs)
            n = len(sorted_diffs)
            median_rel_diff = round(
                sorted_diffs[n // 2]
                if n % 2
                else (sorted_diffs[n // 2 - 1] + sorted_diffs[n // 2]) / 2,
                4,
            )
            mean_rel_diff = round(sum(relative_diffs) / len(relative_diffs), 4)
            mean_correct = round(sum(correct_lengths) / len(correct_lengths), 2)
            mean_incorrect = (
                round(sum(incorrect_lengths) / len(incorrect_lengths), 2)
                if incorrect_lengths
                else 0.0
            )
            # Simple std calculation
            var_correct = sum((x - mean_correct) ** 2 for x in correct_lengths) / len(
                correct_lengths
            )
            std_correct = round(var_correct**0.5, 2)
            if incorrect_lengths:
                var_incorrect = sum((x - mean_incorrect) ** 2 for x in incorrect_lengths) / len(
                    incorrect_lengths
                )
                std_incorrect = round(var_incorrect**0.5, 2)
            else:
                std_incorrect = 0.0

        metrics = {
            "n_questions": n_questions,
            "length_metric": self.length_metric,
            "median_relative_length_diff": median_rel_diff,
            "mean_relative_length_diff": mean_rel_diff,
            "mean_correct_length": mean_correct,
            "mean_incorrect_length": mean_incorrect,
            "std_correct_length": std_correct,
            "std_incorrect_length": std_incorrect,
            "correct_is_longest_pct": round((longest_correct_count / n_questions) * 100, 2),
            "correct_is_shortest_pct": round((shortest_correct_count / n_questions) * 100, 2),
            "length_rank_distribution": rank_distribution,
        }

        # Add statistical tests if scipy is available
        if HAS_SCIPY and len(correct_lengths) > 0 and len(incorrect_lengths) > 0:
            # Mann-Whitney U test: Tests if correct and incorrect lengths come from the same distribution
            # H0: Distributions are the same
            # p < 0.05: Reject H0, lengths significantly different (indicates bias)
            try:
                statistic, p_value = scipy_stats.mannwhitneyu(
                    correct_lengths, incorrect_lengths, alternative="two-sided"
                )
                metrics["mannwhitneyu_statistic"] = round(float(statistic), 2)
                metrics["mannwhitneyu_p_value"] = round(float(p_value), 4)
            except Exception as e:
                logger.warning(f"Mann-Whitney U test failed: {e}")
                metrics["mannwhitneyu_p_value"] = None

            # Cohen's d effect size: Measures magnitude of difference
            # |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, > 0.8: large
            try:
                pooled_std = ((std_correct**2 + std_incorrect**2) / 2) ** 0.5
                if pooled_std > 0:
                    cohen_d = (mean_correct - mean_incorrect) / pooled_std
                    metrics["cohen_d"] = round(float(cohen_d), 4)

                    # Interpret effect size
                    abs_d = abs(cohen_d)
                    if abs_d < 0.2:
                        effect_interpretation = "negligible"
                    elif abs_d < 0.5:
                        effect_interpretation = "small"
                    elif abs_d < 0.8:
                        effect_interpretation = "medium"
                    else:
                        effect_interpretation = "large"
                    metrics["effect_size_interpretation"] = effect_interpretation
                else:
                    metrics["cohen_d"] = None
            except Exception as e:
                logger.warning(f"Cohen's d calculation failed: {e}")
                metrics["cohen_d"] = None

            # Chi-square goodness of fit test for rank distribution
            # H0: Ranks are uniformly distributed
            # p < 0.05: Reject H0, ranks not uniform (indicates positional bias)
            try:
                if len(rank_distribution) > 1:
                    expected_freq = n_questions / len(rank_distribution)
                    chi2_stat, chi2_p = scipy_stats.chisquare(
                        rank_distribution, [expected_freq] * len(rank_distribution)
                    )
                    metrics["rank_chi2_statistic"] = round(float(chi2_stat), 2)
                    metrics["rank_chi2_p_value"] = round(float(chi2_p), 4)
            except Exception as e:
                logger.warning(f"Chi-square test for rank distribution failed: {e}")
                metrics["rank_chi2_p_value"] = None

        return metrics
