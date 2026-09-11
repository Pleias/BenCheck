"""Grammar and clarity diagnostics for multiple-choice options.

This check flags grammar and punctuation issues in MCQ answer options using
an external detector/model. If no detector is provided, options pass through
unchanged and no issues are flagged.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..base import BencheckCheck
from ..types import BencheckQuestion
from ._utils import normalize_correct_index

logger = logging.getLogger(__name__)


@dataclass
class OptionEdits:
    """Holds edit information for a single option."""

    original_text: str
    normalized_text: str
    suggested_text: str
    token_edits: list[dict[str, Any]]
    transformations: list[dict[str, Any]]

    @property
    def edit_count(self) -> int:
        return len(self.token_edits) + len(self.transformations)

    @property
    def flagged(self) -> bool:
        return self.edit_count > 0


class GrammarQualityCheck(BencheckCheck):
    """Detects grammar/spelling issues in MCQ answer options.

    The detector (external or fallback) produces a corrected string: either via
    an injected GEC model (e.g., GECToR) or via optional rule-based corrections.
    It then reports token-level edit operations using ``difflib.SequenceMatcher``.

    Per-question diagnostics include flagged options and suggested corrections.
    Aggregated metrics include:
    - ``Q_any_issue``: Questions with any flagged option
    - ``Q_gold_issue``: Questions with flagged correct answer
    - ``Q_helpful``: Gold clean, distractors flagged (good)
    - ``Q_harmful``: Gold flagged, distractors clean (bad)
    - ``distractor_issue_rate``: Rate of flagged distractors
    - ``ambiguity_histogram``: Distribution of edit counts
    - ``option_position_flagged_rate``: Flagged rate by position

    Args:
        collapse_whitespace: Whether to collapse whitespace (default: True)
        detector: External detector callable (e.g., GECToR model)
        model: Model with .predict() method (alternative to detector)
        enable_rule_based_fallback: **DEPRECATED** - Not recommended for production.
            Apply rule-based corrections when no detector is available (default: False).
            Rules include: capitalize "i" → "I", capitalize first letter, collapse
            repeated punctuation. For production use, always provide a detector/model.
    """

    name = "grammar_quality"

    def __init__(
        self,
        collapse_whitespace: bool = True,
        detector: Callable[[str], Any] | None = None,
        model: Any | None = None,
        enable_rule_based_fallback: bool = False,
        checkpoint_interval: int = 0,
        output_dir: str | None = None,
    ) -> None:
        self.collapse_whitespace = collapse_whitespace
        self.enable_rule_based_fallback = enable_rule_based_fallback
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        # Allow detector to be provided explicitly or via a model wrapper
        if detector is not None and model is not None:
            raise ValueError("Provide either detector or model, not both")

        self._batch_detector = None
        if detector is not None:
            self.detector = detector
        elif model is not None:
            # Prefer explicit predict method, otherwise treat model as callable
            self.detector = getattr(model, "predict", model)
            # If the model exposes a batch helper, keep a reference to reuse it
            self._batch_detector = getattr(model, "run_detector_on_options", None)
        else:
            self.detector = None

    def _apply_text_transforms(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Coerce input to string and record any coercion."""

        updated = text
        transforms: list[dict[str, str]] = []

        if not isinstance(updated, str):
            try:
                updated = "" if updated is None else str(updated)
            except Exception as exc:  # Defensive: avoid breaking on bad __str__
                logger.debug(
                    "grammar_quality: failed to coerce option to string; using empty string fallback",
                    exc_info=exc,
                )
                updated = ""
            transforms.append(
                {
                    "description": "coerced_non_string_option",
                    "before": repr(text),
                    "after": updated,
                }
            )

        return updated, transforms

    def _run_detector(self, text: str) -> dict[str, Any] | None:
        """
        Run external detector (e.g., GECToR) if provided.

        Expected return formats:
            - str: corrected text
            - dict: should include `corrected` or `corrected_text`, and optionally
              `token_edits` / `transformations`
        """
        if self.detector is None:
            return None

        try:
            output = self.detector(text)
        except Exception as exc:
            logger.debug(
                "grammar_quality: detector call failed; falling back to rule-based edits",
                exc_info=exc,
            )
            return None

        if isinstance(output, str):
            return {
                "suggested_text": output,
                "token_edits": [],
                "transformations": [{"description": "external_detector"}],
            }

        if isinstance(output, dict):
            corrected = output.get("corrected") or output.get("corrected_text")
            if not corrected:
                return None

            return {
                "suggested_text": corrected,
                "token_edits": output.get("token_edits") or output.get("edits") or [],
                "transformations": output.get("transformations") or [],
            }

        return None

    def _detect_token_edits(self, original: str, corrected: str) -> list[dict[str, Any]]:
        """Compute token-level edit operations using SequenceMatcher."""

        original_tokens = original.split()
        corrected_tokens = corrected.split()
        matcher = difflib.SequenceMatcher(a=original_tokens, b=corrected_tokens)

        token_edits: list[dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            token_edits.append(
                {
                    "op": tag,
                    "source_tokens": original_tokens[i1:i2],
                    "target_tokens": corrected_tokens[j1:j2],
                    "source_span": [i1, i2],
                    "target_span": [j1, j2],
                }
            )

        return token_edits

    def _apply_rule_based_transforms(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Apply rule-based grammar corrections (optional fallback).

        This method implements simple rule-based corrections:
        1. Capitalize standalone "i" → "I"
        2. Capitalize first letter of the option
        3. Collapse repeated punctuation (!!!, ???, etc.)

        Only used when enable_rule_based_fallback=True and no external detector.
        """
        tokens = text.split(" ") if text else []
        token_transforms: list[dict[str, Any]] = []
        corrected_tokens: list[str] = []

        for idx, token in enumerate(tokens):
            corrected = token
            reasons: list[str] = []

            # Rule 1: "i" → "I"
            if token == "i":
                corrected = "I"
                reasons.append("capitalize_pronoun")

            # Rule 2: Capitalize first letter of option
            if idx == 0:
                match = re.match(r"([^A-Za-z]*)([a-z])(.*)", corrected)
                if match:
                    prefix, first_char, rest = match.groups()
                    corrected = f"{prefix}{first_char.upper()}{rest}"
                    reasons.append("capitalized_start")

            # Rule 3: Collapse repeated punctuation
            collapsed_punct = re.sub(r"([!?.,;:])\1+", r"\1", corrected)
            if collapsed_punct != corrected:
                corrected = collapsed_punct
                reasons.append("collapsed_repeated_punctuation")

            if corrected != token:
                token_transforms.append(
                    {
                        "token_index": idx,
                        "original": token,
                        "suggestion": corrected,
                        "reasons": reasons,
                    }
                )

            corrected_tokens.append(corrected)

        corrected_text = " ".join(corrected_tokens)
        return corrected_text, token_transforms

    def _analyze_option(
        self,
        text: str,
        normalized_text: str | None = None,
        base_transforms: list[dict[str, Any]] | None = None,
        detector_output: dict[str, Any] | None = None,
    ) -> OptionEdits:
        normalized_text, transforms = (
            (normalized_text, list(base_transforms or []))
            if normalized_text is not None
            else self._apply_text_transforms(text)
        )
        corrected_text = normalized_text
        token_edits: list[dict[str, Any]] = []
        all_transforms = list(transforms)

        detector_output = detector_output or self._run_detector(normalized_text)
        if detector_output:
            corrected_text = detector_output["suggested_text"]
            token_edits = detector_output.get("token_edits") or []
            detector_transforms = detector_output.get("transformations") or []

            # Only count detector-provided transforms when an actual change exists
            changed = corrected_text != normalized_text
            if not token_edits:
                token_edits = self._detect_token_edits(normalized_text, corrected_text)
                changed = changed or bool(token_edits)

            if changed and detector_transforms:
                all_transforms.extend(detector_transforms)

        elif self.enable_rule_based_fallback:
            # Fallback to rule-based corrections when no detector is available
            corrected_text, token_transforms = self._apply_rule_based_transforms(normalized_text)
            all_transforms.extend(token_transforms)
            token_edits = self._detect_token_edits(normalized_text, corrected_text)

        return OptionEdits(
            original_text=text,
            normalized_text=normalized_text,
            suggested_text=corrected_text,
            token_edits=token_edits,
            transformations=all_transforms,
        )

    def run_on_question(
        self, question: BencheckQuestion, model: Any | None = None
    ) -> dict[str, Any]:
        # Analyze question text grammar
        question_text = question.question or ""
        question_analysis = self._analyze_option(question_text)
        question_has_grammar_issue = question_analysis.flagged

        gold_index = normalize_correct_index(question.correct)
        gold_in_range = gold_index is not None and 0 <= gold_index < len(question.choices)

        options: list[dict[str, Any]] = []
        flagged_distractors = 0

        normalized_inputs: list[tuple[str, list[dict[str, Any]]]] = [
            self._apply_text_transforms(choice) for choice in question.choices
        ]

        batch_detector_outputs: list[dict[str, Any]] | None = None
        batch_detector = getattr(self, "_batch_detector", None)
        if batch_detector:
            try:
                batch_detector_outputs = batch_detector(
                    [normalized for normalized, _ in normalized_inputs]
                )
            except Exception as exc:
                logger.debug(
                    "grammar_quality: batch detector failed; falling back to per-option processing",
                    exc_info=exc,
                )
                batch_detector_outputs = None

        for idx, choice in enumerate(question.choices):
            detector_output = None
            if batch_detector_outputs and idx < len(batch_detector_outputs):
                detector_output = batch_detector_outputs[idx]

            normalized_text, transforms = normalized_inputs[idx]
            analysis = self._analyze_option(
                choice,
                normalized_text=normalized_text,
                base_transforms=transforms,
                detector_output=detector_output,
            )
            option_dict = {
                "original": analysis.original_text,
                "normalized": analysis.normalized_text,
                "suggested": analysis.suggested_text,
                "flagged": analysis.flagged,
                "edit_count": analysis.edit_count,
                "token_edits": analysis.token_edits,
                "transformations": analysis.transformations,
            }
            options.append(option_dict)

            if analysis.flagged and gold_in_range and idx != gold_index:
                flagged_distractors += 1

        gold_has_issue: bool | None = None
        if gold_in_range:
            gold_has_issue = options[gold_index]["flagged"]

        flagged_option_count = sum(1 for opt in options if opt["flagged"])
        has_issue = flagged_option_count > 0

        # Check if any distractor has grammar issues
        has_distractor_issue = flagged_distractors > 0 if gold_in_range else None

        question_summary: dict[str, Any] = {
            "question_analysis": {
                "original": question_analysis.original_text,
                "normalized": question_analysis.normalized_text,
                "suggested": question_analysis.suggested_text,
                "flagged": question_analysis.flagged,
                "edit_count": question_analysis.edit_count,
                "token_edits": question_analysis.token_edits,
                "transformations": question_analysis.transformations,
            },
            "question_has_grammar_issue": question_has_grammar_issue,
            "has_distractor_issue": has_distractor_issue,
            "options": options,
            "has_issue": has_issue,
            "flagged_option_count": flagged_option_count,
            "total_options": len(options),
            "gold_index": gold_index if gold_in_range else None,
            "gold_has_issue": gold_has_issue,
            "flagged_distractor_count": flagged_distractors,
            "distractor_count": (len(options) - 1) if gold_in_range and len(options) > 0 else 0,
        }

        if not gold_in_range:
            if gold_index is None:
                question_summary["warning"] = "gold_index_missing"
            else:
                question_summary["warning"] = "gold_index_out_of_range"

        return question_summary

    def run(
        self,
        dataset: list["BencheckQuestion"],
        model: Any | None = None,
        seed: int | None = None,
    ) -> "CheckResult":
        """
        Execute check on entire dataset with optional checkpoint saving.

        If checkpoint_interval > 0, saves intermediate results every N questions.
        This allows recovery of partial results if processing is interrupted.

        Args:
            dataset: List of questions to check
            model: DEPRECATED - ignored, uses self.detector
            seed: Optional random seed (not used by grammar check)

        Returns:
            CheckResult with per-question diagnostics and aggregate metrics
        """
        import json
        from pathlib import Path

        from ..core.progress import ProgressLevel, progress_bar
        from ..types import CheckResult

        question_results = {}

        for i, q in enumerate(
            progress_bar(dataset, f"{self.name}", ProgressLevel.VERBOSE, unit="question")
        ):
            question_results[q.id] = self.run_on_question(q, model)

            # Save checkpoint if configured
            if (
                self.checkpoint_interval > 0
                and self.output_dir
                and (i + 1) % self.checkpoint_interval == 0
            ):
                checkpoint_path = Path(self.output_dir) / f"checkpoint_{i+1}.json"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                # Compute metrics on questions processed so far
                checkpoint_metrics = self.aggregate_metrics(question_results)

                checkpoint_data = {
                    "check_name": self.name,
                    "questions_processed": i + 1,
                    "total_questions": len(dataset),
                    "question_results": question_results,
                    "metrics": checkpoint_metrics,
                }

                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)

                logger.info(
                    f"💾 Checkpoint saved: {i+1}/{len(dataset)} questions → {checkpoint_path}"
                )

        # Compute final metrics
        metrics = self.aggregate_metrics(question_results)
        return CheckResult(
            check_name=self.name,
            question_results=question_results,
            metrics=metrics,
        )

    @staticmethod
    def _compute_grammar_impact_test(
        question_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Test whether grammar quality correlates with answer correctness.

        Uses Fisher's exact test (or Chi-square for large samples) to determine
        if there's a statistically significant relationship between grammar issues
        and whether an option is correct vs incorrect.

        Returns:
            Dict with test statistics, or None if insufficient data
        """
        # Build contingency table:
        #                | Grammar OK | Grammar Issue
        # ---------------+------------+--------------
        # Correct Answer |     a      |      b
        # Incorrect      |     c      |      d

        correct_clean = 0  # a
        correct_flagged = 0  # b
        incorrect_clean = 0  # c
        incorrect_flagged = 0  # d

        for res in question_results.values():
            gold_index = res.get("gold_index")
            if gold_index is None:
                continue

            for idx, opt in enumerate(res.get("options", [])):
                is_correct = idx == gold_index
                has_issue = opt.get("flagged", False)

                if is_correct:
                    if has_issue:
                        correct_flagged += 1
                    else:
                        correct_clean += 1
                else:
                    if has_issue:
                        incorrect_flagged += 1
                    else:
                        incorrect_clean += 1

        # Need at least some data in each category
        total = correct_clean + correct_flagged + incorrect_clean + incorrect_flagged
        if total < 10:
            return None

        # Compute rates
        total_correct = correct_clean + correct_flagged
        total_incorrect = incorrect_clean + incorrect_flagged
        correct_issue_rate = correct_flagged / total_correct if total_correct > 0 else 0.0
        incorrect_issue_rate = incorrect_flagged / total_incorrect if total_incorrect > 0 else 0.0

        result = {
            "contingency_table": {
                "correct_clean": correct_clean,
                "correct_flagged": correct_flagged,
                "incorrect_clean": incorrect_clean,
                "incorrect_flagged": incorrect_flagged,
            },
            "correct_issue_rate": round(correct_issue_rate, 4),
            "incorrect_issue_rate": round(incorrect_issue_rate, 4),
            "interpretation": (
                "lower_is_better"
                if correct_issue_rate < incorrect_issue_rate
                else "higher_suggests_bias"
            ),
        }

        # Try Fisher's exact test (requires scipy)
        try:
            from scipy.stats import fisher_exact

            table = [[correct_clean, correct_flagged], [incorrect_clean, incorrect_flagged]]
            odds_ratio, p_value = fisher_exact(table)
            result["fisher_exact_p_value"] = round(float(p_value), 6)
            result["odds_ratio"] = round(float(odds_ratio), 4)
            result["statistical_test"] = "fisher_exact"

            if p_value < 0.001:
                result["significance"] = "very_strong"
            elif p_value < 0.01:
                result["significance"] = "strong"
            elif p_value < 0.05:
                result["significance"] = "significant"
            else:
                result["significance"] = "not_significant"

        except ImportError:
            logger.debug("scipy not available for Fisher's exact test")
            result["note"] = "Install scipy for statistical testing"

        return result

    def aggregate_metrics(self, question_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        total_questions = len(question_results)
        if total_questions == 0:
            return {
                "n_questions": 0,
                "error": "no_questions",
            }

        q_any_issue = 0
        q_gold_issue = 0
        q_prompt_issue = 0  # Questions with grammar issues in question text (prompt)
        q_distractor_issue = 0  # Questions with at least one distractor having grammar issues
        q_helpful = 0  # Gold clean, distractors flagged
        q_harmful = 0  # Gold flagged, distractors clean
        gold_evaluated = 0
        flagged_options_total = 0
        total_options = 0
        flagged_distractors_total = 0
        total_distractors = 0
        flagged_by_position: Counter[int] = Counter()
        total_by_position: Counter[int] = Counter()
        ambiguity_histogram: Counter[int] = Counter()

        for res in question_results.values():
            if res.get("has_issue"):
                q_any_issue += 1

            # Count questions with grammar issues in prompt
            if res.get("question_has_grammar_issue"):
                q_prompt_issue += 1

            # Count questions with at least one distractor having issues
            if res.get("has_distractor_issue"):
                q_distractor_issue += 1

            flagged_options_total += res.get("flagged_option_count", 0)
            total_options += res.get("total_options", 0)

            gold_has_issue = res.get("gold_has_issue")
            if gold_has_issue is not None:
                gold_evaluated += 1
                if gold_has_issue:
                    q_gold_issue += 1

            flagged_distractors_total += res.get("flagged_distractor_count", 0)
            total_distractors += res.get("distractor_count", 0)

            # Count helpful/harmful patterns
            gold_in_range = res.get("gold_index") is not None
            if gold_in_range:
                flagged_distractors = res.get("flagged_distractor_count", 0)
                if gold_has_issue is False and flagged_distractors > 0:
                    q_helpful += 1
                if gold_has_issue is True and flagged_distractors == 0:
                    q_harmful += 1

            # Build ambiguity histogram (edit count distribution)
            for opt in res.get("options", []):
                ambiguity_histogram[opt.get("edit_count", 0)] += 1

            for idx, opt in enumerate(res.get("options", [])):
                total_by_position[idx] += 1
                if opt.get("flagged"):
                    flagged_by_position[idx] += 1

        option_position_flagged_rate = {
            pos: {
                "flagged": flagged_by_position[pos],
                "total": total_by_position[pos],
                "rate": round(flagged_by_position[pos] / total_by_position[pos], 4)
                if total_by_position[pos]
                else 0.0,
            }
            for pos in sorted(total_by_position)
        }

        metrics = {
            "n_questions": total_questions,
            "Q_any_issue": {
                "count": q_any_issue,
                "rate": round(q_any_issue / total_questions, 4),
                "description": "Questions with any grammar issue (in prompt or options)",
            },
            "Q_prompt_issue": {
                "count": q_prompt_issue,
                "rate": round(q_prompt_issue / total_questions, 4),
                "description": "Questions with grammar issues in question text (prompt)",
            },
            "Q_distractor_issue": {
                "count": q_distractor_issue,
                "rate": round(q_distractor_issue / total_questions, 4),
                "description": "Questions with at least one distractor having grammar issues",
            },
            "Q_helpful": {
                "count": q_helpful,
                "rate": round(q_helpful / total_questions, 4),
                "description": "Gold answer clean, distractors flagged (might help discrimination)",
            },
            "Q_harmful": {
                "count": q_harmful,
                "rate": round(q_harmful / total_questions, 4),
                "description": "Gold answer flagged, distractors clean (discriminates against correct answer)",
            },
            "flagged_option_rate": {
                "flagged": flagged_options_total,
                "total": total_options,
                "rate": round(flagged_options_total / total_options, 4) if total_options else 0.0,
                "description": "Overall rate of options with grammar issues",
            },
            "distractor_issue_rate": {
                "flagged": flagged_distractors_total,
                "total": total_distractors,
                "rate": round(flagged_distractors_total / total_distractors, 4)
                if total_distractors
                else 0.0,
                "description": "Rate of distractors (incorrect options) with grammar issues",
            },
            "ambiguity_histogram": dict(sorted(ambiguity_histogram.items())),
            "option_position_flagged_rate": option_position_flagged_rate,
        }

        if gold_evaluated > 0:
            metrics["Q_gold_issue"] = {
                "count": q_gold_issue,
                "evaluated": gold_evaluated,
                "rate": round(q_gold_issue / gold_evaluated, 4),
            }
        else:
            metrics["Q_gold_issue"] = {
                "count": 0,
                "evaluated": 0,
                "rate": 0.0,
                "note": "gold labels missing",
            }

        # Statistical test: Does grammar correlate with correctness?
        # Hypothesis: Correct answers have better grammar than incorrect ones
        grammar_stats = self._compute_grammar_impact_test(question_results)
        if grammar_stats:
            metrics["grammar_impact_test"] = grammar_stats

        return metrics
