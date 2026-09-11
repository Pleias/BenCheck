"""LLM-as-judge grammar check using vLLM guided JSON decoding."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..types import BencheckQuestion, CheckResult
from ._utils import normalize_correct_index
from .grammar_quality import GrammarQualityCheck

logger = logging.getLogger(__name__)

_LABELS = "ABCDEFGHIJ"

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "question_grammar_issue": {"type": "boolean"},
        "correct_answer_grammar_issue": {"type": "boolean"},
        "incorrect_answers_grammar_issue": {"type": "boolean"},
    },
    "required": [
        "question_grammar_issue",
        "correct_answer_grammar_issue",
        "incorrect_answers_grammar_issue",
    ],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You are a grammar checker for multiple-choice benchmark questions. \
Identify GENUINE grammatical errors only.

Flag an error if there is a clear grammatical mistake, such as:
- Wrong word form (e.g. "it's" instead of "its", "their" instead of "there")
- Subject-verb disagreement
- Missing or extra article where grammatically required
- Broken syntax that makes the sentence hard to parse
- Incorrect tense or number agreement

Do NOT flag:
- Informal or colloquial phrasing that is grammatically valid
- British vs. American spelling differences
- Answer choices that are sentence fragments (common in MCQs)
- Capitalisation inconsistencies or style preferences\
"""


def _build_prompt(question: BencheckQuestion) -> str:
    gold_index = normalize_correct_index(question.correct)
    lines = [f"Question: {question.question}", "", "Options:"]
    for i, choice in enumerate(question.choices):
        lines.append(f"  {_LABELS[i]}) {choice}")

    if gold_index is not None and 0 <= gold_index < len(question.choices):
        lines.append(
            f"\nCorrect answer: {_LABELS[gold_index]}) {question.choices[gold_index]}"
        )
    else:
        lines.append("\nCorrect answer: unknown")

    return "\n".join(lines)


def _make_result(
    question: BencheckQuestion,
    flags: dict[str, bool],
    raw_response: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    gold_index = normalize_correct_index(question.correct)
    gold_in_range = gold_index is not None and 0 <= gold_index < len(question.choices)

    q_issue = flags.get("question_grammar_issue", False)
    gold_issue: bool | None = (
        flags.get("correct_answer_grammar_issue", False) if gold_in_range else None
    )
    distractor_issue: bool | None = (
        flags.get("incorrect_answers_grammar_issue", False) if gold_in_range else None
    )

    # Reconstruct a synthetic per-option list so aggregate_metrics works unchanged.
    # All distractors share the single distractor flag (LLM gives one flag for all wrong answers).
    options: list[dict[str, Any]] = []
    for i, choice in enumerate(question.choices):
        if gold_in_range and i == gold_index:
            flagged = bool(gold_issue)
        else:
            flagged = bool(distractor_issue) if distractor_issue is not None else False
        options.append(
            {
                "original": choice,
                "normalized": choice,
                "suggested": choice,
                "flagged": flagged,
                "edit_count": 1 if flagged else 0,
                "token_edits": [],
                "transformations": [],
            }
        )

    flagged_distractors = sum(
        1
        for i, opt in enumerate(options)
        if opt["flagged"] and gold_in_range and i != gold_index
    )
    flagged_count = sum(1 for opt in options if opt["flagged"])
    has_issue = q_issue or flagged_count > 0

    result: dict[str, Any] = {
        "question_has_grammar_issue": q_issue,
        "has_distractor_issue": distractor_issue,
        "options": options,
        "has_issue": has_issue,
        "flagged_option_count": flagged_count,
        "total_options": len(options),
        "gold_index": gold_index if gold_in_range else None,
        "gold_has_issue": gold_issue,
        "flagged_distractor_count": flagged_distractors,
        "distractor_count": (len(options) - 1) if gold_in_range and len(options) > 0 else 0,
        "_raw_response": raw_response,
    }

    if error:
        result["_error"] = error
    if not gold_in_range:
        result["warning"] = (
            "gold_index_missing" if gold_index is None else "gold_index_out_of_range"
        )

    return result


class LLMGrammarCheck(GrammarQualityCheck):
    """Grammar check using an LLM judge with guided JSON decoding.

    Presents the full question + all answer options to a generative model and asks
    for three boolean flags: grammar error in the question stem, in the correct
    answer, and in any of the incorrect options.

    Uses vLLM's guided JSON decoding to guarantee parseable output.
    The full dataset is sent to vLLM in a single batched call.

    Args:
        model: VLLMModel instance (must expose .llm for direct batch inference).
        system_prompt: Instruction prefix. Override to experiment with flagging criteria.
        max_tokens: Token budget for the JSON response (80 is sufficient).
    """

    name = "llm_grammar"

    def __init__(
        self,
        model: Any,
        system_prompt: str = _SYSTEM_PROMPT,
        max_tokens: int = 200,
        checkpoint_interval: int = 0,
        output_dir: str | None = None,
    ) -> None:
        super().__init__(
            detector=None,
            checkpoint_interval=checkpoint_interval,
            output_dir=output_dir,
        )
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def _apply_chat_template(self, user_messages: list[str]) -> list[str]:
        tokenizer = self.model.llm.get_tokenizer()
        formatted = []
        for msg in user_messages:
            chat = [{"role": "user", "content": msg}]
            formatted.append(
                tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted

    def _make_sampling_params(self):
        from vllm import SamplingParams

        schema_str = json.dumps(_JSON_SCHEMA)

        # vLLM >= 0.10: StructuredOutputsParams(json=<str>) + structured_outputs=
        try:
            from vllm.sampling_params import StructuredOutputsParams

            return SamplingParams(
                temperature=0.0,
                max_tokens=self.max_tokens,
                structured_outputs=StructuredOutputsParams(json=schema_str),
            )
        except (ImportError, TypeError):
            pass

        # vLLM 0.4–0.9: GuidedDecodingParams
        try:
            from vllm.sampling_params import GuidedDecodingParams

            return SamplingParams(
                temperature=0.0,
                max_tokens=self.max_tokens,
                guided_decoding=GuidedDecodingParams(json=_JSON_SCHEMA),
            )
        except (ImportError, TypeError):
            pass

        # Legacy vLLM: guided_json kwarg
        return SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            guided_json=_JSON_SCHEMA,
        )

    def _call_batch(self, user_messages: list[str]) -> list[str]:
        prompts = self._apply_chat_template(user_messages)
        params = self._make_sampling_params()
        outputs = self.model.llm.generate(prompts, params)
        return [out.outputs[0].text for out in outputs]

    def _parse_flags(self, response: str) -> tuple[dict[str, bool], str | None]:
        try:
            data = json.loads(response)
            return {k: bool(v) for k, v in data.items()}, None
        except Exception:
            pass

        # Regex fallback: guided decoding can truncate JSON with whitespace padding
        result: dict[str, bool] = {}
        for field in _JSON_SCHEMA["required"]:
            m = re.search(rf'"{re.escape(field)}"\s*:\s*(true|false)', response)
            if m:
                result[field] = m.group(1) == "true"

        if not result:
            logger.warning("llm_grammar: unparseable response: %r", response[:100])
            return {}, "parse_failed"

        missing = [f for f in _JSON_SCHEMA["required"] if f not in result]
        if missing:
            logger.debug("llm_grammar: partial parse, defaulting %s to False", missing)
            for f in missing:
                result[f] = False

        return result, None

    def _user_message(self, question: BencheckQuestion) -> str:
        return (
            f"{self.system_prompt}\n\n"
            f"{_build_prompt(question)}\n\n"
            'Respond with a single compact JSON line, for example:\n'
            '{"question_grammar_issue": false, "correct_answer_grammar_issue": false, "incorrect_answers_grammar_issue": false}'
        )

    def run_on_question(
        self, question: BencheckQuestion, model: Any | None = None
    ) -> dict[str, Any]:
        responses = self._call_batch([self._user_message(question)])
        flags, error = self._parse_flags(responses[0])
        return _make_result(question, flags, responses[0], error)

    def run(
        self,
        dataset: list[BencheckQuestion],
        model: Any | None = None,
        seed: int | None = None,
    ) -> CheckResult:
        from ..core.progress import ProgressLevel, progress_bar

        messages = [self._user_message(q) for q in dataset]
        logger.info("llm_grammar: batch inference on %d questions", len(messages))
        raw_responses = self._call_batch(messages)

        question_results: dict[str, Any] = {}
        for q, response in zip(
            progress_bar(dataset, self.name, ProgressLevel.VERBOSE, unit="question"),
            raw_responses,
        ):
            flags, error = self._parse_flags(response)
            question_results[q.id] = _make_result(q, flags, response, error)

        metrics = self.aggregate_metrics(question_results)
        return CheckResult(
            check_name=self.name,
            question_results=question_results,
            metrics=metrics,
        )
