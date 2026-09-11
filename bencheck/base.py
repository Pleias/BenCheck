"""Core abstract base classes for BenCheck."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union

from .types import BencheckQuestion, CheckResult

# Flexible score output type
ScoreOutput = Union[float, List[float], Dict[str, Any]]


class ScoringModel(Protocol):
    """
    Model that can score text continuations using log-likelihood or similar metrics.

    This protocol defines the interface for models that evaluate how likely a piece
    of text is as a continuation of a given context. Different model types can
    implement this:

    - Encoder-only models (e.g., BERT): masked language modeling scoring
    - Decoder-only models (e.g., GPT): autoregressive next-token scoring
    - Encoder-decoder models (e.g., T5): conditional generation scoring

    The score can be a simple float (e.g., average log-likelihood), a list of
    per-token scores, or a structured dict with additional metadata.
    """

    def score_continuation(self, context: str, continuation: str) -> ScoreOutput:
        """
        Compute score (e.g., log-likelihood) for a single continuation given context.

        Args:
            context: The input context/prompt
            continuation: The text to score as continuation of the context

        Returns:
            ScoreOutput: Can be:
                - float: single aggregate score (e.g., mean log-likelihood)
                - List[float]: per-token scores
                - Dict[str, Any]: structured output with scores and metadata

        Example:
            >>> model.score_continuation("The capital of France is", "Paris")
            -2.4  # mean log-likelihood

            >>> model.score_continuation("What comes next?", "The answer")
            {"mean_logprob": -2.4, "tokens": 2, "perplexity": 11.02}
        """
        ...

    def score_continuations(self, context: str, continuations: List[str]) -> List[ScoreOutput]:
        """
        Score multiple continuations at once (batch processing for efficiency).

        This method allows scoring multiple candidate continuations in a single
        forward pass when possible, which can be more efficient than calling
        score_continuation repeatedly.

        Args:
            context: The input context/prompt (same for all continuations)
            continuations: List of candidate texts to score as continuations

        Returns:
            List[ScoreOutput]: List of scores, one per continuation. Each score
                has the same format as returned by score_continuation.

        Example:
            >>> model.score_continuations(
            ...     "The capital of France is",
            ...     ["Paris", "London", "Berlin", "Madrid"]
            ... )
            [-2.4, -5.1, -4.8, -5.3]

        Note:
            The default implementation loops over score_continuation, but
            subclasses can override for true batch processing.
        """
        return [self.score_continuation(context, cont) for cont in continuations]


class GenerativeModel(Protocol):
    """
    Model that can generate text completions from prompts.

    This protocol is for models that produce text output given an input prompt,
    typically used for open-ended generation tasks rather than scoring.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a text completion for the given prompt.

        Args:
            prompt: The input prompt to complete
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            The generated text completion

        Example:
            >>> model.generate("The capital of France is", max_tokens=10)
            "Paris, a beautiful city known for"
        """
        ...


class BencheckModel(ScoringModel, GenerativeModel, Protocol):
    """
    Model supporting both scoring and generation.

    This combines both ScoringModel and GenerativeModel protocols for models
    that can do both tasks (e.g., decoder-only models like GPT).

    Use this when your model needs to support both:
    - Scoring text continuations (for multiple-choice evaluation)
    - Generating open-ended text (for free-form tasks)
    """

    pass


class BencheckCheck(ABC):
    """
    Abstract base class for diagnostic checks.

    Checks can be model-dependent (requiring a ScoringModel or GenerativeModel)
    or model-free (analyzing dataset properties without model inference).

    Examples:
        - Model-dependent: NoneOfTheAboveCheck (scores options with/without correct answer)
        - Model-free: LengthBiasCheck (analyzes correlation between answer length and correctness)
    """

    name: str

    @abstractmethod
    def run_on_question(
        self, question: BencheckQuestion, model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run diagnostic check on a single question.

        Args:
            question: The question to check
            model: DEPRECATED (v0.2.0+). Model should be passed to check's __init__
                instead. This parameter is kept for backward compatibility and is
                typically ignored by modern checks in favor of self.model.
                - For model-dependent checks: Pass model to __init__(model=...)
                - For model-free checks: No model needed

        Returns:
            Flexible dict with check-specific diagnostics.
            Can include scores, predictions, flags, or any diagnostic data.

        Example:
            {
                "predicted_choice": 2,
                "is_correct": True,
                "confidence": 0.87,
                "scores": [0.1, 0.3, 0.9, 0.2]
            }
        """

    @abstractmethod
    def aggregate_metrics(self, question_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute dataset-level metrics from per-question results.

        Args:
            question_results: Map of question_id → result from run_on_question

        Returns:
            Dict mapping metric_name → value(s). Values can be:
            - float: single value (e.g., accuracy: 85.0)
            - tuple: multiple values (e.g., mean±std: (85.0, 3.2))
            - dict: nested metrics (e.g., by_category: {"A": 90.0, "B": 80.0})

        Examples:
            {
                "accuracy": 85.0,
                "confidence": (0.76, 0.12),  # mean, std
                "by_difficulty": {"easy": 95.0, "hard": 60.0}
            }
        """

    def run(
        self,
        dataset: List[BencheckQuestion],
        model: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> CheckResult:
        """
        Execute check on entire dataset (default implementation).

        Args:
            dataset: List of questions to check
            model: DEPRECATED (v0.2.0+). Model should be passed to check's __init__
                instead. Modern checks ignore this parameter and use self.model.
            seed: Optional random seed for reproducibility (for checks using randomness).
                Checks that don't use randomness can ignore this parameter.

        Returns:
            CheckResult with per-question diagnostics and aggregate metrics

        Can be overridden for custom behavior (e.g., batch processing, using seed).
        """
        from .core.progress import ProgressLevel, progress_bar

        # Run with progress bar (VERBOSE level: shown only in verbose mode, hidden by default)
        question_results = {}
        for q in progress_bar(dataset, f"{self.name}", ProgressLevel.VERBOSE, unit="question"):
            # model=None is passed for backward compatibility, but checks use self.model
            # seed is available but most checks don't use it currently
            question_results[q.id] = self.run_on_question(q, model)

        metrics = self.aggregate_metrics(question_results)
        return CheckResult(
            check_name=self.name,
            question_results=question_results,
            metrics=metrics,
        )


class DatasetAdapter(ABC):
    """Abstract base class for loading benchmark datasets."""

    @abstractmethod
    def load(self, source: Any | None = None) -> List[BencheckQuestion]:
        """
        Load questions from source.

        Args:
            source: Dataset source (file path, URL, dict, etc.)

        Returns:
            List of normalized BencheckQuestion objects
        """
