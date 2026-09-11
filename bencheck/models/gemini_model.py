"""
Gemini Model implementation for BenCheck.

Supports:
- Text generation (generate)
- Full prompt scoring via generation (not log-likelihood)
"""

import os
from typing import Any, List, Optional

from ..base import ScoreOutput


class GeminiModel:
    """
    Gemini API model for cloud-based inference.

    **Note**: Gemini API does NOT support log-likelihood scoring.
    Only generation mode is available.

    For MCQ evaluation with Gemini, use generation mode:
    - Format question + choices as full prompt
    - Generate answer
    - Parse which option was selected

    Example:
        >>> # With API key
        >>> model = GeminiModel("gemini-2.0-flash-exp", api_key="your-key")

        >>> # Or from environment
        >>> model = GeminiModel("gemini-2.0-flash-exp")  # Uses GEMINI_API_KEY env var

        >>> # Generate answer
        >>> answer = model.generate("What is the capital of France?")

        >>> # Score MCQ (via generation)
        >>> scores = model.score_full_prompt(
        ...     "What is 2+2?",
        ...     ["3", "4", "5", "6"]
        ... )
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """
        Initialize Gemini model.

        Args:
            model_name: Gemini model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro")
            api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Google GenAI not installed. Install with: pip install google-genai")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Either:\n"
                "  1. Pass api_key parameter: GeminiModel(api_key='your-key')\n"
                "  2. Set environment variable: export GEMINI_API_KEY='your-key'"
            )

        print(f"Initializing Gemini model: {model_name}")
        self.client = genai.Client(api_key=key)
        self.genai_types = types

        print(f"✅ Gemini model initialized: {model_name}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters

        Returns:
            Generated text
        """
        config = self.genai_types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            if response.text:
                return response.text
            else:
                return ""

        except Exception as e:
            print(f"Warning: Gemini generation failed: {e}")
            return ""

    def score_continuation(self, context: str, continuation: str) -> ScoreOutput:
        """
        NOT SUPPORTED for Gemini.

        Gemini API does not provide log-likelihood scoring.
        Use score_full_prompt() instead.
        """
        raise NotImplementedError(
            "Gemini API does not support log-likelihood scoring.\n"
            "Use score_full_prompt() or generation-based evaluation instead."
        )

    def score_continuations(self, context: str, continuations: List[str]) -> List[ScoreOutput]:
        """
        NOT SUPPORTED for Gemini.

        Gemini API does not provide log-likelihood scoring.
        Use score_full_prompt() instead.
        """
        raise NotImplementedError(
            "Gemini API does not support log-likelihood scoring.\n"
            "Use score_full_prompt() or generation-based evaluation instead."
        )

    def score_full_prompt(self, prompt: str, choices: List[str]) -> List[float]:
        """
        Score MCQ by generating answer from full prompt.

        Formats question + choices as MCQ, generates answer, parses selection.

        Args:
            prompt: Question text
            choices: Answer options

        Returns:
            Scores (1.0 for selected answer, 0.0 for others)
        """
        # Format as MCQ
        labels = ["A", "B", "C", "D", "E", "F"][: len(choices)]
        formatted_choices = [f"{label}) {choice}" for label, choice in zip(labels, choices)]

        full_prompt = (
            f"Question: {prompt}\n\n"
            f"Choices:\n" + "\n".join(formatted_choices) + "\n\n"
            f"Answer with just the letter (A, B, C, etc.):"
        )

        # Generate answer
        output = self.generate(full_prompt, max_tokens=10, temperature=0.0)

        # Parse which option was chosen
        output_upper = output.strip().upper()
        scores = [0.0] * len(choices)

        for i, label in enumerate(labels[: len(choices)]):
            if label in output_upper:
                scores[i] = 1.0
                break

        # If no match, assume first option (fallback)
        if sum(scores) == 0:
            scores[0] = 1.0

        return scores

    def evaluate_mcq(self, question: str, choices: List[str]) -> int:
        """
        Convenience method: Evaluate MCQ and return predicted index.

        Args:
            question: Question text
            choices: Answer options

        Returns:
            Index of predicted answer (0-based)
        """
        scores = self.score_full_prompt(question, choices)
        return int(scores.index(max(scores)))
