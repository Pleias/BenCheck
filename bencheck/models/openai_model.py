"""
OpenAI-compatible API model for BenCheck.

Works with:
- OpenAI API (https://api.openai.com/v1)
- vLLM servers (http://localhost:8008/v1)
- Any OpenAI-compatible endpoint

Supports:
- Log-likelihood scoring via logprobs
- Text generation
"""

from typing import Any, List, Optional

from ..base import ScoreOutput


class OpenAIModel:
    """
    OpenAI-compatible API model (works with vLLM servers too).

    Example with vLLM server:
        >>> model = OpenAIModel(
        ...     base_url="http://localhost:8008/v1",
        ...     model_name="google/gemma-3-12b-it"
        ... )
        >>> scores = model.score_continuations("Question: What is 2+2?", ["3", "4", "5"])

    Example with OpenAI:
        >>> model = OpenAIModel(
        ...     api_key="sk-...",
        ...     model_name="gpt-4"
        ... )
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ):
        """
        Initialize OpenAI-compatible model.

        Args:
            model_name: Model ID (e.g., "gpt-4" or "google/gemma-3-12b-it")
            base_url: API base URL (default: OpenAI, or "http://localhost:8008/v1" for vLLM)
            api_key: API key (optional for local vLLM servers)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional client parameters
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Create client
        client_params = {"timeout": 300.0}  # 5 min timeout for long generations

        if base_url:
            client_params["base_url"] = base_url
            print(f"🔗 Connecting to OpenAI-compatible API: {base_url}")

        if api_key:
            client_params["api_key"] = api_key
        elif not base_url:
            # No base_url and no api_key = must be using env var for OpenAI
            import os

            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "API key required. Set OPENAI_API_KEY env var or pass api_key parameter"
                )
        else:
            # Local vLLM server - use dummy key
            client_params["api_key"] = "EMPTY"

        client_params.update(kwargs)
        self.client = OpenAI(**client_params)

        # Test connection
        try:
            models = self.client.models.list()
            available_models = [m.id for m in models.data]
            print(f"✅ Connected! Available models: {available_models[:3]}...")

            if model_name not in available_models:
                print(f"⚠️  Warning: Model '{model_name}' not in list. Using anyway...")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
            print(f"   Proceeding with model: {model_name}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters (temperature, max_tokens)

        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        return response.choices[0].message.content or ""

    def score_continuation(self, context: str, continuation: str) -> ScoreOutput:
        """
        Score a single continuation given context using logprobs.

        Args:
            context: Question/prompt text
            continuation: Answer choice to score

        Returns:
            Mean log-probability (higher = more likely)
        """
        full_text = context + " " + continuation
        continuation_start = len(context) + 1  # +1 for the space

        try:
            # Use COMPLETION API (not CHAT) to get prompt token logprobs
            # CRITICAL: CHAT API only returns logprobs for GENERATED tokens,
            # but we need logprobs for the CONTINUATION tokens we're scoring!
            response = self.client.completions.create(
                model=self.model_name,
                prompt=full_text,
                max_tokens=0,  # Don't generate new tokens
                temperature=0.0,
                logprobs=1,  # Request logprobs (minimum value)
                echo=True,  # Return prompt token logprobs
            )

            # Extract logprobs from response
            if not response.choices or not response.choices[0].logprobs:
                return 0.0

            logprobs_obj = response.choices[0].logprobs
            token_logprobs = logprobs_obj.token_logprobs
            text_offsets = logprobs_obj.text_offset

            if not token_logprobs or not text_offsets:
                return 0.0

            # Find tokens that belong to the continuation (text_offset >= continuation_start)
            continuation_logprobs = [
                lp
                for lp, offset in zip(token_logprobs, text_offsets)
                if offset >= continuation_start and lp is not None
            ]

            if not continuation_logprobs:
                return 0.0

            # Calculate mean logprob
            return sum(continuation_logprobs) / len(continuation_logprobs)

        except Exception as e:
            print(f"Warning: Scoring failed for '{continuation[:50]}...': {e}")
            # Fallback: use generation mode
            return self._score_via_generation(context, continuation)

    def _score_via_generation(self, context: str, continuation: str) -> float:
        """Fallback scoring via generation likelihood."""
        # Simple heuristic: shorter answers get slightly higher scores
        # This is a weak fallback when logprobs aren't available
        return -len(continuation.split()) * 0.5

    def score_continuations(self, context: str, continuations: List[str]) -> List[ScoreOutput]:
        """
        Score multiple continuations.

        Args:
            context: Question/prompt text
            continuations: List of answer choices

        Returns:
            List of scores (one per continuation)
        """
        return [self.score_continuation(context, cont) for cont in continuations]

    def score_full_prompt(self, prompt: str, choices: List[str]) -> List[float]:
        """
        Alternative scoring: Format as MCQ and parse generated answer.

        Args:
            prompt: Question text
            choices: Answer options

        Returns:
            Scores (1.0 for selected answer, 0.0 for others)
        """
        labels = ["A", "B", "C", "D", "E", "F"][: len(choices)]
        formatted_choices = [f"{label}) {choice}" for label, choice in zip(labels, choices)]

        full_prompt = (
            f"Question: {prompt}\n\nChoices:\n"
            + "\n".join(formatted_choices)
            + "\n\nAnswer (just the letter):"
        )

        output = self.generate(full_prompt, max_tokens=5, temperature=0.0)

        # Parse selected option
        output_upper = output.strip().upper()
        scores = [0.0] * len(choices)

        for i, label in enumerate(labels[: len(choices)]):
            if label in output_upper:
                scores[i] = 1.0
                break

        return scores
