"""
vLLM Model implementation for BenCheck.

Supports:
- Log-likelihood scoring (score_continuations)
- Text generation (generate)
- Full prompt mode (question + choices as one text)
- Separate scoring mode (question separate, score each choice)
"""

from typing import Any, List, Optional

from ..base import ScoreOutput


class VLLMModel:
    """
    vLLM model for local GPU inference.

    Supports TWO modes:
    1. Log-likelihood scoring: Score individual continuations
    2. Full generation: Generate answer from full prompt

    Example:
        >>> # Log-likelihood mode (for MCQ)
        >>> model = VLLMModel("Qwen/Qwen2.5-3B")
        >>> scores = model.score_continuations("Question: What is 2+2?", ["3", "4", "5"])
        >>> predicted = max(range(len(scores)), key=lambda i: scores[i])

        >>> # Generation mode (for open-ended)
        >>> answer = model.generate("What is the capital of France?")
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        gpu_memory_utilization: float = 0.3,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize vLLM model.

        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B")
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            gpu_memory_utilization: GPU memory fraction to use (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            **kwargs: Additional vLLM parameters
        """
        # Suppress vLLM's verbose logging BEFORE importing
        import os

        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm torch")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        print(f"Loading vLLM model: {model_name}")

        self.llm = LLM(
            model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            disable_log_stats=True,  # Suppress stats logging
            **kwargs,
        )

        # Sampling params for generation
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        print(f"✅ vLLM model loaded: {model_name}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt (can be full question + choices)
            **kwargs: Override generation parameters

        Returns:
            Generated text
        """
        from vllm import SamplingParams

        # Use custom params if provided
        if kwargs:
            params = SamplingParams(
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 0.95),
            )
        else:
            params = self.sampling_params

        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text if outputs else ""

    def score_continuation(self, context: str, continuation: str) -> ScoreOutput:
        """
        Score a single continuation given context.

        Uses vLLM's logprobs to compute log-likelihood.

        Args:
            context: Question/prompt text
            continuation: Answer choice to score

        Returns:
            Mean log-probability (higher = more likely)
        """
        from vllm import SamplingParams

        # Create full prompt
        full_text = context + " " + continuation

        # Get prompt logprobs
        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,  # Get logprobs for prompt tokens
            skip_special_tokens=True,
        )

        try:
            outputs = self.llm.generate([full_text], params)

            if not outputs or not outputs[0].prompt_logprobs:
                return 0.0

            # Get logprobs for continuation part
            # Approximate: use last N tokens where N ~ len(continuation tokens)
            prompt_logprobs = outputs[0].prompt_logprobs

            # Simple approach: sum all logprobs, normalize by length
            total_logprob = 0.0
            count = 0

            for token_logprobs in prompt_logprobs:
                if token_logprobs:  # Dict of token_id: Logprob object
                    # Get the actual logprob (first value)
                    logprob_obj = list(token_logprobs.values())[0]
                    # Extract float value from Logprob object
                    if hasattr(logprob_obj, "logprob"):
                        total_logprob += logprob_obj.logprob
                    else:
                        # Fallback if it's already a float
                        total_logprob += float(logprob_obj)
                    count += 1

            # Return mean logprob
            return total_logprob / count if count > 0 else 0.0

        except Exception as e:
            print(f"Warning: Scoring failed: {e}")
            # Fallback to simple length-based heuristic
            return -len(continuation.split()) * 2.0

    def score_continuations(self, context: str, continuations: List[str]) -> List[ScoreOutput]:
        """
        Score multiple continuations.

        Args:
            context: Question/prompt text
            continuations: List of answer choices

        Returns:
            List of scores (one per continuation)
        """
        # For now, score sequentially
        # TODO: Optimize with batching
        return [self.score_continuation(context, cont) for cont in continuations]

    def score_full_prompt(self, prompt: str, choices: List[str]) -> List[float]:
        """
        Alternative scoring: Format as full MCQ prompt and score.

        This is for when you want to score the model's preference
        by generating and parsing the answer.

        Args:
            prompt: Question text
            choices: Answer options

        Returns:
            Scores based on generation (0 or 1 for selected answer)
        """
        # Format as MCQ
        labels = ["A", "B", "C", "D", "E", "F"][: len(choices)]
        formatted_choices = [f"{label}) {choice}" for label, choice in zip(labels, choices)]

        full_prompt = f"Question: {prompt}\n\nChoices:\n" + "\n".join(formatted_choices)
        full_prompt += "\n\nAnswer (just the letter):"

        # Generate answer
        output = self.generate(full_prompt, max_tokens=5, temperature=0.0)

        # Parse which option was chosen
        output_upper = output.strip().upper()
        scores = [0.0] * len(choices)

        for i, label in enumerate(labels[: len(choices)]):
            if label in output_upper:
                scores[i] = 1.0
                break

        return scores
