"""A tiny deterministic model for demonstrating the bencheck pipeline."""

from __future__ import annotations

from typing import List

from ..base import ScoreOutput


class DummyModel:
    """
    Placeholder model used for internal testing of the bencheck infrastructure.

    Implements both scoring and generation with simple lexical heuristics.
    This model satisfies both ScoringModel and GenerativeModel protocols.
    """

    def __init__(self) -> None:
        self._generation_counter = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Return a deterministic response including the prompt length.

        Args:
            prompt: The input prompt
            **kwargs: Ignored (for protocol compatibility)

        Returns:
            A string describing the prompt length
        """
        self._generation_counter += 1
        trimmed = prompt.strip()
        token_count = len(trimmed.split()) if trimmed else 0
        return f"[dummy:{self._generation_counter}] {token_count} tokens observed"

    def score_continuation(self, context: str, continuation: str) -> ScoreOutput:
        """
        Assign a pseudo score based on the lexical overlap between context and continuation.

        Scoring strategy:
        - +1 for each word in continuation that appears in context
        - +0.1 for each word in continuation (length bonus)

        Args:
            context: The context/prompt
            continuation: The text to score as continuation

        Returns:
            Float score (higher = better match)
        """
        context_tokens = set(token.lower().strip(".,?!") for token in context.split())
        continuation_tokens = [token.lower().strip(".,?!") for token in continuation.split()]
        overlap = sum(1 for token in continuation_tokens if token in context_tokens)
        bonus = len(continuation_tokens) * 0.1
        return float(overlap) + bonus

    def score_continuations(self, context: str, continuations: List[str]) -> List[ScoreOutput]:
        """
        Score multiple continuations (batch processing).

        Args:
            context: The context/prompt
            continuations: List of texts to score

        Returns:
            List of scores
        """
        return [self.score_continuation(context, cont) for cont in continuations]
