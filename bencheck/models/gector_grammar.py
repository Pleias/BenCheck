"""GECToR grammar detector wrapper targeted at the gotutiyan/gector fork.

This implementation assumes the fork's module layout and APIs:
    - ``gector.modeling.GECToR`` for model construction/loading
    - ``gector.predict.predict`` for batch inference
    - ``gector.predict.load_verb_dict`` for verb form encoders/decoders

It exposes a minimal detector interface compatible with ``GrammarQualityCheck``
via ``predict`` (single option) and ``run_detector_on_options`` (batch).
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


class GECToRGrammarModel:
    """Wrapper around the gotutiyan/gector fork for grammar correction."""

    def __init__(
        self,
        model_paths: Sequence[str],
        vocab_path: str,
        tokenizer_name: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        iterations: int = 3,
        # We are being very conservative for precision by setting a lower threshold for KEEP tag confidence,
        # and a high minimum error probability; these settings reduce false positives at the expense of recall.
        keep_confidence: float = 0.4,
        min_error_prob: float = 0.8,
        verb_dict_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_paths: One or more GECToR checkpoint paths.
            vocab_path: Path to the action vocabulary file (usually
                ``output_vocabulary`` from GECToR training).
            tokenizer_name: Optional Hugging Face tokenizer name
                (e.g., ``"roberta-base"``).
            device: Device string. If omitted, prefers MPS, then CUDA
                when available, otherwise CPU.
            batch_size: Batch size for inference.
            iterations: Number of correction iterations.
            keep_confidence: KEEP tag confidence threshold.
            min_error_prob: Minimum error probability threshold.
            verb_dict_path: Optional path to verb dictionary.
            **kwargs: Additional keyword arguments forwarded to the model loader.
        """
        self.batch_size = batch_size
        self.iterations = iterations
        self.keep_confidence = keep_confidence
        self.min_error_prob = min_error_prob

        try:
            import torch
            from gector import GECToR
            from gector.predict import load_verb_dict
            from gector.predict import predict as gector_predict
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "The gotutiyan/gector fork (plus torch and transformers) is required "
                "for GECToRGrammarModel. Install the fork and dependencies "
                "with `pip install -e .[grammar]` or `pip install bencheck[grammar]`."
            ) from exc

        def pick_device() -> str:
            if device:
                return device
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"

        self.device = pick_device()
        self.model_paths = list(model_paths)
        if not self.model_paths:
            raise ValueError("At least one model path is required for GECToRGrammarModel.")

        transformer_model = kwargs.pop("transformer_model", tokenizer_name)
        if transformer_model is None:
            raise ValueError(
                "transformer_model or tokenizer_name is required to load the tokenizer."
            )

        model_path = self.model_paths[0]

        # Check if model_path is a local file or a HuggingFace model ID
        # Try to resolve as a path first
        resolved_path = Path(model_path)
        is_local_file = resolved_path.exists() or (not "/" in model_path.replace("\\", "/"))

        if is_local_file and resolved_path.exists():
            # Local checkpoint file - use from_official_pretrained
            # Monkey-patch torch.load to use map_location for CPU device
            original_torch_load = torch.load
            if self.device == "cpu":
                torch.load = lambda *args, **kwargs: original_torch_load(
                    *args, **{**kwargs, "map_location": "cpu"}
                )

            try:
                self._model = GECToR.from_official_pretrained(
                    model_path,
                    vocab_path=vocab_path,
                    special_tokens_fix=1,
                    transformer_model=transformer_model,
                    max_length=512,
                )
            finally:
                # Restore original torch.load
                torch.load = original_torch_load
        else:
            # HuggingFace model ID - use from_pretrained (auto-downloads)
            # Note: from_pretrained() includes vocab, so vocab_path is ignored
            self._model = GECToR.from_pretrained(model_path)
            if self.device == "cpu":
                # Move to CPU if needed
                self._model = self._model.cpu()

        if hasattr(self._model, "to"):
            self._model.to(self.device)
        elif hasattr(self._model, "device"):
            self._model.device = self.device
        self._tokenizer = AutoTokenizer.from_pretrained(transformer_model, add_prefix_space=True)

        if verb_dict_path:
            self._encode, self._decode = load_verb_dict(verb_dict_path)
        else:
            self._encode, self._decode = {}, {}
        self._predict_fn: Callable[..., Any] = gector_predict

    def _detect_token_edits(self, original: str, corrected: str) -> list[dict[str, Any]]:
        """Compute token-level diffs between original and corrected strings."""
        original_tokens = original.split()
        corrected_tokens = corrected.split()
        matcher = difflib.SequenceMatcher(a=original_tokens, b=corrected_tokens)

        edits: list[dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            edits.append(
                {
                    "op": tag,
                    "source_tokens": original_tokens[i1:i2],
                    "target_tokens": corrected_tokens[j1:j2],
                    "source_span": [i1, i2],
                    "target_span": [j1, j2],
                }
            )
        return edits

    def _predict_batch(self, sentences: list[str]) -> list[str]:
        """
        Run GECToR on a batch of sentences.

        Returns:
            List of corrected sentences (one per input).
        """
        corrected = self._predict_fn(
            self._model,
            self._tokenizer,
            sentences,
            self._encode,
            self._decode,
            keep_confidence=self.keep_confidence,
            min_error_prob=self.min_error_prob,
            n_iteration=self.iterations,
            batch_size=self.batch_size,
        )
        # Some implementations return (preds, meta); handle both.
        if isinstance(corrected, tuple):
            corrected = corrected[0]
        return list(corrected)

    def run_detector_on_options(self, options: Sequence[str]) -> list[dict[str, Any]]:
        """
        Batch-correct multiple options.

        Returns:
            List of detector outputs matching the ``GrammarQualityCheck`` expectations.
        """
        normalized = [(opt or "").strip() for opt in options]
        corrected = self._predict_batch(normalized)

        results: list[dict[str, Any]] = []
        for original, suggestion in zip(normalized, corrected):
            results.append(
                {
                    "suggested_text": suggestion,
                    "token_edits": self._detect_token_edits(original, suggestion),
                    "transformations": [{"description": "gector"}],
                }
            )
        return results

    def predict(self, text: str) -> dict[str, Any]:
        """Correct a single option."""
        return self.run_detector_on_options([text])[0]
