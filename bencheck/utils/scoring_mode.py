"""Scoring mode detection and filtering utilities."""

import inspect
from typing import Any, List


def get_supported_modes(model: Any) -> List[str]:
    """
    Detect which scoring modes a model supports.

    Args:
        model: Model instance to inspect

    Returns:
        List of supported modes: ["log_likelihood"] or ["generation"] or both

    Example:
        >>> from bencheck.models import VLLMModel, GeminiModel
        >>> vllm = VLLMModel("Qwen/Qwen2.5-3B")
        >>> get_supported_modes(vllm)
        ['log_likelihood', 'generation']
        >>>
        >>> gemini = GeminiModel("gemini-2.0-flash-exp")
        >>> get_supported_modes(gemini)
        ['generation']
    """
    if model is None:
        return []

    modes = []

    # Check for log_likelihood support (score_continuations method)
    if hasattr(model, "score_continuations"):
        try:
            # Check it's not just a NotImplementedError stub
            source = inspect.getsource(model.score_continuations)
            if "NotImplementedError" not in source:
                modes.append("log_likelihood")
        except (OSError, TypeError):
            # inspect.getsource can fail for built-in or C-extension methods
            # In that case, assume it's implemented
            modes.append("log_likelihood")

    # Check for generation support (generate method)
    if hasattr(model, "generate"):
        modes.append("generation")

    return modes


def filter_modes_for_check(check_name: str, requested_modes: List[str], model: Any) -> List[str]:
    """
    Filter scoring modes based on check requirements and model capabilities.

    Args:
        check_name: Name of the check
        requested_modes: List of requested modes (e.g., ["log_likelihood", "generation"])
        model: Model instance

    Returns:
        List of applicable modes for this check-model combination

    Examples:
        >>> # Model-free check - no modes needed
        >>> filter_modes_for_check("LengthBiasCheck", ["log_likelihood"], None)
        []
        >>>
        >>> # ContextRequirementCheck supports both modes
        >>> filter_modes_for_check(
        ...     "ContextRequirementCheck",
        ...     ["log_likelihood", "generation"],
        ...     vllm_model
        ... )
        ['log_likelihood', 'generation']
        >>>
        >>> # EnumerationBiasCheck only uses log_likelihood
        >>> filter_modes_for_check(
        ...     "EnumerationBiasCheck",
        ...     ["log_likelihood", "generation"],
        ...     vllm_model
        ... )
        ['log_likelihood']
    """
    # Model-free checks don't use scoring modes
    if check_name in ["LengthBiasCheck", "DummyCheck"]:
        return []

    if model is None:
        # No model provided, can't score
        return []

    # Get model's supported modes
    supported = get_supported_modes(model)

    # Check-specific logic
    if check_name == "ContextRequirementCheck":
        # This check supports both modes via scoring_mode parameter
        # Return intersection of requested and supported
        return [m for m in requested_modes if m in supported]

    # Most checks (EnumerationBiasCheck, NoneOfTheAboveCheck, etc.)
    # only use log_likelihood (score_continuations)
    if "log_likelihood" in supported and "log_likelihood" in requested_modes:
        return ["log_likelihood"]

    return []


def should_run_check_with_mode(check_name: str, model: Any, scoring_mode: str) -> bool:
    """
    Determine if a check should run with a given scoring mode.

    Args:
        check_name: Name of the check
        model: Model instance
        scoring_mode: Scoring mode ("log_likelihood" or "generation")

    Returns:
        True if check should run with this mode, False otherwise

    Example:
        >>> # ContextRequirementCheck can run with both modes
        >>> should_run_check_with_mode("ContextRequirementCheck", vllm, "log_likelihood")
        True
        >>> should_run_check_with_mode("ContextRequirementCheck", vllm, "generation")
        True
        >>>
        >>> # EnumerationBiasCheck only runs with log_likelihood
        >>> should_run_check_with_mode("EnumerationBiasCheck", vllm, "log_likelihood")
        True
        >>> should_run_check_with_mode("EnumerationBiasCheck", vllm, "generation")
        False
    """
    if check_name in ["LengthBiasCheck", "DummyCheck"]:
        # Model-free checks always run (mode doesn't matter)
        return True

    if model is None:
        return False

    supported = get_supported_modes(model)

    if scoring_mode not in supported:
        return False

    # Check-specific requirements
    if check_name == "ContextRequirementCheck":
        # Supports both modes
        return True

    # Most checks only use log_likelihood
    return scoring_mode == "log_likelihood"
