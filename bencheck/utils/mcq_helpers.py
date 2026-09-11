"""Helper utilities for multiple-choice question adapters.

These utilities are used by MCQ adapters to normalize various dataset formats.
Extracted from exp1.ipynb for reusability across adapters.
"""

import string
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Field name candidates (in priority order)
OPTION_KEYS = (
    "multiple_choice_targets",
    "mc1_targets",
    "mc2_targets",
    "endings",
    "options",
    "choices",
    "candidates",
    "answer_choices",
)

LABEL_KEYS = (
    "mc1_targets",
    "mc2_targets",
    "mc1_targets_scores",
    "mc2_targets_scores",
    "answer_index",
    "answer_idx",
    "target_index",
    "label",
    "gold",
    "answer",
    "correct",
    "correct_answer",
    "target",
    "targets",
    "mc1_labels",
)

CONTEXT_KEYS = (
    "inputs",
    "context",
    "ctx",
    "question",
    "prompt",
    "passage",
    "description",
)

ID_KEYS = ("id", "uid", "ind", "idx", "question_id")

# Letter to index mapping (A=0, B=1, C=2, D=3, ...)
LETTER_TO_INDEX = {c: i for i, c in enumerate(string.ascii_uppercase)}


def detect_field_key(
    record: Mapping[str, Any], candidates: tuple[str, ...], provided_key: Optional[str] = None
) -> Optional[str]:
    """
    Detect field key from record, trying provided key first, then candidates.

    Args:
        record: The data record to inspect
        candidates: Tuple of candidate field names (in priority order)
        provided_key: Explicit key to try first (None = auto-detect)

    Returns:
        Detected field key or None if not found
    """
    # Use provided key if given and exists
    if provided_key and provided_key in record:
        return provided_key

    # Try candidates in order
    for key in candidates:
        if key in record:
            return key

    return None


def extract_options(raw_options: Any) -> List[str]:
    """
    Extract option list from various formats.

    Handles:
    - Dict with "choices" key (e.g., TruthfulQA mc1_targets)
    - List/sequence of strings
    - Single value (fallback)

    Args:
        raw_options: Raw options in various formats

    Returns:
        List of option strings
    """
    if raw_options is None:
        return []

    # Dict with "choices" key (e.g., TruthfulQA mc1_targets)
    if isinstance(raw_options, Mapping):
        if "choices" in raw_options:
            return [str(o) for o in raw_options["choices"]]
        # Fallback: use all values
        return [str(v) for v in raw_options.values() if v]

    # List/sequence
    if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
        return [str(o) for o in raw_options]

    # Single value (shouldn't happen, but handle gracefully)
    return [str(raw_options)]


def extract_label_index(record: Mapping[str, Any], label_key: str, options: List[str]) -> int:
    """
    Extract correct label index from various formats.

    Handles:
    - Integer indices
    - String indices (convertible to int)
    - Letter indices (A, B, C, D)
    - Dict with "labels" array (TruthfulQA)
    - List of binary labels (mc1_labels style)
    - String matching against options

    Args:
        record: The data record
        label_key: Key to look up label value
        options: List of option strings (for matching)

    Returns:
        Integer index of correct answer (0-based)
    """
    label_val = record.get(label_key)

    if label_val is None:
        return 0

    # Dict with "labels" array (e.g., TruthfulQA mc1_targets)
    if isinstance(label_val, Mapping) and "labels" in label_val:
        labels_seq = label_val.get("labels", [])
        if labels_seq:
            # Find index of first 1 (or max value)
            import numpy as np

            return int(np.argmax(labels_seq))

    # Integer or convertible to integer
    if isinstance(label_val, int):
        return label_val
    if isinstance(label_val, (float, bool)):
        return int(label_val)

    # String processing
    if isinstance(label_val, str):
        # Try direct match against options
        if label_val in options:
            return options.index(label_val)

        # Try letter index (A=0, B=1, etc.)
        if label_val.upper() in LETTER_TO_INDEX:
            return LETTER_TO_INDEX[label_val.upper()]

        # Try converting to integer
        try:
            return int(label_val)
        except ValueError:
            pass

    # List of labels (mc1_labels style - list of 0s and 1s)
    if isinstance(label_val, Sequence) and not isinstance(label_val, (str, bytes)):
        for j, v in enumerate(label_val):
            try:
                if int(v) == 1:
                    return j
            except (ValueError, TypeError):
                continue

    # Default fallback
    return 0


def extract_label_indices(
    record: Mapping[str, Any], label_key: str, options: List[str]
) -> List[int]:
    """
    Extract multiple correct label indices from various formats.

    Similar to extract_label_index() but returns ALL correct answers as a list.
    Used for true multiple-choice questions where more than one answer is correct.

    Handles:
    - List of integer indices: [0, 2, 3]
    - List of string indices: ["1", "3", "4"]
    - List of letters: ["A", "C", "D"]
    - Binary labels array: [1, 0, 1, 1] (TruthfulQA mc2 format)
    - Dict with "labels" array with multiple 1s

    Args:
        record: The data record
        label_key: Key to look up label value
        options: List of option strings (for matching)

    Returns:
        List of integer indices of all correct answers (0-based)

    Examples:
        >>> extract_label_indices({"answer": [0, 2, 3]}, "answer", ["a", "b", "c", "d"])
        [0, 2, 3]
        >>> extract_label_indices({"labels": [1, 0, 1, 1]}, "labels", ["a", "b", "c", "d"])
        [0, 2, 3]
    """
    label_val = record.get(label_key)

    if label_val is None:
        return [0]

    # Dict with "labels" array (e.g., TruthfulQA mc2_targets)
    if isinstance(label_val, Mapping) and "labels" in label_val:
        labels_seq = label_val.get("labels", [])
        if labels_seq:
            # Find ALL indices where value == 1
            import numpy as np

            return [i for i, v in enumerate(labels_seq) if int(v) == 1]

    # List of labels (mc2_labels style - list of 0s and 1s)
    # OR list of indices
    if isinstance(label_val, Sequence) and not isinstance(label_val, (str, bytes)):
        # Check if it's a binary labels array (contains only 0s and 1s)
        try:
            int_vals = [int(v) for v in label_val]
            if all(v in (0, 1) for v in int_vals):
                # Binary labels - return indices where value == 1
                return [i for i, v in enumerate(int_vals) if v == 1]
        except (ValueError, TypeError):
            pass

        # Not binary labels - treat as list of indices/letters
        indices = []
        for item in label_val:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str):
                # Try letter index (A=0, B=1, etc.)
                if item.upper() in LETTER_TO_INDEX:
                    indices.append(LETTER_TO_INDEX[item.upper()])
                # Try converting to integer
                else:
                    try:
                        indices.append(int(item))
                    except ValueError:
                        pass
        return indices if indices else [0]

    # Single value - use extract_label_index and wrap in list
    single_idx = extract_label_index(record, label_key, options)
    return [single_idx]


def extract_context(record: Mapping[str, Any], context_key: Optional[str] = None) -> str:
    """
    Extract question context/prompt from record.

    Handles:
    - Direct context field
    - HellaSwag-style ctx / ctx_a + ctx_b
    - Fallback to empty string

    Args:
        record: The data record
        context_key: Key to look up context (None = auto-detect)

    Returns:
        Context string
    """
    # Try provided context key
    if context_key and context_key in record:
        return str(record[context_key])

    # Try HellaSwag-style ctx
    if "ctx" in record:
        return str(record["ctx"])

    # Try HellaSwag-style ctx_a + ctx_b
    if "ctx_a" in record:
        ctx_a = str(record.get("ctx_a", ""))
        ctx_b = str(record.get("ctx_b", ""))
        return f"{ctx_a} {ctx_b}".strip()

    # Try common context keys
    for key in CONTEXT_KEYS:
        if key in record:
            return str(record[key])

    return ""


def extract_question_id(record: Mapping[str, Any], idx: int, id_key: Optional[str] = None) -> str:
    """
    Extract question ID from record, with fallback to index.

    Args:
        record: The data record
        idx: Fallback index if no ID found
        id_key: Key to look up ID (None = auto-detect)

    Returns:
        Question ID as string
    """
    # Try provided ID key
    if id_key and id_key in record:
        return str(record[id_key])

    # Try common ID keys
    for key in ID_KEYS:
        if key in record:
            return str(record[key])

    # Fallback to index
    return str(idx)


def build_metadata(
    record: Mapping[str, Any],
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    config_name: Optional[str] = None,
    extra_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build metadata dict for BencheckQuestion.

    Args:
        record: The data record
        dataset_name: Name of dataset
        split: Dataset split (train/validation/test)
        config_name: Optional config/subset name
        extra_keys: Additional keys to extract from record (e.g., "source_id")

    Returns:
        Metadata dictionary
    """
    metadata: Dict[str, Any] = {}

    if dataset_name:
        metadata["dataset"] = dataset_name
    if split:
        metadata["split"] = split
    if config_name:
        metadata["config"] = config_name

    # Extract extra keys from record
    if extra_keys:
        for key in extra_keys:
            if key in record:
                metadata[key] = record[key]

    return metadata
