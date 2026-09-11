"""
Question transformation utilities for dataset quality testing.
Supports replacing question text with empty strings or Lorem Ipsum.
"""

import random
from typing import Literal

# Standard Lorem Ipsum text segments
LOREM_IPSUM_SEGMENTS = [
    "Lorem ipsum dolor sit amet",
    "consectetur adipiscing elit",
    "sed do eiusmod tempor incididunt",
    "ut labore et dolore magna aliqua",
    "Ut enim ad minim veniam",
    "quis nostrud exercitation ullamco",
    "laboris nisi ut aliquip ex ea commodo consequat",
    "Duis aute irure dolor in reprehenderit",
    "in voluptate velit esse cillum dolore",
    "eu fugiat nulla pariatur",
    "Excepteur sint occaecat cupidatat non proident",
    "sunt in culpa qui officia deserunt",
    "mollit anim id est laborum",
    "Sed ut perspiciatis unde omnis iste",
    "natus error sit voluptatem accusantium",
    "doloremque laudantium totam rem aperiam",
    "eaque ipsa quae ab illo inventore",
    "veritatis et quasi architecto beatae",
    "vitae dicta sunt explicabo",
    "Nemo enim ipsam voluptatem quia",
]


def generate_lorem_ipsum(
    target_length: int, match_type: Literal["characters", "words"] = "characters", seed: int = 42
) -> str:
    """
    Generate Lorem Ipsum text matching a target length.

    Args:
        target_length: Target length to match
        match_type: Whether to match by "characters" or "words"
        seed: Random seed for reproducibility

    Returns:
        Lorem Ipsum text of approximately the target length
    """
    random.seed(seed)

    if target_length == 0:
        return ""

    if match_type == "words":
        # Generate by word count
        words = []
        for segment in LOREM_IPSUM_SEGMENTS:
            words.extend(segment.split())

        # Cycle through words until we reach target length
        result_words = []
        while len(result_words) < target_length:
            remaining = target_length - len(result_words)
            if remaining >= len(words):
                result_words.extend(words)
            else:
                result_words.extend(words[:remaining])

        return " ".join(result_words[:target_length])

    else:  # match_type == "characters"
        # Generate by character count
        text = " ".join(LOREM_IPSUM_SEGMENTS)

        # Repeat text until we have enough characters
        repeats = (target_length // len(text)) + 2
        full_text = (text + " ") * repeats

        # Trim to exact length, trying to end at a word boundary
        if target_length < len(full_text):
            trimmed = full_text[:target_length]
            # Try to cut at last space
            last_space = trimmed.rfind(" ")
            if last_space > target_length * 0.9:  # Only if we're close to target
                return trimmed[:last_space]
            return trimmed

        return full_text[:target_length]


def remove_question_text(question_dict: dict) -> dict:
    """
    Replace question text with empty string.

    Args:
        question_dict: Dictionary with 'question' key

    Returns:
        Modified dictionary with empty question
    """
    result = question_dict.copy()
    result["question"] = ""
    result["_transform"] = "empty"
    return result


def replace_with_lorem_ipsum(
    question_dict: dict,
    match_by: Literal["none", "characters", "words"] = "characters",
    seed: int = 42,
) -> dict:
    """
    Replace question text with Lorem Ipsum.

    Args:
        question_dict: Dictionary with 'question' key
        match_by: How to match length - "none" (fixed), "characters", or "words"
        seed: Random seed for reproducibility

    Returns:
        Modified dictionary with Lorem Ipsum question

    Examples:
        >>> q = {"question": "What is 2+2?", "choices": ["3", "4", "5"], "answer": 1}
        >>> replace_with_lorem_ipsum(q, match_by="words")
        {'question': 'Lorem ipsum dolor', 'choices': ['3', '4', '5'], 'answer': 1, '_transform': 'lorem_ipsum_words'}

        >>> replace_with_lorem_ipsum(q, match_by="characters")
        {'question': 'Lorem ipsum do', 'choices': ['3', '4', '5'], 'answer': 1, '_transform': 'lorem_ipsum_chars'}

        >>> replace_with_lorem_ipsum(q, match_by="none")
        {'question': 'Lorem ipsum dolor sit amet', 'choices': ['3', '4', '5'], 'answer': 1, '_transform': 'lorem_ipsum_fixed'}
    """
    result = question_dict.copy()
    original_question = result["question"]

    if match_by == "none":
        # Fixed Lorem Ipsum phrase
        result["question"] = "Lorem ipsum dolor sit amet"
        result["_transform"] = "lorem_ipsum_fixed"
    elif match_by == "words":
        # Match word count
        word_count = len(original_question.split())
        result["question"] = generate_lorem_ipsum(word_count, match_type="words", seed=seed)
        result["_transform"] = "lorem_ipsum_words"
    elif match_by == "characters":
        # Match character count
        char_count = len(original_question)
        result["question"] = generate_lorem_ipsum(char_count, match_type="characters", seed=seed)
        result["_transform"] = "lorem_ipsum_chars"
    else:
        raise ValueError(f"Invalid match_by: {match_by}. Use 'none', 'characters', or 'words'")

    return result


def apply_transform(
    question_dict: dict,
    transform_type: Literal["original", "empty", "lorem_ipsum"],
    lorem_match_by: Literal["none", "characters", "words"] = "characters",
    seed: int = 42,
) -> dict:
    """
    Apply a transformation to a question.

    Args:
        question_dict: Dictionary with 'question' key
        transform_type: Type of transformation - "original", "empty", or "lorem_ipsum"
        lorem_match_by: For lorem_ipsum, how to match length
        seed: Random seed for reproducibility

    Returns:
        Transformed question dictionary
    """
    if transform_type == "original":
        result = question_dict.copy()
        result["_transform"] = "original"
        return result
    elif transform_type == "empty":
        return remove_question_text(question_dict)
    elif transform_type == "lorem_ipsum":
        return replace_with_lorem_ipsum(question_dict, match_by=lorem_match_by, seed=seed)
    else:
        raise ValueError(f"Invalid transform_type: {transform_type}")


def format_question_for_inference(question_dict: dict, instruction_model: bool = False) -> str:
    """
    Format a question dictionary into a prompt string for inference.

    Args:
        question_dict: Dictionary with 'question' and 'choices' keys

    Returns:
        Formatted prompt string
    """
    question_text = question_dict.get("question", "")
    choices = question_dict.get("choices", [])

    if not question_text and not choices:
        return ""

    # Format choices with labels A, B, C, D, etc.
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    formatted_choices = []
    for i, choice in enumerate(choices):
        if i < len(choice_labels):
            formatted_choices.append(f"{choice_labels[i]}) {choice}")
        else:
            formatted_choices.append(f"{i + 1}) {choice}")

    # Build prompt
    instruction = (
        "Reply to the following multiple choice question by only returning the plain letter of the correct choice"
        if instruction_model
        else ""
    )
    if question_text:
        instruction += f"Question: {question_text}\n\n"
    else:
        instruction += ""

    if formatted_choices:
        instruction += "Choices:\n" + "\n".join(formatted_choices) + "\n\nAnswer:"

    return instruction
