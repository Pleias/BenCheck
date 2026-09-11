"""Tests for HellaSwag adapter - ensuring all questions are loaded correctly."""

import pytest

from bencheck.adapters.hellaswag import HellaSwagAdapter


def test_hellaswag_loads_all_questions():
    """Test that HellaSwag adapter loads all 10,042 questions."""
    adapter = HellaSwagAdapter(split="validation")
    questions = adapter.load()

    # HellaSwag validation has 10,042 questions
    assert len(questions) == 10042, f"Expected 10042 questions, got {len(questions)}"


def test_hellaswag_unique_question_ids():
    """Test that all question IDs are unique (no duplicates)."""
    adapter = HellaSwagAdapter(split="validation")
    questions = adapter.load()

    question_ids = [q.id for q in questions]
    unique_ids = set(question_ids)

    assert len(question_ids) == len(unique_ids), (
        f"Found duplicate question_ids: {len(question_ids)} total, "
        f"{len(unique_ids)} unique, {len(question_ids) - len(unique_ids)} duplicates"
    )


def test_hellaswag_preserves_original_ind():
    """Test that original 'ind' field is preserved in metadata."""
    adapter = HellaSwagAdapter(split="validation")
    questions = adapter.load()

    # Check that original_ind is present in metadata
    for q in questions[:10]:  # Check first 10 questions
        assert "original_ind" in q.metadata, (
            f"Question {q.id} missing 'original_ind' in metadata"
        )
        assert isinstance(q.metadata["original_ind"], int), (
            f"Question {q.id} has non-integer 'original_ind': "
            f"{type(q.metadata['original_ind'])}"
        )


def test_hellaswag_question_id_is_row_index():
    """Test that question_id equals row index (0, 1, 2, ...)."""
    adapter = HellaSwagAdapter(split="validation")
    questions = adapter.load()

    # Question IDs should be sequential: "0", "1", "2", ...
    expected_ids = [str(i) for i in range(len(questions))]
    actual_ids = [q.id for q in questions]

    assert actual_ids == expected_ids, (
        f"Question IDs are not sequential row indices. "
        f"First 10: {actual_ids[:10]}, Expected: {expected_ids[:10]}"
    )


def test_hellaswag_original_ind_has_duplicates():
    """Test that original_ind field has duplicates (validates the fix was needed)."""
    adapter = HellaSwagAdapter(split="validation")
    questions = adapter.load()

    original_inds = [q.metadata["original_ind"] for q in questions]
    unique_inds = set(original_inds)

    # Should have 9,609 unique 'ind' values in 10,042 questions (433 duplicates)
    assert len(unique_inds) == 9609, (
        f"Expected 9609 unique original_ind values, got {len(unique_inds)}"
    )
    assert len(original_inds) - len(unique_inds) == 433, (
        f"Expected 433 duplicates, got {len(original_inds) - len(unique_inds)}"
    )


def test_hellaswag_basic_structure():
    """Test basic structure of loaded questions."""
    adapter = HellaSwagAdapter(split="validation")
    questions = adapter.load()

    assert len(questions) > 0, "No questions loaded"

    # Check first question structure
    q = questions[0]
    assert q.question is not None and len(q.question) > 0, "Question is empty"
    assert q.choices is not None and len(q.choices) == 4, f"Expected 4 choices, got {len(q.choices)}"
    assert isinstance(q.correct, int), f"Expected int correct index, got {type(q.correct)}"
    assert 0 <= q.correct < 4, f"Correct index {q.correct} out of range [0-3]"


@pytest.mark.slow
def test_hellaswag_no_data_loss():
    """Integration test: verify no data is lost during loading."""
    from datasets import load_dataset

    # Load original dataset
    original_ds = load_dataset("Rowan/hellaswag", split="validation")
    original_count = len(original_ds)

    # Load through adapter
    adapter = HellaSwagAdapter(split="validation")
    questions = adapter.load()
    adapter_count = len(questions)

    assert adapter_count == original_count, (
        f"Data loss detected: original has {original_count} records, "
        f"adapter loaded {adapter_count}"
    )
