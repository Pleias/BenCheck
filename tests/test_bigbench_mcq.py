"""Test BigBench adapter MCQ handling.

This module tests that BigBenchAdapter correctly handles:
1. Single-choice tasks (one correct answer)
2. Multiple-choice tasks (multiple correct answers)
3. Proper question type detection
"""

import pytest

from bencheck.adapters import BigBenchAdapter
from bencheck.types import QuestionType


class TestBigBenchSingleChoice:
    """Test BigBench adapter with single-choice tasks."""

    def test_single_choice_task_detection(self):
        """
        Test that tasks with one correct answer are detected as SINGLE_CHOICE.

        BigBench format uses multiple_choice_scores with binary labels.
        Single correct answer = exactly one 1 in the scores array.
        """
        mock_data = [
            {
                "idx": 0,
                "inputs": "Q: What movie does this emoji describe? 👦👓⚡️\n  choice: harry potter\n  choice: shutter island\n  choice: die hard",
                "targets": ["harry potter"],
                "multiple_choice_targets": ["harry potter", "shutter island", "die hard"],
                "multiple_choice_scores": [1, 0, 0],  # Only one correct
            }
        ]

        adapter = BigBenchAdapter(task_name="emoji_movie")
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert questions[0].question_type == QuestionType.SINGLE_CHOICE
        assert questions[0].correct == 0  # First choice is correct

    def test_extracts_choices_correctly(self):
        """Test that multiple_choice_targets are extracted as choices."""
        mock_data = [
            {
                "idx": 0,
                "inputs": "Q: 2 + 2 = ?\n  choice: 3\n  choice: 4\n  choice: 5",
                "targets": ["4"],
                "multiple_choice_targets": ["3", "4", "5"],
                "multiple_choice_scores": [0, 1, 0],
            }
        ]

        adapter = BigBenchAdapter(task_name="simple_arithmetic")
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert questions[0].choices == ["3", "4", "5"]
        assert questions[0].correct == 1  # Second choice (index 1)


class TestBigBenchMultipleChoice:
    """Test BigBench adapter with true MCQ tasks (multiple correct answers)."""

    def test_mcq_task_detection(self):
        """
        Test that tasks with multiple correct answers are detected as MULTIPLE_CHOICE.

        BigBench MCQ format: multiple 1s in multiple_choice_scores array.
        """
        mock_data = [
            {
                "idx": 0,
                "inputs": "Q: Which are prime numbers?\n  choice: 2\n  choice: 3\n  choice: 4\n  choice: 5",
                "targets": ["2", "3", "5"],
                "multiple_choice_targets": ["2", "3", "4", "5"],
                "multiple_choice_scores": [1, 1, 0, 1],  # Three correct answers
            }
        ]

        adapter = BigBenchAdapter(task_name="prime_numbers")
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert questions[0].question_type == QuestionType.MULTIPLE_CHOICE
        # Should extract all correct indices
        assert questions[0].correct == [0, 1, 3]

    def test_mcq_metadata(self):
        """Test that MCQ metadata is correctly populated."""
        mock_data = [
            {
                "idx": 0,
                "inputs": "Q: Which are even?\n  choice: 1\n  choice: 2\n  choice: 3\n  choice: 4",
                "targets": ["2", "4"],
                "multiple_choice_targets": ["1", "2", "3", "4"],
                "multiple_choice_scores": [0, 1, 0, 1],
            }
        ]

        adapter = BigBenchAdapter(task_name="even_numbers")
        questions = adapter.load(source=mock_data)

        assert questions[0].metadata["question_type"] == "multiple_choice"
        assert questions[0].metadata["num_correct_answers"] == 2


class TestBigBenchEdgeCases:
    """Test edge cases and validation."""

    def test_missing_multiple_choice_fields(self):
        """Test handling when multiple choice fields are missing."""
        mock_data = [
            {
                "idx": 0,
                "inputs": "Some text input",
                "targets": ["answer"],
                # No multiple_choice_targets or multiple_choice_scores
            }
        ]

        adapter = BigBenchAdapter(task_name="text_task")
        questions = adapter.load(source=mock_data)

        # Should skip record without MCQ fields
        assert len(questions) == 0

    def test_mismatched_targets_and_scores_length(self):
        """Test handling when targets and scores have different lengths."""
        mock_data = [
            {
                "idx": 0,
                "inputs": "Q: Test?",
                "targets": ["A"],
                "multiple_choice_targets": ["A", "B", "C"],
                "multiple_choice_scores": [1, 0],  # Length mismatch
            }
        ]

        adapter = BigBenchAdapter(task_name="test_task")
        # Should handle gracefully (implementation dependent)
        questions = adapter.load(source=mock_data)
        # BaseMCQAdapter may skip malformed records
        assert isinstance(questions, list)

    def test_all_zeros_in_scores(self):
        """Test handling when no answer is marked correct."""
        mock_data = [
            {
                "idx": 0,
                "inputs": "Q: Test?",
                "targets": [],
                "multiple_choice_targets": ["A", "B", "C"],
                "multiple_choice_scores": [0, 0, 0],  # No correct answer
            }
        ]

        adapter = BigBenchAdapter(task_name="no_correct")
        questions = adapter.load(source=mock_data)

        # Should skip records with no correct answer
        assert len(questions) == 0


class TestBigBenchFormatConsistency:
    """Test BigBench format consistency with BaseMCQAdapter."""

    def test_uses_base_mcq_adapter_with_custom_fields(self):
        """Verify that BigBenchAdapter correctly specifies BigBench field names."""
        from bencheck.adapters.base_mcq import BaseMCQAdapter

        adapter = BigBenchAdapter(task_name="test")
        assert isinstance(adapter, BaseMCQAdapter)
        # BigBench has specific field names that must be configured
        assert adapter.option_key == "multiple_choice_targets"
        assert adapter.label_key == "multiple_choice_scores"
        assert adapter.context_key == "inputs"

    def test_compatible_with_checks(self):
        """Test that BigBench questions work with existing checks."""
        from bencheck.checks import LengthBiasCheck

        mock_data = [
            {
                "idx": 0,
                "inputs": "Q: Short question?",
                "targets": ["A"],
                "multiple_choice_targets": ["Short A", "Very long answer B"],
                "multiple_choice_scores": [1, 0],
            }
        ]

        adapter = BigBenchAdapter(task_name="test")
        questions = adapter.load(source=mock_data)

        # Should work with checks
        check = LengthBiasCheck()
        result = check.run_on_question(questions[0])
        assert result is not None
        assert isinstance(result, dict)
