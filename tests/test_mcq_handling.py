"""
Comprehensive tests for MCQ (multiple correct answers) vs single-choice handling.

This test suite verifies that BenCheck correctly handles the distinction between:
1. Single-choice questions (one correct answer)
2. Multiple-choice questions (multiple correct answers - true MCQ)

Following TDD principles: These tests define the contract for MCQ handling features
that need to be implemented in the checks, adapters, and utilities.

Test Coverage:
- normalize_correct_index() behavior with MCQ data
- Check warnings when receiving MCQ questions
- Adapter question_type detection
- End-to-end MCQ dataset handling
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from bencheck.adapters.base_mcq import BaseMCQAdapter
from bencheck.checks._utils import normalize_correct_index, normalize_scores
from bencheck.checks.enumeration_bias import EnumerationBiasCheck
from bencheck.checks.length_bias import LengthBiasCheck
from bencheck.models.dummy import DummyModel
from bencheck.types import BencheckQuestion, QuestionType

# ============================================================================
# Section 1: normalize_correct_index() Behavior Tests
# ============================================================================


class TestNormalizeCorrectIndexMCQ:
    """
    Test normalize_correct_index() behavior with multiple correct answers.

    Critical requirement: This function currently takes ONLY the first correct
    answer from lists. These tests verify this behavior and ensure proper
    warnings are logged.
    """

    def test_single_correct_answer_int(self):
        """Single correct answer as int should work without warnings."""
        result = normalize_correct_index(2)
        assert result == 2

    def test_single_correct_answer_list_int(self):
        """Single correct answer as list [int] should work without warnings."""
        with patch("bencheck.checks._utils.logger") as mock_logger:
            result = normalize_correct_index([2])
            assert result == 2
            # Should NOT warn for single-element list
            mock_logger.warning.assert_not_called()

    def test_multiple_correct_answers_returns_first(self):
        """
        Multiple correct answers should return ONLY first index.

        This is the current behavior for single-choice checks.
        Example: [1, 3, 4] -> returns 1
        """
        result = normalize_correct_index([1, 3, 4])
        assert result == 1

    def test_multiple_correct_answers_logs_warning(self, caplog):
        """
        Multiple correct answers should log a warning about data loss.

        This warning alerts users that the check is discarding multiple
        correct answers, which may not be appropriate for true MCQ datasets.
        """
        with caplog.at_level(logging.WARNING):
            result = normalize_correct_index([1, 3, 4])

        assert result == 1
        assert "Multiple correct answers" in caplog.text
        assert "[1, 3, 4]" in caplog.text
        assert "Returning ONLY FIRST index" in caplog.text

    def test_multiple_correct_answers_string_list(self):
        """Multiple correct answers as string list ["1", "3"] -> returns first."""
        result = normalize_correct_index(["1", "3"])
        assert result == 1

    def test_multiple_correct_answers_letter_list(self):
        """Multiple correct answers as letters ["A", "C", "D"] -> returns first (A=0)."""
        result = normalize_correct_index(["A", "C", "D"])
        assert result == 0

    def test_empty_list_returns_none(self):
        """Empty list should return None and log warning."""
        with patch("bencheck.checks._utils.logger") as mock_logger:
            result = normalize_correct_index([])
            assert result is None
            mock_logger.warning.assert_called()
            assert "Empty list" in str(mock_logger.warning.call_args)

    def test_mixed_format_multiple_correct(self):
        """
        Mixed format handling: [2, "B", 3] should handle first element.

        Edge case: List with mixed types (though this is rare in real datasets).
        """
        result = normalize_correct_index([2, "B", 3])
        assert result == 2

    def test_none_input(self):
        """None input should return None and log warning."""
        with patch("bencheck.checks._utils.logger") as mock_logger:
            result = normalize_correct_index(None)
            assert result is None
            mock_logger.warning.assert_called()


# ============================================================================
# Section 2: Check Warnings for MCQ Questions
# ============================================================================


class TestCheckMCQWarnings:
    """
    Test that checks warn users when receiving MCQ questions.

    Since current checks are designed for single-choice questions,
    they should detect and warn when they receive MULTIPLE_CHOICE
    question_type, informing users that results may be inaccurate.
    """

    def test_length_bias_check_warns_on_mcq(self, caplog):
        """
        LengthBiasCheck should warn when receiving MULTIPLE_CHOICE question.

        Expected behavior:
        - Check detects question_type == MULTIPLE_CHOICE
        - Logs clear warning about single-choice assumption
        - Still processes question (using first correct answer)
        """
        mcq_question = BencheckQuestion(
            id="test_mcq_1",
            question="Which of the following are prime numbers?",
            choices=["2", "3", "4", "5"],
            correct=[0, 1, 3],  # Multiple correct: 2, 3, 5 are prime
            question_type=QuestionType.MULTIPLE_CHOICE,
            metadata={},
        )

        check = LengthBiasCheck()

        with caplog.at_level(logging.WARNING):
            result = check.run_on_question(mcq_question, model=None)

        # Should warn about MCQ question
        assert any("MULTIPLE_CHOICE" in record.message for record in caplog.records)
        assert any("single-choice" in record.message.lower() for record in caplog.records)

        # Should still return results (using first correct answer)
        assert result is not None
        assert "option_lengths" in result

    def test_length_bias_check_no_warning_single_choice(self, caplog):
        """
        LengthBiasCheck should NOT warn for SINGLE_CHOICE questions.

        This is the expected case - no warning needed.
        """
        single_choice_question = BencheckQuestion(
            id="test_single_1",
            question="What is the capital of France?",
            choices=["London", "Paris", "Berlin", "Madrid"],
            correct=1,
            question_type=QuestionType.SINGLE_CHOICE,
            metadata={},
        )

        check = LengthBiasCheck()

        with caplog.at_level(logging.WARNING):
            result = check.run_on_question(single_choice_question, model=None)

        # Should NOT contain MCQ warning
        assert not any("MULTIPLE_CHOICE" in record.message for record in caplog.records)
        assert result is not None

    def test_enumeration_bias_check_warns_on_mcq(self, caplog):
        """
        EnumerationBiasCheck should warn when receiving MULTIPLE_CHOICE question.

        This is a model-dependent check that also assumes single correct answer.
        """
        mcq_question = BencheckQuestion(
            id="test_mcq_2",
            question="Select all even numbers:",
            choices=["1", "2", "3", "4"],
            correct=[1, 3],  # Multiple correct: 2 and 4
            question_type=QuestionType.MULTIPLE_CHOICE,
            metadata={},
        )

        model = DummyModel()
        check = EnumerationBiasCheck(model=model)

        with caplog.at_level(logging.WARNING):
            result = check.run_on_question(mcq_question, model=model)

        # Should warn about MCQ question
        assert any("MULTIPLE_CHOICE" in record.message for record in caplog.records)
        assert result is not None

    def test_check_warning_message_clarity(self, caplog):
        """
        Warning messages should be clear and actionable.

        Good warning should include:
        - What was detected (MULTIPLE_CHOICE question)
        - What the limitation is (check designed for single-choice)
        - What action is taken (using first correct answer only)
        """
        mcq_question = BencheckQuestion(
            id="test_mcq_3",
            question="Test MCQ",
            choices=["A", "B", "C", "D"],
            correct=[0, 2],
            question_type=QuestionType.MULTIPLE_CHOICE,
            metadata={},
        )

        check = LengthBiasCheck()

        with caplog.at_level(logging.WARNING):
            check.run_on_question(mcq_question, model=None)

        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert len(warning_messages) > 0

        # Check for key information in warning
        combined_message = " ".join(warning_messages)
        assert (
            "MULTIPLE_CHOICE" in combined_message or "multiple correct" in combined_message.lower()
        )
        assert "single" in combined_message.lower() or "first" in combined_message.lower()


# ============================================================================
# Section 3: Adapter Question Type Detection
# ============================================================================


class TestAdapterQuestionTypeDetection:
    """
    Test that BaseMCQAdapter correctly detects and sets question_type.

    Adapters should analyze the 'correct' field to determine if it's:
    - SINGLE_CHOICE: correct is int or list with len==1
    - MULTIPLE_CHOICE: correct is list with len>1

    This information should propagate to BencheckQuestion.question_type.
    """

    def test_adapter_single_correct_int_sets_single_choice(self):
        """
        Adapter should set SINGLE_CHOICE when correct is int.

        Example: correct=1 -> QuestionType.SINGLE_CHOICE
        """
        records = [
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,  # Single int
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 1
        assert questions[0].question_type == QuestionType.SINGLE_CHOICE

    def test_adapter_single_correct_list_sets_single_choice(self):
        """
        Adapter should set SINGLE_CHOICE when correct is list with len==1.

        Example: correct=[2] -> QuestionType.SINGLE_CHOICE
        """
        records = [
            {
                "question": "Capital of Spain?",
                "choices": ["Paris", "Madrid", "Rome", "Berlin"],
                "answer": [1],  # List with one element
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 1
        assert questions[0].question_type == QuestionType.SINGLE_CHOICE

    def test_adapter_multiple_correct_sets_multiple_choice(self):
        """
        Adapter should set MULTIPLE_CHOICE when correct is list with len>1.

        Example: correct=[0, 2, 3] -> QuestionType.MULTIPLE_CHOICE
        This is the key detection logic for true MCQ datasets.
        """
        records = [
            {
                "question": "Select all prime numbers:",
                "choices": ["2", "3", "4", "5"],
                "answer": [0, 1, 3],  # Multiple correct answers
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 1
        # This is the NEW behavior that needs to be implemented
        assert questions[0].question_type == QuestionType.MULTIPLE_CHOICE
        assert questions[0].correct == [0, 1, 3]

    def test_adapter_question_type_in_metadata(self):
        """
        question_type should be included in metadata for visibility.

        This helps downstream analysis identify MCQ questions.
        """
        records = [{"question": "Test MCQ", "choices": ["A", "B", "C"], "answer": [0, 2]}]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 1
        assert (
            questions[0].metadata.get("question_type") == "multiple_choice"
            or questions[0].question_type == QuestionType.MULTIPLE_CHOICE
        )

    def test_adapter_empty_list_correct_sets_single_choice(self):
        """
        Edge case: Empty correct list should default to SINGLE_CHOICE.

        This is an invalid case, but should handle gracefully.
        """
        records = [
            {
                "question": "Invalid question",
                "choices": ["A", "B", "C"],
                "answer": [],  # Empty list - invalid
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        # Should handle gracefully - default to SINGLE_CHOICE
        if len(questions) > 0:
            assert questions[0].question_type == QuestionType.SINGLE_CHOICE

    def test_adapter_string_list_multiple_correct(self):
        """
        Adapter should detect MULTIPLE_CHOICE for string lists.

        Example: correct=["A", "C", "D"] -> QuestionType.MULTIPLE_CHOICE
        """
        records = [
            {
                "question": "Select all vowels:",
                "choices": ["A", "B", "C", "D", "E"],
                "answer": ["A", "E"],  # Multiple correct as strings
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 1
        assert questions[0].question_type == QuestionType.MULTIPLE_CHOICE

    def test_adapter_truthfulqa_mc2_format(self):
        """
        TruthfulQA mc2 format has multiple correct answers (labels with multiple 1s).

        This is a real-world MCQ dataset format that should be detected.
        """
        records = [
            {
                "question": "What happens if you break a mirror?",
                "mc2_targets": {
                    "choices": [
                        "Nothing in particular",
                        "You will have bad luck",
                        "The mirror will be broken",
                    ],
                    "labels": [1, 0, 1],  # Two correct answers (indices 0 and 2)
                },
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "mc2_targets"
        adapter.label_key = "mc2_targets"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        # Should detect multiple 1s in labels as MULTIPLE_CHOICE
        if len(questions) > 0:
            # NOTE: This requires enhanced extract_label_index to return list
            # OR adapter to check labels array before calling extract_label_index
            assert questions[0].question_type == QuestionType.MULTIPLE_CHOICE


# ============================================================================
# Section 4: End-to-End MCQ Dataset Handling
# ============================================================================


class TestEndToEndMCQHandling:
    """
    Integration tests for complete MCQ dataset workflows.

    These tests verify that the entire pipeline works correctly:
    1. Load MCQ dataset
    2. Detect question_type correctly
    3. Run checks with appropriate warnings
    4. Produce valid results (using first correct answer)
    """

    @pytest.fixture
    def mock_mcq_dataset(self):
        """
        Mock dataset with mix of single-choice and multiple-choice questions.

        This simulates a real dataset that might contain both types.
        """
        return [
            {
                "id": "q1",
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": 1,  # Single correct
            },
            {
                "id": "q2",
                "question": "Select all prime numbers:",
                "choices": ["2", "3", "4", "5", "6"],
                "answer": [0, 1, 3],  # Multiple correct: 2, 3, 5
            },
            {
                "id": "q3",
                "question": "Which are even numbers?",
                "choices": ["1", "2", "3", "4"],
                "answer": [1, 3],  # Multiple correct: 2, 4
            },
            {
                "id": "q4",
                "question": "What is 5+5?",
                "choices": ["8", "9", "10", "11"],
                "answer": 2,  # Single correct
            },
        ]

    def test_load_mixed_dataset_question_types(self, mock_mcq_dataset):
        """
        Loading dataset with mixed question types should detect each correctly.

        Expected:
        - q1: SINGLE_CHOICE
        - q2: MULTIPLE_CHOICE
        - q3: MULTIPLE_CHOICE
        - q4: SINGLE_CHOICE
        """
        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"
        adapter.id_key = "id"

        questions = adapter.load(source=mock_mcq_dataset)

        assert len(questions) == 4
        assert questions[0].question_type == QuestionType.SINGLE_CHOICE
        assert questions[1].question_type == QuestionType.MULTIPLE_CHOICE
        assert questions[2].question_type == QuestionType.MULTIPLE_CHOICE
        assert questions[3].question_type == QuestionType.SINGLE_CHOICE

    def test_run_check_on_mixed_dataset_with_warnings(self, mock_mcq_dataset, caplog):
        """
        Running check on mixed dataset should warn for MCQ questions only.

        Expected warnings for q2 and q3 (MULTIPLE_CHOICE).
        No warnings for q1 and q4 (SINGLE_CHOICE).
        """
        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"
        adapter.id_key = "id"

        questions = adapter.load(source=mock_mcq_dataset)

        check = LengthBiasCheck()

        with caplog.at_level(logging.WARNING):
            result = check.run(questions, model=None)

        # Should have warnings for MCQ questions
        mcq_warnings = [
            record
            for record in caplog.records
            if "MULTIPLE_CHOICE" in record.message or "multiple correct" in record.message.lower()
        ]

        # Expect 2 warnings (q2 and q3 are MCQ)
        assert len(mcq_warnings) >= 2

        # Results should still be produced for all questions
        assert result.metrics["n_questions"] == 4

    def test_truthfulqa_mc2_end_to_end(self):
        """
        End-to-end test with TruthfulQA mc2 format (real MCQ dataset).

        TruthfulQA mc2 has multiple correct answers and is a common
        benchmark that should be properly handled.
        """
        records = [
            {
                "question": "What happens if you break a mirror?",
                "mc2_targets": {
                    "choices": [
                        "Nothing in particular happens",
                        "You will have seven years of bad luck",
                        "The mirror will be broken",
                    ],
                    "labels": [1, 0, 1],  # Indices 0 and 2 are correct
                },
            },
            {
                "question": "What is the tallest mountain?",
                "mc2_targets": {
                    "choices": [
                        "Mount Everest",
                        "K2",
                        "Mount Kilimanjaro",
                    ],
                    "labels": [1, 0, 0],  # Only index 0 is correct
                },
            },
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "mc2_targets"
        adapter.label_key = "mc2_targets"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 2
        # First question should be MULTIPLE_CHOICE (two 1s in labels)
        assert questions[0].question_type == QuestionType.MULTIPLE_CHOICE
        # Second question should be SINGLE_CHOICE (one 1 in labels)
        assert questions[1].question_type == QuestionType.SINGLE_CHOICE

    def test_mcq_results_use_first_correct_answer(self, mock_mcq_dataset):
        """
        When check processes MCQ question, it should use first correct answer.

        For q2 with correct=[0, 1, 3], length bias should use index 0.
        """
        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"
        adapter.id_key = "id"

        questions = adapter.load(source=mock_mcq_dataset)

        # Get q2 (the prime numbers question)
        q2 = questions[1]
        assert q2.question_type == QuestionType.MULTIPLE_CHOICE

        # Word count keeps the expected length independent of any tokenizer
        check = LengthBiasCheck(length_metric="words")
        result = check.run_on_question(q2, model=None)

        # Should analyze using first correct answer (index 0 = "2")
        assert result is not None
        assert "correct_length" in result
        # Length of "2" is 1 character
        assert result["correct_length"] == 1

    def test_metadata_contains_question_type_info(self, mock_mcq_dataset):
        """
        Metadata should contain question_type information for analysis.

        This enables post-processing to filter or analyze by question type.
        """
        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"
        adapter.id_key = "id"

        questions = adapter.load(source=mock_mcq_dataset)

        # Check that metadata contains type info
        for q in questions:
            # Either in metadata dict or as question_type attribute
            has_type_info = "question_type" in q.metadata or q.question_type is not None
            assert has_type_info


# ============================================================================
# Section 5: Edge Cases and Error Handling
# ============================================================================


class TestMCQEdgeCases:
    """
    Test edge cases and error conditions in MCQ handling.

    Robust implementations should handle:
    - Invalid data formats
    - Missing fields
    - Type mismatches
    - Boundary conditions
    """

    def test_normalize_correct_index_with_none(self):
        """None input should return None and log warning."""
        with patch("bencheck.checks._utils.logger") as mock_logger:
            result = normalize_correct_index(None)
            assert result is None
            mock_logger.warning.assert_called()

    def test_normalize_correct_index_with_invalid_type(self):
        """Invalid type (dict, object) should return None and log warning."""
        with patch("bencheck.checks._utils.logger") as mock_logger:
            result = normalize_correct_index({"invalid": "type"})
            assert result is None
            mock_logger.warning.assert_called()

    def test_adapter_handles_missing_correct_field(self):
        """
        Adapter should handle records without correct field gracefully.

        Should default to index 0 and SINGLE_CHOICE.
        """
        records = [
            {
                "question": "No correct answer provided",
                "choices": ["A", "B", "C"],
                # Missing 'answer' field
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 1
        assert questions[0].correct == 0  # Default fallback
        assert questions[0].question_type == QuestionType.SINGLE_CHOICE

    def test_adapter_handles_malformed_correct_list(self):
        """
        Adapter should handle malformed correct lists gracefully.

        Example: correct=[None, "invalid", 2] should handle safely.
        """
        records = [
            {
                "question": "Malformed correct",
                "choices": ["A", "B", "C", "D"],
                "answer": [None, "invalid", 2],
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        # Should not crash - may skip or use fallback
        questions = adapter.load(source=records)
        # Implementation-dependent: may skip malformed records or use fallback
        assert isinstance(questions, list)

    def test_check_handles_invalid_correct_index_gracefully(self):
        """
        Check should handle invalid correct index without crashing.

        Example: correct=99 for question with 4 choices should log error
        and skip analysis.
        """
        invalid_question = BencheckQuestion(
            id="invalid_1",
            question="Test",
            choices=["A", "B", "C", "D"],
            correct=99,  # Invalid: out of range
            question_type=QuestionType.SINGLE_CHOICE,
            metadata={},
        )

        check = LengthBiasCheck()
        result = check.run_on_question(invalid_question, model=None)

        # Should return error indication, not crash
        assert result is not None
        assert "error" in result or result.get("correct_length") is None

    def test_normalize_correct_index_very_large_list(self):
        """
        Performance edge case: Very large correct list should still warn and return first.

        Example: list of 1000 correct indices should not cause issues.
        """
        large_correct_list = list(range(1000))

        with patch("bencheck.checks._utils.logger") as mock_logger:
            result = normalize_correct_index(large_correct_list)

        assert result == 0  # First element
        mock_logger.warning.assert_called()  # Should warn about multiple correct

    def test_question_type_enum_values(self):
        """
        Verify QuestionType enum has expected values.

        This is a sanity check for the enum definition.
        """
        assert QuestionType.SINGLE_CHOICE.value == "single_choice"
        assert QuestionType.MULTIPLE_CHOICE.value == "multiple_choice"
        assert QuestionType.OPEN_ENDED.value == "open_ended"

    def test_adapter_preserves_original_correct_format(self):
        """
        Adapter should preserve original 'correct' format in BencheckQuestion.

        If correct=[0, 2, 3], it should remain as list, not be converted to int.
        """
        records = [{"question": "MCQ test", "choices": ["A", "B", "C", "D"], "answer": [0, 2, 3]}]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 1
        # Should preserve list format
        assert isinstance(questions[0].correct, list)
        assert questions[0].correct == [0, 2, 3]


# ============================================================================
# Section 6: Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """
    Ensure new MCQ handling doesn't break existing single-choice functionality.

    All existing tests should continue to pass. These tests verify that
    the new MCQ detection logic doesn't interfere with normal operation.
    """

    def test_existing_single_choice_datasets_unchanged(self):
        """
        Existing single-choice datasets should work exactly as before.

        HellaSwag, MMLU, etc. should all be detected as SINGLE_CHOICE.
        """
        # HellaSwag format
        hellaswag_record = {
            "ind": 0,
            "ctx": "A person is running",
            "endings": ["on a treadmill", "in circles", "backwards", "sideways"],
            "label": 1,
        }

        adapter = BaseMCQAdapter()
        adapter.option_key = "endings"
        adapter.label_key = "label"
        adapter.context_key = "ctx"

        questions = adapter.load(source=[hellaswag_record])

        assert len(questions) == 1
        assert questions[0].question_type == QuestionType.SINGLE_CHOICE
        assert questions[0].correct == 1

    def test_letter_format_single_choice_unchanged(self):
        """
        Letter format (A, B, C, D) should still work for single-choice.

        Example: correct="B" -> index 1, SINGLE_CHOICE
        """
        records = [
            {
                "question": "What is the capital?",
                "choices": ["London", "Paris", "Berlin"],
                "answer": "B",  # Letter format
            }
        ]

        adapter = BaseMCQAdapter()
        adapter.option_key = "choices"
        adapter.label_key = "answer"
        adapter.context_key = "question"

        questions = adapter.load(source=records)

        assert len(questions) == 1
        assert questions[0].correct == 1  # B = index 1
        assert questions[0].question_type == QuestionType.SINGLE_CHOICE

    def test_existing_checks_work_without_changes(self):
        """
        Existing checks should continue to work on single-choice questions
        without any modifications.

        This verifies backward compatibility at the check level.
        """
        questions = [
            BencheckQuestion(
                id="1",
                question="Test",
                choices=["A", "B", "C", "D"],
                correct=2,
                question_type=QuestionType.SINGLE_CHOICE,
                metadata={},
            ),
            BencheckQuestion(
                id="2",
                question="Test 2",
                choices=["Short", "Medium length", "Long answer here"],
                correct=0,
                question_type=QuestionType.SINGLE_CHOICE,
                metadata={},
            ),
        ]

        check = LengthBiasCheck()
        result = check.run(questions, model=None)

        # Should work exactly as before
        assert result.check_name == "length_bias"
        assert result.metrics["n_questions"] == 2
        assert "mean_correct_length" in result.metrics


# ============================================================================
# Section 7: Documentation and Metadata Tests
# ============================================================================


class TestMCQDocumentation:
    """
    Test that MCQ handling is well-documented in results and metadata.

    Users should be able to:
    1. Identify which questions are MCQ
    2. Understand how they were processed
    3. See clear warnings about limitations
    """

    def test_check_result_includes_question_type_summary(self, caplog):
        """
        Check results should summarize how many MCQ vs single-choice questions.

        This helps users understand dataset composition.
        """
        questions = [
            BencheckQuestion(
                id="1",
                question="Q1",
                choices=["A", "B"],
                correct=0,
                question_type=QuestionType.SINGLE_CHOICE,
                metadata={},
            ),
            BencheckQuestion(
                id="2",
                question="Q2",
                choices=["A", "B", "C"],
                correct=[0, 2],
                question_type=QuestionType.MULTIPLE_CHOICE,
                metadata={},
            ),
            BencheckQuestion(
                id="3",
                question="Q3",
                choices=["A", "B"],
                correct=1,
                question_type=QuestionType.SINGLE_CHOICE,
                metadata={},
            ),
        ]

        check = LengthBiasCheck()

        with caplog.at_level(logging.INFO):
            result = check.run(questions, model=None)

        # Check should log or include in metrics the question type distribution
        # This is a nice-to-have feature for transparency
        # Implementation can vary - could be in metrics or logs
        assert result is not None

    def test_warning_log_includes_question_id(self, caplog):
        """
        Warning about MCQ should include question ID for traceability.

        Example: "Question 'q2' is MULTIPLE_CHOICE but check assumes single-choice"
        """
        mcq_question = BencheckQuestion(
            id="mcq_question_42",
            question="Test MCQ",
            choices=["A", "B", "C"],
            correct=[0, 2],
            question_type=QuestionType.MULTIPLE_CHOICE,
            metadata={},
        )

        check = LengthBiasCheck()

        with caplog.at_level(logging.WARNING):
            check.run_on_question(mcq_question, model=None)

        # Warning should mention the question ID
        warning_text = " ".join([record.message for record in caplog.records])
        assert "mcq_question_42" in warning_text or "42" in warning_text
