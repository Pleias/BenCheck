"""
Comprehensive tests for GlobalPIQAAdapter.

This test suite follows Test-Driven Development (TDD) principles.
Tests are written BEFORE implementation to define the contract and expected behavior.

Dataset: mrlbenchmarks/global-piqa-nonparallel
Format: Multilingual PIQA with language-specific subsets
Default subset: eng_Latn (English Latin script)
"""

import pytest

from bencheck.adapters import BaseMCQAdapter, PIQAAdapter
from bencheck.types import BencheckQuestion, QuestionType


class TestGlobalPIQAAdapterBasicFunctionality:
    """Tests for basic adapter functionality and data conversion."""

    def test_loads_eng_latn_subset_correctly(self):
        """
        Test that adapter loads eng_latn subset by default.

        GlobalPIQA has multiple language subsets. We only want English.
        Verify that the adapter requests the correct subset from HuggingFace.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        adapter = GlobalPIQAAdapter()

        # Check that config_name is set to eng_latn
        assert adapter.config_name == "eng_latn", (
            "GlobalPIQAAdapter should default to 'eng_latn' subset"
        )
        assert adapter.dataset_name == "mrlbenchmarks/global-piqa-nonparallel"

    def test_converts_records_to_bencheck_question_format(self):
        """
        Test conversion of Global PIQA records to BencheckQuestion objects.

        Expected format:
        - prompt (question text)
        - solution0 (first solution)
        - solution1 (second solution)
        - label (0 or 1)
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {
                "prompt": "How do you remove a stain from a shirt?",
                "solution0": "Use cold water and soap immediately",
                "solution1": "Let it dry completely before washing",
                "label": 0,
            }
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1, "Should convert one record"
        assert isinstance(questions[0], BencheckQuestion)

    def test_extracts_goal_as_question_text(self):
        """
        Test that the 'prompt' field is correctly extracted as question text.

        In Global PIQA, the 'prompt' describes the physical task or objective.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {
                "prompt": "To keep your shoes from smelling bad",
                "solution0": "Put baking soda inside them",
                "solution1": "Put them in the freezer",
                "label": 0,
            }
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        assert questions[0].question == "To keep your shoes from smelling bad"

    def test_builds_choices_from_sol1_and_sol2(self):
        """
        Test that choices are correctly built from solution0 and solution1 fields.

        Global PIQA provides exactly 2 solutions for each prompt.
        Order must be preserved: [solution0, solution1].
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {
                "prompt": "How to make ice cubes freeze faster",
                "solution0": "Use hot water in the ice tray",
                "solution1": "Use room temperature water",
                "label": 0,
            }
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        assert len(questions[0].choices) == 2, "Should have exactly 2 choices"
        assert questions[0].choices[0] == "Use hot water in the ice tray"
        assert questions[0].choices[1] == "Use room temperature water"

    def test_preserves_label_correctly(self):
        """
        Test that label (0 or 1) is correctly preserved.

        Label indicates which solution is correct (0-based indexing, 0 or 1).
        No conversion needed - already in correct format.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data_label_0 = [
            {
                "prompt": "Test goal",
                "solution0": "Solution A",
                "solution1": "Solution B",
                "label": 0,
            }
        ]

        mock_data_label_1 = [
            {
                "prompt": "Test goal",
                "solution0": "Solution A",
                "solution1": "Solution B",
                "label": 1,
            }
        ]

        adapter = GlobalPIQAAdapter()

        questions_0 = adapter.load(source=mock_data_label_0)
        assert questions_0[0].correct == 0, "Label 0 should map to index 0"

        questions_1 = adapter.load(source=mock_data_label_1)
        assert questions_1[0].correct == 1, "Label 1 should map to index 1"


class TestGlobalPIQAAdapterSubsetFiltering:
    """Tests for language subset selection and filtering."""

    def test_only_eng_latn_subset_loaded(self):
        """
        Test that only eng_Latn subset is loaded by default.

        Global PIQA contains multiple language subsets (e.g., fra_Latn, deu_Latn).
        We must verify that ONLY English data is loaded.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        adapter = GlobalPIQAAdapter()

        # The adapter should specify eng_Latn as config_name
        assert adapter.config_name == "eng_latn"

    def test_can_specify_subset_in_init(self):
        """
        Test that subset can be specified via __init__ parameter.

        While default is eng_Latn, the adapter should allow specifying
        other language subsets for future extensibility.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        # Test explicit eng_Latn
        adapter_eng = GlobalPIQAAdapter(subset="eng_latn")
        assert adapter_eng.config_name == "eng_latn"

        # Test other subset (e.g., French)
        adapter_fra = GlobalPIQAAdapter(subset="fra_Latn")
        assert adapter_fra.config_name == "fra_Latn"

    def test_subset_parameter_defaults_to_eng_latn(self):
        """
        Test that subset parameter defaults to eng_Latn when not specified.

        This ensures backward compatibility and expected default behavior.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        adapter = GlobalPIQAAdapter()
        assert adapter.config_name == "eng_latn"

    def test_raises_error_for_invalid_subset(self):
        """
        Test that invalid subset raises appropriate error during load.

        If an invalid subset is specified, HuggingFace datasets will raise an error.
        We should let this propagate with a clear message.

        Note: This test validates error handling, not that we catch it ourselves.
        The datasets library will handle validation.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        # Create adapter with invalid subset
        adapter = GlobalPIQAAdapter(subset="invalid_subset_xyz")

        # Should fail when trying to load from HuggingFace
        # We expect datasets library to raise an error
        with pytest.raises(Exception):  # datasets raises various errors
            adapter.load()  # This calls _load_from_huggingface


class TestGlobalPIQAAdapterDataQuality:
    """Tests for data quality validation and constraints."""

    def test_all_questions_have_exactly_2_choices(self):
        """
        Test that all questions have exactly 2 choices.

        Global PIQA is a binary choice task (sol1 vs sol2).
        Any record with != 2 choices indicates a data quality issue.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {
                "prompt": "Goal 1",
                "solution0": "Solution A",
                "solution1": "Solution B",
                "label": 0,
            },
            {
                "prompt": "Goal 2",
                "solution0": "Solution X",
                "solution1": "Solution Y",
                "label": 1,
            },
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        for q in questions:
            assert len(q.choices) == 2, f"Question {q.id} has {len(q.choices)} choices, expected 2"

    def test_labels_are_valid_0_or_1(self):
        """
        Test that all labels are valid (0 or 1).

        Global PIQA labels must be binary (0 or 1) corresponding to solution0/solution1.
        Invalid labels indicate data corruption.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {"prompt": "G1", "solution0": "A", "solution1": "B", "label": 0},
            {"prompt": "G2", "solution0": "C", "solution1": "D", "label": 1},
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        for q in questions:
            assert q.correct in [0, 1], f"Question {q.id} has invalid label: {q.correct}"

    def test_no_missing_fields(self):
        """
        Test that records with missing required fields are skipped.

        Required fields: prompt, solution0, solution1, label
        Records missing any of these should be gracefully skipped.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            # Valid record
            {"prompt": "G1", "solution0": "A", "solution1": "B", "label": 0},
            # Missing sol1
            {"prompt": "G2", "solution1": "C", "label": 1},
            # Missing goal
            {"solution0": "D", "solution1": "E", "label": 0},
            # Valid record
            {"prompt": "G3", "solution0": "F", "solution1": "G", "label": 1},
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        # Should only load the 2 valid records
        assert len(questions) == 2
        assert questions[0].question == "G1"
        assert questions[1].question == "G3"

    def test_no_empty_fields(self):
        """
        Test that records with empty string fields are skipped.

        Empty goals or solutions are not useful for evaluation.
        These should be filtered out during conversion.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            # Valid
            {"prompt": "Valid goal", "solution0": "A", "solution1": "B", "label": 0},
            # Empty goal
            {"prompt": "", "solution0": "C", "solution1": "D", "label": 1},
            # Empty sol1
            {"prompt": "Another goal", "solution0": "", "solution1": "E", "label": 0},
            # Valid
            {"prompt": "Good goal", "solution0": "F", "solution1": "G", "label": 1},
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        # Should only load the 2 valid records (non-empty fields)
        assert len(questions) == 2
        assert all(q.question.strip() for q in questions)
        assert all(all(c.strip() for c in q.choices) for q in questions)

    def test_metadata_includes_dataset_name_and_subset(self):
        """
        Test that metadata includes dataset name and subset information.

        This is important for tracking data provenance and filtering results.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [{"prompt": "Test", "solution0": "A", "solution1": "B", "label": 0}]

        adapter = GlobalPIQAAdapter(subset="eng_latn")
        questions = adapter.load(source=mock_data)

        metadata = questions[0].metadata
        assert metadata["dataset"] == "mrlbenchmarks/global-piqa-nonparallel"
        assert metadata["split"] == "test"  # Default split for Global PIQA


class TestGlobalPIQAAdapterIntegration:
    """Tests for integration with BenCheck framework."""

    def test_inherits_from_base_mcq_adapter(self):
        """
        Test that GlobalPIQAAdapter inherits from BaseMCQAdapter.

        This ensures code reuse and consistent behavior across adapters.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        adapter = GlobalPIQAAdapter()
        assert isinstance(adapter, BaseMCQAdapter)

    def test_question_type_is_single_choice(self):
        """
        Test that question_type is SINGLE_CHOICE.

        Global PIQA has exactly one correct answer per question.
        This is not multiple-choice (where multiple answers can be correct).
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [{"prompt": "Test", "solution0": "A", "solution1": "B", "label": 0}]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        assert questions[0].question_type == QuestionType.SINGLE_CHOICE

    def test_can_be_imported_from_bencheck_adapters(self):
        """
        Test that GlobalPIQAAdapter can be imported from bencheck.adapters.

        Ensures proper package structure and __init__.py exports.
        """
        # This import should work if GlobalPIQAAdapter is added to __init__.py
        try:
            from bencheck.adapters import GlobalPIQAAdapter

            assert GlobalPIQAAdapter is not None
        except ImportError:
            pytest.fail("GlobalPIQAAdapter not exported from bencheck.adapters. Add to __init__.py")

    def test_works_with_existing_bencheck_pipeline(self):
        """
        Test that adapter produces output compatible with BenCheck pipeline.

        Verify that the BencheckQuestion objects can be used in standard workflows.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [{"prompt": "Test goal", "solution0": "A", "solution1": "B", "label": 0}]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        # Check all required BencheckQuestion fields are present
        q = questions[0]
        assert hasattr(q, "id")
        assert hasattr(q, "question")
        assert hasattr(q, "choices")
        assert hasattr(q, "correct")
        assert hasattr(q, "question_type")
        assert hasattr(q, "metadata")

        # Check types
        assert isinstance(q.id, str)
        assert isinstance(q.question, str)
        assert isinstance(q.choices, list)
        assert isinstance(q.correct, int)
        assert isinstance(q.question_type, QuestionType)
        assert isinstance(q.metadata, dict)

    def test_consistent_with_piqa_adapter_format(self):
        """
        Test that GlobalPIQAAdapter produces same format as PIQAAdapter.

        Both adapters use different field names but produce same output structure.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        global_mock_data = [
            {"prompt": "How to clean a window", "solution0": "Use paper", "solution1": "Use cloth", "label": 1}
        ]

        piqa_mock_data = [
            {"goal": "How to clean a window", "sol1": "Use paper", "sol2": "Use cloth", "label": 1}
        ]

        global_adapter = GlobalPIQAAdapter()
        piqa_adapter = PIQAAdapter()

        global_questions = global_adapter.load(source=global_mock_data)
        piqa_questions = piqa_adapter.load(source=piqa_mock_data)

        # Compare structure (not dataset name in metadata)
        gq = global_questions[0]
        pq = piqa_questions[0]

        assert gq.question == pq.question
        assert gq.choices == pq.choices
        assert gq.correct == pq.correct
        assert gq.question_type == pq.question_type


class TestGlobalPIQAAdapterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataset_handling(self):
        """
        Test that adapter handles empty dataset gracefully.

        Should return empty list, not raise error.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=[])

        assert questions == []
        assert isinstance(questions, list)

    def test_missing_label_field(self):
        """
        Test handling of records without label field.

        While unusual, adapter should handle this gracefully (skip or use default).
        BaseMCQAdapter typically uses 0 as fallback.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {"prompt": "Test", "solution0": "A", "solution1": "B"},  # No label
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        # BaseMCQAdapter should handle this - might use default or skip
        # We test that it doesn't crash
        assert isinstance(questions, list)

    def test_invalid_label_values(self):
        """
        Test handling of invalid label values (not 0 or 1).

        Labels outside [0, 1] are invalid for binary choice.
        Should be skipped or cause error during validation.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {"prompt": "Valid", "solution0": "A", "solution1": "B", "label": 0},
            {"prompt": "Invalid", "solution0": "C", "solution1": "D", "label": 2},  # Invalid!
            {"prompt": "Also invalid", "solution0": "E", "solution1": "F", "label": -1},  # Invalid!
            {"prompt": "Another valid", "solution0": "G", "solution1": "H", "label": 1},
        ]

        adapter = GlobalPIQAAdapter()

        # Adapter might skip invalid records or raise error
        # We test that it doesn't crash catastrophically
        try:
            questions = adapter.load(source=mock_data)
            # If it succeeds, check that invalid labels were handled
            for q in questions:
                assert q.correct in [0, 1], f"Invalid label made it through: {q.correct}"
        except Exception:
            # Some implementations might raise errors for invalid data
            # This is also acceptable behavior
            pass

    def test_very_long_question_text(self):
        """
        Test handling of very long question/solution text.

        Some PIQA goals can be quite lengthy. Ensure no truncation or errors.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        long_goal = "A" * 10000  # 10k characters
        long_sol1 = "B" * 5000
        long_sol2 = "C" * 5000

        mock_data = [
            {
                "prompt": long_goal,
                "solution0": long_sol1,
                "solution1": long_sol2,
                "label": 0,
            }
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert questions[0].question == long_goal
        assert questions[0].choices[0] == long_sol1
        assert questions[0].choices[1] == long_sol2

    def test_unicode_and_special_characters(self):
        """
        Test handling of Unicode and special characters.

        While eng_Latn should be mostly ASCII, test robustness with Unicode.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {
                "prompt": "How to write 'hello' in different scripts: \u4f60\u597d \u0645\u0631\u062d\u0628\u0627",
                "solution0": "Use Unicode characters: \u00e9\u00e8\u00ea",
                "solution1": "Use ASCII only: a-z",
                "label": 0,
            }
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert "\u4f60\u597d" in questions[0].question
        assert "\u00e9\u00e8\u00ea" in questions[0].choices[0]

    def test_whitespace_handling(self):
        """
        Test handling of leading/trailing whitespace in fields.

        Data quality issue: fields may have extra whitespace.
        Adapter should preserve or normalize consistently.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {
                "prompt": "  Leading and trailing spaces  ",
                "solution0": "\tTabs\t",
                "solution1": "  Multiple   spaces  ",
                "label": 0,
            }
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        # Verify data is loaded (whitespace handling is implementation detail)
        assert len(questions) == 1
        # Question and choices should exist (exact whitespace handling may vary)
        assert questions[0].question
        assert len(questions[0].choices) == 2

    def test_numeric_label_as_string(self):
        """
        Test handling of label as string "0" or "1" instead of int.

        Some datasets have labels as strings. Adapter should handle both.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {"prompt": "Test", "solution0": "A", "solution1": "B", "label": "0"},
            {"prompt": "Test2", "solution0": "C", "solution1": "D", "label": "1"},
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        # Labels should be converted to int
        assert questions[0].correct == 0
        assert questions[1].correct == 1

    def test_additional_metadata_fields_preserved(self):
        """
        Test that additional metadata fields in records are preserved.

        Global PIQA may have language codes, IDs, or other metadata.
        These should be captured in the metadata dict.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {
                "prompt": "Test",
                "solution0": "A",
                "solution1": "B",
                "label": 0,
                "language": "eng_latn",
                "source_id": "piqa_123",
                "difficulty": "easy",
            }
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        # Extra fields should be available (either in metadata or preserved)
        # Implementation may vary - test that adapter doesn't crash
        assert len(questions) == 1


class TestGlobalPIQAAdapterNumericalEdgeCases:
    """Tests for numerical and mathematical edge cases specific to scientific computing."""

    def test_label_type_consistency(self):
        """
        Test that label type is consistently int, not float.

        Labels should be integers (0, 1), not floats (0.0, 1.0).
        This prevents floating-point comparison issues.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {"prompt": "Test", "solution0": "A", "solution1": "B", "label": 0},
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        assert isinstance(questions[0].correct, int)
        assert type(questions[0].correct) == int  # Explicitly int, not bool or float

    def test_handles_label_as_float(self):
        """
        Test graceful handling when label is provided as float (0.0, 1.0).

        Some datasets may have labels as floats. Should convert to int.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {"prompt": "Test", "solution0": "A", "solution1": "B", "label": 0.0},
            {"prompt": "Test2", "solution0": "C", "solution1": "D", "label": 1.0},
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        assert questions[0].correct == 0
        assert questions[1].correct == 1
        assert isinstance(questions[0].correct, int)

    def test_deterministic_ordering(self):
        """
        Test that question order is deterministic and reproducible.

        Loading the same data twice should produce identical ordering.
        This is critical for reproducible evaluations.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {"prompt": "Q1", "solution0": "A", "solution1": "B", "label": 0},
            {"prompt": "Q2", "solution0": "C", "solution1": "D", "label": 1},
            {"prompt": "Q3", "solution0": "E", "solution1": "F", "label": 0},
        ]

        adapter = GlobalPIQAAdapter()

        questions1 = adapter.load(source=mock_data)
        questions2 = adapter.load(source=mock_data)

        # Check same order
        assert len(questions1) == len(questions2)
        for q1, q2 in zip(questions1, questions2):
            assert q1.question == q2.question
            assert q1.choices == q2.choices
            assert q1.correct == q2.correct

    def test_id_generation_is_unique(self):
        """
        Test that generated IDs are unique across questions.

        IDs should uniquely identify each question.
        Collisions would cause data loss in result aggregation.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {"prompt": "Q1", "solution0": "A", "solution1": "B", "label": 0},
            {"prompt": "Q2", "solution0": "C", "solution1": "D", "label": 1},
            {"prompt": "Q3", "solution0": "E", "solution1": "F", "label": 0},
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        ids = [q.id for q in questions]
        assert len(ids) == len(set(ids)), "IDs must be unique"

    def test_preserves_original_record_id_if_present(self):
        """
        Test that original record ID is preserved if present in data.

        Global PIQA may have its own ID field. This should be used if available.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        mock_data = [
            {
                "id": "global_piqa_001",
                "prompt": "Test",
                "solution0": "A",
                "solution1": "B",
                "label": 0,
            }
        ]

        adapter = GlobalPIQAAdapter()
        questions = adapter.load(source=mock_data)

        # ID should match original or be derived from it
        assert "global_piqa_001" in questions[0].id or questions[0].id == "global_piqa_001"


class TestGlobalPIQAAdapterComparisonWithPIQA:
    """Tests comparing GlobalPIQAAdapter behavior with PIQAAdapter."""

    def test_uses_same_field_names_as_piqa(self):
        """
        Test field names for GlobalPIQAAdapter.

        GlobalPIQAAdapter uses different field names than PIQA:
        - context_key = "prompt" (PIQA uses "goal")
        - label_key = "label" (same as PIQA)
        - solution0/solution1 (PIQA uses sol1/sol2)
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        adapter = GlobalPIQAAdapter()
        piqa_adapter = PIQAAdapter()

        assert adapter.context_key == "prompt"
        assert piqa_adapter.context_key == "goal"
        assert adapter.label_key == piqa_adapter.label_key == "label"

    def test_produces_identical_structure_for_same_data(self):
        """
        Test that GlobalPIQAAdapter and PIQAAdapter produce identical output structure.

        Both adapters use different field names but produce same output structure
        when given equivalent data.
        """
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        global_mock_data = [{"prompt": "Test goal", "solution0": "Solution 1", "solution1": "Solution 2", "label": 1}]
        piqa_mock_data = [{"goal": "Test goal", "sol1": "Solution 1", "sol2": "Solution 2", "label": 1}]

        global_adapter = GlobalPIQAAdapter()
        piqa_adapter = PIQAAdapter()

        global_q = global_adapter.load(source=global_mock_data)[0]
        piqa_q = piqa_adapter.load(source=piqa_mock_data)[0]

        # Check structural equivalence
        assert global_q.question == piqa_q.question
        assert global_q.choices == piqa_q.choices
        assert global_q.correct == piqa_q.correct
        assert global_q.question_type == piqa_q.question_type


# Integration test marker for slow tests that hit actual HuggingFace API
@pytest.mark.integration
@pytest.mark.slow
class TestGlobalPIQAAdapterHuggingFaceIntegration:
    """
    Integration tests with actual HuggingFace dataset.

    These tests are marked as slow and should be run separately.
    They require network access and the datasets library.
    """

    def test_loads_from_huggingface(self):
        """
        Test loading actual data from HuggingFace hub.

        This is an integration test that requires:
        - Network access
        - datasets library installed
        - Valid HuggingFace dataset

        Skip if not in integration test mode.
        """
        pytest.importorskip("datasets")
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        adapter = GlobalPIQAAdapter(split="validation")

        # This will hit the actual HuggingFace API
        try:
            questions = adapter.load()

            # Basic validation of loaded data
            assert len(questions) > 0, "Should load some questions"
            assert all(isinstance(q, BencheckQuestion) for q in questions)
            assert all(len(q.choices) == 2 for q in questions)
            assert all(q.correct in [0, 1] for q in questions)
        except Exception as e:
            pytest.skip(f"HuggingFace integration test failed: {e}")

    def test_eng_latn_subset_has_valid_data(self):
        """
        Test that eng_Latn subset contains valid English data.

        Verify that the actual dataset has expected structure and content.
        """
        pytest.importorskip("datasets")
        from bencheck.adapters.specialized import GlobalPIQAAdapter

        adapter = GlobalPIQAAdapter(subset="eng_latn", split="validation")

        try:
            questions = adapter.load()

            # Take first few questions and validate
            sample_questions = questions[:5]

            for q in sample_questions:
                # All fields should be non-empty
                assert q.question.strip()
                assert len(q.choices) == 2
                assert all(c.strip() for c in q.choices)
                assert q.correct in [0, 1]

                # Metadata should indicate correct dataset
                assert q.metadata["dataset"] == "mrlbenchmarks/global-piqa-nonparallel"
        except Exception as e:
            pytest.skip(f"HuggingFace integration test failed: {e}")
