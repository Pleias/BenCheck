"""Tests for specialized MCQ adapters (MMLU, TruthfulQA, etc.)."""

import pytest

from bencheck.adapters import (
    BaseMCQAdapter,
    CommonsenseQAAdapter,
    MMLUAdapter,
    TruthfulQAAdapter,
    WinograndeAdapter,
)
from bencheck.types import BencheckQuestion


class TestBaseMCQAdapter:
    """Tests for BaseMCQAdapter base class."""

    def test_requires_dataset_name(self):
        """Base adapter should raise error if dataset_name not set."""
        adapter = BaseMCQAdapter()

        with pytest.raises(ValueError, match="requires dataset_name"):
            adapter.load()

    def test_can_set_dataset_name_in_init(self):
        """Dataset name can be provided in __init__."""
        mock_data = [{"question": "Test?", "choices": ["A", "B"], "answer": 0}]

        adapter = BaseMCQAdapter(dataset_name="test/dataset")
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert questions[0].metadata["dataset"] == "test/dataset"


class TestMMLUAdapter:
    """Tests for MMLU adapter."""

    def test_loads_mmlu_format(self):
        """Test MMLU format with integer answer."""
        mock_data = [
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
            }
        ]

        adapter = MMLUAdapter()
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert questions[0].question == "What is 2+2?"
        assert questions[0].choices == ["3", "4", "5", "6"]
        assert questions[0].correct == 1

    def test_handles_letter_answer(self):
        """Test MMLU with letter answer (A, B, C, D)."""
        mock_data = [
            {
                "question": "Capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": "B",
            }
        ]

        adapter = MMLUAdapter()
        questions = adapter.load(source=mock_data)

        assert questions[0].correct == 1  # B = index 1

    def test_metadata_includes_dataset(self):
        """Test metadata includes MMLU dataset name."""
        mock_data = [{"question": "Test?", "choices": ["A", "B"], "answer": 0}]

        adapter = MMLUAdapter()
        questions = adapter.load(source=mock_data)

        assert questions[0].metadata["dataset"] == "cais/mmlu"


class TestTruthfulQAAdapter:
    """Tests for TruthfulQA adapter."""

    def test_loads_mc1_format(self):
        """Test TruthfulQA mc1_targets format."""
        mock_data = [
            {
                "question": "What happens if you break a mirror?",
                "mc1_targets": {
                    "choices": [
                        "Seven years of bad luck",
                        "Nothing in particular",
                        "You will be cursed",
                    ],
                    "labels": [0, 1, 0],  # Second choice is correct
                },
            }
        ]

        adapter = TruthfulQAAdapter(mc_type="mc1")
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert questions[0].correct == 1  # Index of label==1
        assert len(questions[0].choices) == 3

    def test_mc2_format(self):
        """Test TruthfulQA can switch to mc2_targets."""
        mock_data = [
            {
                "question": "Test question?",
                "mc2_targets": {
                    "choices": ["A", "B", "C"],
                    "labels": [0, 0, 1],
                },
            }
        ]

        adapter = TruthfulQAAdapter(mc_type="mc2")
        questions = adapter.load(source=mock_data)

        assert questions[0].correct == 2


class TestWinograndeAdapter:
    """Tests for Winogrande adapter."""

    def test_loads_option1_option2_format(self):
        """Test Winogrande's option1/option2 format."""
        mock_data = [
            {
                "sentence": "The trophy doesn't fit in the brown suitcase because _ is too large.",
                "option1": "the trophy",
                "option2": "the suitcase",
                "answer": "1",
            }
        ]

        adapter = WinograndeAdapter()
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert len(questions[0].choices) == 2
        assert questions[0].choices[0] == "the trophy"
        assert questions[0].choices[1] == "the suitcase"
        assert questions[0].correct == 0  # "1" = first option = index 0

    def test_answer_conversion(self):
        """Test answer "2" correctly maps to index 1."""
        mock_data = [
            {
                "sentence": "Test",
                "option1": "A",
                "option2": "B",
                "answer": "2",
            }
        ]

        adapter = WinograndeAdapter()
        questions = adapter.load(source=mock_data)

        assert questions[0].correct == 1  # "2" = second option = index 1


class TestCommonsenseQAAdapter:
    """Tests for CommonsenseQA adapter."""

    def test_loads_nested_choices_format(self):
        """Test CommonsenseQA nested choices dict format."""
        mock_data = [
            {
                "question": "What do people do at the beach?",
                "choices": {
                    "text": ["swim", "run", "fly", "dig", "sleep"],
                    "label": ["A", "B", "C", "D", "E"],
                },
                "answerKey": "A",
            }
        ]

        adapter = CommonsenseQAAdapter()
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert len(questions[0].choices) == 5
        assert questions[0].choices[0] == "swim"
        assert questions[0].correct == 0  # A = index 0


class TestCodeReuse:
    """Tests verifying code reuse through inheritance."""

    def test_all_adapters_inherit_from_base(self):
        """All specialized adapters should inherit from BaseMCQAdapter."""
        adapters = [
            MMLUAdapter(),
            TruthfulQAAdapter(),
            WinograndeAdapter(),
            CommonsenseQAAdapter(),
        ]

        for adapter in adapters:
            assert isinstance(adapter, BaseMCQAdapter)

    def test_adapters_share_load_logic(self):
        """All adapters should have consistent load() behavior."""
        mock_data = [{"question": "Test?", "choices": ["A", "B"], "answer": 0}]

        # Should work for any adapter that accepts standard MCQ format
        adapter = MMLUAdapter()
        questions = adapter.load(source=mock_data)

        assert len(questions) == 1
        assert isinstance(questions[0], BencheckQuestion)
