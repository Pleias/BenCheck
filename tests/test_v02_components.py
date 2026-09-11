"""Tests for v0.2.0 components: per-check models, new checks, and adapters."""

import pytest

from bencheck.adapters import HellaSwagAdapter, UniversalMCQAdapter
from bencheck.checks import (
    EnumerationBiasCheck,
    LengthBiasCheck,
    NoneOfTheAboveCheck,
)
from bencheck.core.runner import BencheckRunner
from bencheck.models.dummy import DummyModel
from bencheck.types import BencheckQuestion

# Mock data fixtures


@pytest.fixture
def mock_hellaswag_records():
    """Mock HellaSwag-format records."""
    return [
        {
            "ind": 0,
            "ctx": "A person is trying to stack eggs",
            "endings": [
                "on a plate",
                "on top of each other vertically",
                "on a book",
                "on a brick",
            ],
            "label": 1,
            "source_id": "activitynet~v_001",
        },
        {
            "ind": 1,
            "ctx": "Someone is washing dishes",
            "endings": [
                "with soap and water",
                "with gasoline",
                "with paint",
                "with mud",
            ],
            "label": 0,
            "source_id": "activitynet~v_002",
        },
        {
            "ind": 2,
            "ctx_a": "A chef prepares",
            "ctx_b": "a gourmet meal",
            "endings": [
                "burns everything",
                "chops vegetables carefully",
                "throws food away",
                "eats raw ingredients",
            ],
            "label": 1,
            "source_id": "wikihow~001",
        },
    ]


@pytest.fixture
def mock_mmlu_records():
    """Mock MMLU-format records."""
    return [
        {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
        },
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": "B",  # Test letter format
        },
    ]


@pytest.fixture
def mock_truthfulqa_records():
    """Mock TruthfulQA-format records."""
    return [
        {
            "question": "What happens if you break a mirror?",
            "mc1_targets": {
                "choices": [
                    "You will have seven years of bad luck",
                    "Nothing in particular happens if you break a mirror",
                    "You will be cursed",
                ],
                "labels": [0, 1, 0],
            },
        },
    ]


# Adapter Tests


class TestHellaSwagAdapter:
    """Tests for HellaSwagAdapter."""

    def test_loads_mock_data(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        assert len(questions) == 3
        assert all(isinstance(q, BencheckQuestion) for q in questions)

    def test_handles_ctx_format(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        # First record uses simple "ctx"
        assert questions[0].question == "A person is trying to stack eggs"

        # Third record uses "ctx_a" + "ctx_b"
        assert questions[2].question == "A chef prepares a gourmet meal"

    def test_handles_string_label(self):
        adapter = HellaSwagAdapter()
        records = [
            {
                "ind": 0,
                "ctx": "Test",
                "endings": ["a", "b", "c", "d"],
                "label": "2",  # String label
            }
        ]
        questions = adapter.load(source=records)

        assert questions[0].correct == 2

    def test_filters_by_source(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter(filter_source="activitynet")
        questions = adapter.load(source=mock_hellaswag_records)

        # Only 2 activitynet records
        assert len(questions) == 2
        assert all("activitynet" in q.metadata["source_id"] for q in questions)

    def test_includes_metadata(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter(dataset_name="Rowan/hellaswag", split="validation")
        questions = adapter.load(source=mock_hellaswag_records)

        q = questions[0]
        assert q.metadata["dataset"] == "Rowan/hellaswag"
        assert q.metadata["split"] == "validation"
        assert "source_id" in q.metadata


class TestUniversalMCQAdapter:
    """Tests for UniversalMCQAdapter."""

    def test_loads_hellaswag_format(self, mock_hellaswag_records):
        adapter = UniversalMCQAdapter(dataset_name="Rowan/hellaswag")
        questions = adapter.load(source=mock_hellaswag_records)

        assert len(questions) == 3
        assert questions[0].question == "A person is trying to stack eggs"
        assert questions[0].correct == 1

    def test_loads_mmlu_format(self, mock_mmlu_records):
        adapter = UniversalMCQAdapter(dataset_name="cais/mmlu")
        questions = adapter.load(source=mock_mmlu_records)

        assert len(questions) == 2
        assert questions[0].question == "What is 2+2?"
        assert questions[0].correct == 1

        # Test letter conversion (B = index 1)
        assert questions[1].correct == 1

    def test_loads_truthfulqa_format(self, mock_truthfulqa_records):
        adapter = UniversalMCQAdapter(dataset_name="truthful_qa")
        questions = adapter.load(source=mock_truthfulqa_records)

        assert len(questions) == 1
        # Label at index 1 has value 1
        assert questions[0].correct == 1

    def test_auto_detects_keys(self):
        """Test automatic field name detection."""
        records = [
            {
                "prompt": "Test question",
                "options": ["A", "B", "C"],
                "correct_answer": 1,
            }
        ]

        adapter = UniversalMCQAdapter()
        questions = adapter.load(source=records)

        assert len(questions) == 1
        assert questions[0].question == "Test question"
        assert questions[0].choices == ["A", "B", "C"]
        assert questions[0].correct == 1

    def test_explicit_key_override(self):
        """Test explicit field name override."""
        records = [
            {
                "my_context": "Question",
                "my_choices": ["A", "B"],
                "my_answer": 0,
            }
        ]

        adapter = UniversalMCQAdapter(
            option_key="my_choices", label_key="my_answer", context_key="my_context"
        )
        questions = adapter.load(source=records)

        assert len(questions) == 1
        assert questions[0].question == "Question"


# Model Interface Tests


class TestDummyModelV02:
    """Tests for updated DummyModel with v0.2.0 interface."""

    def test_score_continuation_returns_float(self):
        model = DummyModel()
        score = model.score_continuation("hello world", "hello there")

        assert isinstance(score, float)
        assert score > 0  # Lexical overlap should give positive score

    def test_score_continuations_batch(self):
        model = DummyModel()
        scores = model.score_continuations("test", ["a", "b", "c"])

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_generate_returns_string(self):
        model = DummyModel()
        output = model.generate("test prompt")

        assert isinstance(output, str)
        assert "tokens observed" in output


# Check Tests


class TestLengthBiasCheck:
    """Tests for LengthBiasCheck (model-free)."""

    def test_model_free(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        check = LengthBiasCheck(length_metric="words")
        result = check.run(questions, model=None)  # No model needed

        assert result.check_name == "length_bias"
        assert "n_questions" in result.metrics
        assert result.metrics["n_questions"] == 3

    def test_analyzes_length_distribution(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        check = LengthBiasCheck(length_metric="words")
        result = check.run(questions)

        metrics = result.metrics
        assert "mean_correct_length" in metrics
        assert "mean_incorrect_length" in metrics
        assert "correct_is_longest_pct" in metrics
        assert "correct_is_shortest_pct" in metrics

    def test_question_level_results(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        check = LengthBiasCheck()
        result = check.run(questions)

        q_result = result.question_results["0"]
        assert "option_lengths" in q_result
        assert "correct_length" in q_result
        assert "longest_is_correct" in q_result
        assert len(q_result["option_lengths"]) == 4


class TestEnumerationBiasCheck:
    """Tests for EnumerationBiasCheck (model-dependent)."""

    def test_requires_model(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        check = EnumerationBiasCheck(model=None)

        with pytest.raises(ValueError, match="requires a model"):
            check.run(questions)

    def test_with_dummy_model(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        model = DummyModel()
        check = EnumerationBiasCheck(model=model)
        result = check.run(questions)

        assert result.check_name == "enumeration_bias"
        assert "overall_accuracy" in result.metrics
        assert "gold_position_dist" in result.metrics
        assert "pred_position_dist" in result.metrics

    def test_computes_positional_stats(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        model = DummyModel()
        check = EnumerationBiasCheck(model=model)
        result = check.run(questions)

        metrics = result.metrics
        assert "accuracy_by_gold_position" in metrics
        assert "gold_chi2_stat" in metrics
        assert "pred_chi2_stat" in metrics


class TestNoneOfTheAboveCheck:
    """Tests for NoneOfTheAboveCheck (model-dependent)."""

    def test_requires_model(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        check = NoneOfTheAboveCheck(model=None)

        with pytest.raises(ValueError, match="requires a model"):
            check.run(questions)

    def test_with_dummy_model(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        model = DummyModel()
        check = NoneOfTheAboveCheck(model=model)
        result = check.run(questions)

        assert result.check_name == "none_of_the_above"
        assert "baseline_accuracy" in result.metrics
        assert "placeholder_accuracy" in result.metrics
        assert "placeholder_selected_rate" in result.metrics

    def test_replaces_correct_answer(self, mock_hellaswag_records):
        adapter = HellaSwagAdapter()
        questions = adapter.load(source=mock_hellaswag_records)

        model = DummyModel()
        check = NoneOfTheAboveCheck(model=model, placeholder_text="None of the above")
        result = check.run(questions)

        # Check question-level results
        q_result = result.question_results["0"]
        assert "baseline_prediction" in q_result
        assert "placeholder_prediction" in q_result
        assert "placeholder_selected" in q_result
        assert "prediction_changed" in q_result


# Integration Tests


class TestV02Pipeline:
    """Integration tests for v0.2.0 per-check model pipeline."""

    def test_full_pipeline_mixed_checks(self, mock_hellaswag_records):
        """Test pipeline with both model-free and model-dependent checks."""
        adapter = HellaSwagAdapter()

        # Model-free check
        length_check = LengthBiasCheck()

        # Model-dependent checks (share same model)
        model = DummyModel()
        enum_check = EnumerationBiasCheck(model=model)
        none_check = NoneOfTheAboveCheck(model=model)

        # Run pipeline (no model parameter to runner!)
        runner = BencheckRunner(
            adapter=adapter,
            checks=[length_check, enum_check, none_check],
            benchmark_name="Test",
        )
        results = runner.run(source=mock_hellaswag_records)

        # Verify structure
        assert results.benchmark_name == "Test"
        assert len(results.questions) == 3
        assert set(results.metrics.keys()) == {
            "length_bias",
            "enumeration_bias",
            "none_of_the_above",
        }

        # Verify each question has all check results
        q = results.questions["0"]
        assert "length_bias" in q["checks"]
        assert "enumeration_bias" in q["checks"]
        assert "none_of_the_above" in q["checks"]

    def test_runner_no_model_parameter(self, mock_hellaswag_records):
        """Verify runner doesn't have global model parameter."""
        adapter = HellaSwagAdapter()
        check = LengthBiasCheck()

        # Should work without model parameter
        runner = BencheckRunner(adapter=adapter, checks=[check])
        results = runner.run(source=mock_hellaswag_records)

        assert len(results.questions) == 3
