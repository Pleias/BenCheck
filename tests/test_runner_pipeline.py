import pytest

from bencheck.adapters import DummyAdapter
from bencheck.checks import DummyCheck
from bencheck.core.runner import BencheckRunner
from bencheck.models import DummyModel
from bencheck.types import BencheckQuestion


@pytest.fixture()
def pipeline_components():
    adapter = DummyAdapter()
    model = DummyModel()
    check = DummyCheck()
    return adapter, model, check


def test_runner_produces_expected_structure(pipeline_components):
    adapter, model, check = pipeline_components
    # v0.2.0 API: No model parameter, checks manage their own models
    runner = BencheckRunner(
        adapter=adapter,
        checks=[check],  # DummyCheck works model-free by default
        benchmark_name="demo_benchmark",
    )
    results = runner.run()

    # Check BenchmarkEvaluation structure
    assert results.benchmark_name == "demo_benchmark"
    assert set(results.metrics.keys()) == {check.name}

    # Check questions structure
    assert len(results.questions) == len(adapter.load(None))
    for q_data in results.questions.values():
        assert "checks" in q_data
        assert check.name in q_data["checks"]
        assert "question_type" in q_data

    # Check metrics
    check_metrics = results.metrics[check.name]
    assert "accuracy" in check_metrics
    assert "challenging_questions_pct" in check_metrics
    assert "cheatable_questions_pct" in check_metrics
    assert 0.0 <= check_metrics["challenging_questions_pct"] <= 100.0
    assert 0.0 <= check_metrics["cheatable_questions_pct"] <= 100.0


def test_dummy_adapter_loads_questions():
    adapter = DummyAdapter()
    dataset = adapter.load(None)

    assert len(dataset) == 2
    assert all(isinstance(q, BencheckQuestion) for q in dataset)
    assert dataset[0].id == "q1"
    assert dataset[1].id == "q2"


def test_dummy_check_produces_per_question_details(pipeline_components):
    adapter, model, check = pipeline_components
    dataset = adapter.load(None)
    # In v0.2.0, model parameter is ignored for model-free checks
    results = check.run(dataset, model=None)

    # Check CheckResult structure
    assert results.check_name == check.name
    assert hasattr(results, "question_results")
    assert hasattr(results, "metrics")

    # Check per-question results
    for question in dataset:
        assert question.id in results.question_results
        entry = results.question_results[question.id]
        # v0.2.0: DummyCheck returns score_spread instead of model_output
        assert {
            "predicted_choice",
            "is_correct",
            "challenging",
            "cheatable",
            "score_spread",
            "scores",
        }.issubset(entry.keys())
        assert isinstance(entry["scores"], list)
        assert len(entry["scores"]) == len(question.choices)


def test_metrics_support_multiple_value_types():
    """Test that CheckResult.metrics can store float, tuple, and dict values."""
    from bencheck.types import CheckResult

    # Create a CheckResult with various metric types
    metrics = {
        "accuracy": 85.5,  # float
        "confidence": (0.76, 0.12),  # tuple (mean, std)
        "by_category": {"easy": 95.0, "hard": 60.0},  # dict
    }

    result = CheckResult(
        check_name="test_check", question_results={"q1": {"score": 1.0}}, metrics=metrics
    )

    # Verify each type
    assert isinstance(result.metrics["accuracy"], float)
    assert result.metrics["accuracy"] == 85.5

    assert isinstance(result.metrics["confidence"], tuple)
    assert result.metrics["confidence"] == (0.76, 0.12)

    assert isinstance(result.metrics["by_category"], dict)
    assert result.metrics["by_category"]["easy"] == 95.0
    assert result.metrics["by_category"]["hard"] == 60.0
