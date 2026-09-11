from pathlib import Path

import pytest

from bencheck.adapters import JsonAdapter
from bencheck.checks import DummyCheck
from bencheck.core.runner import BencheckRunner

EXAMPLE_DATASET = Path("examples/data/none_above/sample_questions.json")


def test_json_adapter_loads_example_dataset():
    if not EXAMPLE_DATASET.exists():
        pytest.skip("example dataset missing")

    adapter = JsonAdapter()
    questions = adapter.load(EXAMPLE_DATASET)

    assert len(questions) >= 2
    assert all(question.choices for question in questions)


def test_pipeline_runs_with_json_adapter():
    if not EXAMPLE_DATASET.exists():
        pytest.skip("example dataset missing")

    adapter = JsonAdapter()
    # v0.2.0 API: No model parameter, checks manage their own models
    runner = BencheckRunner(
        adapter=adapter,
        checks=[DummyCheck()],  # DummyCheck works model-free by default
        benchmark_name="json-demo",
    )
    results = runner.run(source=EXAMPLE_DATASET)

    # Check BenchmarkEvaluation structure
    assert results.benchmark_name == "json-demo"
    assert "dummy_check" in results.metrics
    assert len(results.questions) == len(adapter.load(EXAMPLE_DATASET))
