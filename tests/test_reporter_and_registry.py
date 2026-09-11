import json

import pytest

from bencheck.adapters import DummyAdapter
from bencheck.checks import DummyCheck
from bencheck.core.registry import BencheckRegistry, adapter_registry
from bencheck.core.reporter import BencheckReporter
from bencheck.core.runner import BencheckRunner


@pytest.fixture()
def pipeline_results():
    # v0.2.0 API: No model parameter, checks manage their own models
    runner = BencheckRunner(
        adapter=DummyAdapter(),
        checks=[DummyCheck()],  # DummyCheck works model-free by default
        benchmark_name="reporter",
    )
    return runner.run()


def test_reporter_exports_json(tmp_path, pipeline_results):
    reporter = BencheckReporter()
    json_path = tmp_path / "results.json"
    reporter.to_json(pipeline_results, json_path)

    with json_path.open() as handle:
        payload = json.load(handle)

    assert payload == reporter.to_dict(pipeline_results)
    assert payload["benchmark_name"] == "reporter"
    assert "model_config" in payload
    assert "questions" in payload
    assert "metrics" in payload


def test_registry_registers_and_resolves_custom_factory():
    registry = BencheckRegistry()
    registry.register("test", lambda: "ok")
    assert registry.resolve("test")() == "ok"


def test_global_registry_handles_adapter_registration():
    adapter_registry.register("dummy", DummyAdapter)
    adapter_cls = adapter_registry.resolve("dummy")
    adapter = adapter_cls()
    assert isinstance(adapter, DummyAdapter)
