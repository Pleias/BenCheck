"""Utilities for running multiple benchmarks and models."""

from typing import Any, Callable, Dict, List, Tuple

from ..base import DatasetAdapter, BencheckCheck
from ..core.runner import BencheckRunner
from ..types import BenchmarkEvaluation


def run_multiple_benchmarks(
    benchmarks: Dict[str, Tuple[DatasetAdapter, Any]],
    checks: List[BencheckCheck],
) -> List[BenchmarkEvaluation]:
    """
    Run evaluation on multiple benchmarks with pre-configured checks.

    In v0.2.0, checks manage their own models. Pass pre-configured check
    instances (e.g., EnumerationBiasCheck(model=my_model)).

    Args:
        benchmarks: Dict mapping benchmark_name → (adapter, source)
        checks: List of pre-configured checks to run

    Returns:
        List of BenchmarkEvaluation objects, one per benchmark

    Example:
        >>> # Model-free checks
        >>> results = run_multiple_benchmarks(
        ...     benchmarks={
        ...         "hellaswag": (HellaswagAdapter(), "data/hellaswag.json"),
        ...         "mmlu": (MMLUAdapter(), "data/mmlu/"),
        ...     },
        ...     checks=[LengthBiasCheck()],
        ... )
        >>>
        >>> # Model-dependent checks
        >>> model = DummyModel()
        >>> results = run_multiple_benchmarks(
        ...     benchmarks={...},
        ...     checks=[
        ...         EnumerationBiasCheck(model=model),
        ...         LengthBiasCheck(),  # Model-free
        ...     ],
        ... )
    """
    evaluations = []
    for name, (adapter, source) in benchmarks.items():
        runner = BencheckRunner(
            adapter=adapter,
            checks=checks,
            benchmark_name=name,
        )
        evaluations.append(runner.run(source))
    return evaluations


def run_multiple_models(
    adapter: DatasetAdapter,
    source: Any,
    models: Dict[str, Any],
    check_factories: List[Callable[[Any], BencheckCheck]],
    benchmark_name: str | None = None,
) -> Dict[str, BenchmarkEvaluation]:
    """
    Run evaluation with multiple models on a single benchmark.

    In v0.2.0, checks manage their own models. This function accepts check
    factories (callables that create check instances given a model).

    Args:
        adapter: Dataset adapter
        source: Data source for adapter
        models: Dict mapping model_name → model instance
        check_factories: List of functions that create checks given a model.
            For model-dependent checks: lambda m: EnumerationBiasCheck(model=m)
            For model-free checks: lambda _: LengthBiasCheck()
        benchmark_name: Optional benchmark name

    Returns:
        Dict mapping model_name → BenchmarkEvaluation

    Example:
        >>> results = run_multiple_models(
        ...     adapter=HellaswagAdapter(),
        ...     source="data/hellaswag.json",
        ...     models={
        ...         "model_a": ModelA(),
        ...         "model_b": ModelB(),
        ...     },
        ...     check_factories=[
        ...         lambda m: EnumerationBiasCheck(model=m),  # Model-dependent
        ...         lambda _: LengthBiasCheck(),  # Model-free
        ...     ],
        ... )
        >>> results["model_a"]  # BenchmarkEvaluation for model_a
    """
    evaluations = {}
    for model_name, model in models.items():
        # Create check instances for this model
        checks = [factory(model) for factory in check_factories]

        runner = BencheckRunner(
            adapter=adapter,
            checks=checks,
            benchmark_name=benchmark_name,
        )
        result = runner.run(source)
        # Tag with model name
        result.model_config["user_label"] = model_name
        evaluations[model_name] = result
    return evaluations


def run_multiple_benchmarks_and_models(
    benchmarks: Dict[str, Tuple[DatasetAdapter, Any]],
    models: Dict[str, Any],
    check_factories: List[Callable[[Any], BencheckCheck]],
) -> Dict[str, Dict[str, BenchmarkEvaluation]]:
    """
    Run evaluation on multiple benchmarks with multiple models.

    In v0.2.0, checks manage their own models. This function accepts check
    factories (callables that create check instances given a model).

    Args:
        benchmarks: Dict mapping benchmark_name → (adapter, source)
        models: Dict mapping model_name → model instance
        check_factories: List of functions that create checks given a model.
            For model-dependent checks: lambda m: EnumerationBiasCheck(model=m)
            For model-free checks: lambda _: LengthBiasCheck()

    Returns:
        Nested dict: {benchmark_name: {model_name: BenchmarkEvaluation}}

    Example:
        >>> results = run_multiple_benchmarks_and_models(
        ...     benchmarks={
        ...         "hellaswag": (HellaswagAdapter(), "data/hellaswag.json"),
        ...         "mmlu": (MMLUAdapter(), "data/mmlu/"),
        ...     },
        ...     models={
        ...         "model_a": ModelA(),
        ...         "model_b": ModelB(),
        ...     },
        ...     check_factories=[
        ...         lambda m: EnumerationBiasCheck(model=m),
        ...         lambda _: LengthBiasCheck(),
        ...     ],
        ... )
        >>> results["hellaswag"]["model_a"]  # BenchmarkEvaluation
    """
    results = {}
    for benchmark_name, (adapter, source) in benchmarks.items():
        results[benchmark_name] = {}
        for model_name, model in models.items():
            # Create check instances for this model
            checks = [factory(model) for factory in check_factories]

            runner = BencheckRunner(
                adapter=adapter,
                checks=checks,
                benchmark_name=benchmark_name,
            )
            result = runner.run(source)
            result.model_config["user_label"] = model_name
            results[benchmark_name][model_name] = result
    return results


def aggregate_by_model(
    evaluations: List[BenchmarkEvaluation],
) -> Dict[str, List[BenchmarkEvaluation]]:
    """
    Group evaluations by model name.

    Useful when you have run_multiple_benchmarks() with different models
    and want to compare how each model performed across benchmarks.
    """
    by_model = {}
    for eval in evaluations:
        model_name = eval.model_config.get("user_label") or eval.model_config["name"]
        if model_name not in by_model:
            by_model[model_name] = []
        by_model[model_name].append(eval)
    return by_model


def aggregate_by_benchmark(
    evaluations: List[BenchmarkEvaluation],
) -> Dict[str, List[BenchmarkEvaluation]]:
    """
    Group evaluations by benchmark name.

    Useful when you have run_multiple_models() on different benchmarks
    and want to compare different models on the same benchmark.
    """
    by_benchmark = {}
    for eval in evaluations:
        if eval.benchmark_name not in by_benchmark:
            by_benchmark[eval.benchmark_name] = []
        by_benchmark[eval.benchmark_name].append(eval)
    return by_benchmark
