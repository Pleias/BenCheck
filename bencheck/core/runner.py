"""Coordinates adapters, models and checks."""

from typing import Any, Dict, List, Optional, Union

from ..base import DatasetAdapter, BencheckCheck
from ..types import BenchmarkEvaluation, CheckResult, MultiRunEvaluation


class BencheckRunner:
    """
    Orchestrates evaluation pipeline for a single benchmark.

    The runner coordinates dataset loading (via adapter) and diagnostic checks.
    Models are now provided per-check rather than globally, allowing different
    checks to use different models (or no model at all).

    Supports both single-run (n_runs=1) and multi-run (n_runs>1) execution
    with automatic statistical aggregation.

    Example:
        >>> from bencheck import BencheckRunner
        >>> from bencheck.adapters import JsonAdapter
        >>> from bencheck.checks import LengthBiasCheck
        >>>
        >>> # Single run
        >>> runner = BencheckRunner(
        ...     adapter=JsonAdapter(),
        ...     checks=[LengthBiasCheck()],  # model-free check
        ...     benchmark_name="my_benchmark"
        ... )
        >>> results = runner.run("data.json")  # Returns BenchmarkEvaluation
        >>>
        >>> # Multi-run with statistics
        >>> runner = BencheckRunner(
        ...     adapter=JsonAdapter(),
        ...     checks=[LengthBiasCheck()],
        ...     benchmark_name="my_benchmark",
        ...     n_runs=5,
        ...     seed=42
        ... )
        >>> results = runner.run("data.json")  # Returns MultiRunEvaluation
    """

    def __init__(
        self,
        adapter: DatasetAdapter,
        checks: List[BencheckCheck],
        benchmark_name: str | None = None,
        n_runs: int = 1,
        seed: Optional[int] = None,
        scoring_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the runner.

        Args:
            adapter: Dataset adapter to load benchmark questions
            checks: List of diagnostic checks to run. Each check can have its
                own model or be model-free.
            benchmark_name: Name of the benchmark (defaults to adapter class name)
            n_runs: Number of times to run the evaluation (for statistical reliability)
                Default: 1 (single run, returns BenchmarkEvaluation)
                If > 1: Returns MultiRunEvaluation with aggregated statistics
            seed: Random seed for reproducibility. Each run gets seed + run_index
            scoring_mode: Scoring mode used ("log_likelihood", "generation", or None)
                Tracked in results for experiment matrix support
        """
        self._adapter = adapter
        self._checks = checks
        self._benchmark_name = benchmark_name or adapter.__class__.__name__
        self.n_runs = n_runs
        self.seed = seed
        self.scoring_mode = scoring_mode

    def run(self, source: Any | None = None) -> Union[BenchmarkEvaluation, MultiRunEvaluation]:
        """
        Execute the adapter + checks pipeline.

        Args:
            source: Data source to pass to adapter (file path, URL, dict, etc.)

        Returns:
            - BenchmarkEvaluation if n_runs=1 (single run, backward compatible)
            - MultiRunEvaluation if n_runs>1 (multi-run with statistics)
        """
        if self.n_runs == 1:
            return self._run_single(source)
        else:
            return self._run_multiple(source)

    def _run_single(self, source: Any | None = None) -> BenchmarkEvaluation:
        """
        Execute single run of the pipeline.

        Args:
            source: Data source to pass to adapter

        Returns:
            BenchmarkEvaluation with results
        """
        # Load dataset
        dataset = self._adapter.load(source)

        # Run all checks (each check manages its own model if needed)
        check_results: Dict[str, CheckResult] = {}
        for check in self._checks:
            # v0.2.0+: Checks use their own models (passed to __init__), not to run()
            # We pass model=None for backward compatibility with the signature,
            # but checks ignore it and use self.model instead
            result = check.run(dataset, model=None)
            check_results[check.name] = result

        # Build question-centric structure
        questions_dict = {}
        for q in dataset:
            question_checks = {
                name: result.question_results[q.id]
                for name, result in check_results.items()
                if q.id in result.question_results
            }

            questions_dict[q.id] = {
                "question": q.question,
                "choices": q.choices,
                "correct": q.correct,
                "question_type": q.question_type.value,
                "metadata": q.metadata,
                "checks": question_checks,
            }

        # Aggregate metrics
        metrics = {name: result.metrics for name, result in check_results.items()}

        # Build model config (now empty since models are per-check)
        model_config = {"note": "Models are now configured per-check rather than globally"}

        return BenchmarkEvaluation(
            benchmark_name=self._benchmark_name,
            model_config=model_config,
            questions=questions_dict,
            metrics=metrics,
        )

    def _run_multiple(self, source: Any | None = None) -> MultiRunEvaluation:
        """
        Execute multiple runs of the pipeline with statistical aggregation.

        Args:
            source: Data source to pass to adapter

        Returns:
            MultiRunEvaluation with aggregated statistics across runs
        """
        from ..utils.statistics import aggregate_check_results_across_runs
        from .progress import ProgressLevel, progress_bar

        # Load dataset once (shared across runs)
        dataset = self._adapter.load(source)

        individual_runs = []

        # Run N times with progress bar (NORMAL level: visible by default for runs)
        for run_idx in progress_bar(
            range(self.n_runs), f"Running {self._benchmark_name}", ProgressLevel.NORMAL, unit="run"
        ):
            # Each run gets its own seed
            run_seed = (self.seed + run_idx) if self.seed is not None else None

            # Run all checks with seed propagation
            # Checks that don't use randomness will ignore the seed parameter
            check_results: Dict[str, CheckResult] = {}
            for check in self._checks:
                result = check.run(dataset, model=None, seed=run_seed)
                check_results[check.name] = result

            # Build BenchmarkEvaluation for this run
            questions_dict = {}
            for q in dataset:
                question_checks = {
                    name: result.question_results[q.id]
                    for name, result in check_results.items()
                    if q.id in result.question_results
                }

                questions_dict[q.id] = {
                    "question": q.question,
                    "choices": q.choices,
                    "correct": q.correct,
                    "question_type": q.question_type.value,
                    "metadata": q.metadata,
                    "checks": question_checks,
                }

            metrics = {name: result.metrics for name, result in check_results.items()}

            model_config = {
                "note": "Models are now configured per-check rather than globally",
                "run_index": run_idx,
                "seed": run_seed,
            }

            evaluation = BenchmarkEvaluation(
                benchmark_name=self._benchmark_name,
                model_config=model_config,
                questions=questions_dict,
                metrics=metrics,
            )

            individual_runs.append(evaluation)

        # Aggregate statistics across runs
        metrics_stats = aggregate_check_results_across_runs(
            [run.metrics for run in individual_runs]
        )

        # Create summary
        summary = self._create_summary(metrics_stats)

        return MultiRunEvaluation(
            benchmark_name=self._benchmark_name,
            model_config={"note": "Models are now configured per-check rather than globally"},
            scoring_mode=self.scoring_mode,
            n_runs=self.n_runs,
            seed=self.seed,
            individual_runs=individual_runs,
            metrics_stats=metrics_stats,
            summary=summary,
        )

    def _create_summary(
        self, metrics_stats: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Create high-level summary from aggregated statistics.

        Args:
            metrics_stats: Aggregated statistics

        Returns:
            Summary dict with key metrics
        """
        summary = {"n_checks": len(metrics_stats), "checks": list(metrics_stats.keys())}

        # Extract accuracy metrics if available
        for check_name, check_stats in metrics_stats.items():
            for metric_name, stats in check_stats.items():
                if "accuracy" in metric_name.lower() and isinstance(stats, dict):
                    if "mean" in stats:
                        summary[f"{check_name}_{metric_name}_mean"] = stats["mean"]
                        summary[f"{check_name}_{metric_name}_std"] = stats.get("std", 0.0)

        return summary
