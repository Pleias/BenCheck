#!/usr/bin/env python3
"""
Run experiment matrix from config with incremental result saving.

This script executes all combinations from an experiment matrix config,
saving results after each benchmark completion to enable early inspection.

Usage:
    python scripts/run_experiment.py configs/examples/quickstart.yaml
    python scripts/run_experiment.py configs/examples/quickstart.yaml --progress verbose
"""

import argparse
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from bencheck import BencheckRunner, adapters
from bencheck import checks as bencheck_checks
from bencheck.core.progress import ProgressLevel, set_progress_level
from bencheck.utils.config import Config

# Import output utilities from same directory
import sys
from pathlib import Path as _Path
_SCRIPTS_DIR = _Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from output_utils import (
    save_questions_base,
    save_check_result,
    save_per_question,
    aggregate_runs,
    create_experiment_summary,
    load_json,
)


def save_result(result: Any, output_path: Path, format: str = "json") -> None:
    """Save a single result to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            # Convert result to dict for JSON serialization
            if hasattr(result, "__dict__"):
                data = _serialize_result(result)
            else:
                data = result
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _serialize_result(obj: Any) -> Any:
    """Recursively serialize dataclass objects to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {field: _serialize_result(getattr(obj, field)) for field in obj.__dataclass_fields__}
    elif isinstance(obj, list):
        return [_serialize_result(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _serialize_result(v) for k, v in obj.items()}
    else:
        return obj


def aggregate_runs_for_combination(
    runs: List[Any], benchmark_name: str, model_name: str, scoring_mode: str
) -> Dict[str, Any]:
    """
    Aggregate multiple runs for a single benchmark-model-mode combination.

    Args:
        runs: List of BenchmarkEvaluation results
        benchmark_name: Name of benchmark
        model_name: Name of model
        scoring_mode: Scoring mode used

    Returns:
        Aggregated statistics dict
    """
    from bencheck.utils.statistics import aggregate_check_results_across_runs

    # Extract metrics without metadata for aggregation
    metrics_list = []
    for run in runs:
        metrics_without_metadata = {}
        for check_name, check_data in run.metrics.items():
            if isinstance(check_data, dict) and "_metadata" in check_data:
                # Remove metadata before aggregation
                metrics_without_metadata[check_name] = {k: v for k, v in check_data.items() if k != "_metadata"}
            else:
                metrics_without_metadata[check_name] = check_data
        metrics_list.append(metrics_without_metadata)

    metrics_stats = aggregate_check_results_across_runs(metrics_list)

    # Add metadata back to aggregated stats
    if runs and runs[0].metrics:
        for check_name in metrics_stats.keys():
            if check_name in runs[0].metrics and isinstance(runs[0].metrics[check_name], dict):
                if "_metadata" in runs[0].metrics[check_name]:
                    # Prepend metadata to aggregated stats
                    metadata = runs[0].metrics[check_name]["_metadata"]
                    metrics_stats[check_name] = {"_metadata": metadata, **metrics_stats[check_name]}

    return {
        "benchmark_name": benchmark_name,
        "model_name": model_name,
        "scoring_mode": scoring_mode,
        "n_runs": len(runs),
        "timestamp": datetime.now().isoformat(),
        "metrics_stats": metrics_stats,
        "individual_runs": [{"run_index": i, "metrics": run.metrics} for i, run in enumerate(runs)],
    }


def _get_check_metadata(check_name_or_class: str, model_name: str, scoring_mode: str) -> Dict[str, Any]:
    """
    Get metadata for a check (model/mode/type).

    Args:
        check_name_or_class: Check class name (e.g. "LengthBiasCheck") or check.name (e.g. "length_bias")
        model_name: Model name being used
        scoring_mode: Scoring mode (log_likelihood/generation/none)

    Returns:
        Metadata dict with model, mode, check_type
    """
    # Model-free checks (no model needed at all)
    # Map both class names and check.name values
    MODEL_FREE_CHECKS = {
        "LengthBiasCheck", "length_bias",
        "DummyCheck", "dummy"
    }

    # Scoring-independent checks (need model but don't depend on scoring_mode)
    SCORING_INDEPENDENT_CHECKS = {
        "EnumerationBiasCheck", "enumeration_bias",
        "GrammarQualityCheck", "grammar_quality",  # Uses separate GECToR model
        "NoneOfTheAboveCheck", "none_of_the_above"  # Requires generation model only
    }

    # Scoring-dependent checks (behavior changes based on scoring_mode)
    SCORING_DEPENDENT_CHECKS = {
        "ContextRequirementCheck", "context_requirement"
    }

    if check_name_or_class in MODEL_FREE_CHECKS:
        return {
            "model": None,
            "mode": None,
            "check_type": "model_free"
        }
    elif check_name_or_class in SCORING_INDEPENDENT_CHECKS:
        return {
            "model": model_name,
            "mode": None,  # Always uses log_likelihood internally
            "check_type": "scoring_independent"
        }
    elif check_name_or_class in SCORING_DEPENDENT_CHECKS:
        return {
            "model": model_name,
            "mode": scoring_mode,
            "check_type": "scoring_dependent"
        }
    else:
        # Unknown check - assume scoring-dependent to be safe
        return {
            "model": model_name,
            "mode": scoring_mode,
            "check_type": "unknown"
        }


def _add_metadata_to_metrics(metrics: Dict[str, Any], model_name: str, scoring_mode: str) -> Dict[str, Any]:
    """Add metadata to each check in metrics dict."""
    result = {}
    for check_name, check_metrics in metrics.items():
        metadata = _get_check_metadata(check_name, model_name, scoring_mode)
        result[check_name] = {
            "_metadata": metadata,
            **check_metrics
        }
    return result


def _extract_per_question_results(result: Any, check_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-question results for a specific check from BenchmarkEvaluation.

    Args:
        result: BenchmarkEvaluation object
        check_name: Name of the check to extract

    Returns:
        Dict mapping question_id to check-specific results
    """
    per_question = {}
    for question_id, question_data in result.questions.items():
        if "checks" in question_data and check_name in question_data["checks"]:
            per_question[question_id] = question_data["checks"][check_name]
    return per_question


def _save_check_files(
    benchmark_dir: Path,
    check_name: str,
    check_metrics: Dict[str, Any],
    per_question_results: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any]
) -> None:
    """
    Save check results to files (summary + per-question).

    Args:
        benchmark_dir: Directory for this benchmark
        check_name: Name of the check
        check_metrics: Summary metrics (without metadata)
        per_question_results: Per-question results
        metadata: Metadata dict
    """
    # Save summary metrics
    check_file = benchmark_dir / f"{check_name}.json"
    save_check_result(check_file, check_metrics, metadata)

    # Save per-question results
    per_questions_dir = benchmark_dir / "per_questions"
    per_questions_file = per_questions_dir / f"{check_name}.json"
    save_per_question(per_questions_file, per_question_results)


def run_experiment(config_path: str, progress_level: str = "normal", dry_run: bool = False) -> int:
    """
    Execute experiment matrix with incremental saving.

    Args:
        config_path: Path to experiment config file
        progress_level: Progress verbosity ("quiet", "normal", "verbose")
        dry_run: If True, only print what would be executed

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Set progress level
    level_map = {
        "quiet": ProgressLevel.QUIET,
        "normal": ProgressLevel.NORMAL,
        "verbose": ProgressLevel.VERBOSE,
    }
    set_progress_level(level_map[progress_level])

    # Load config
    print(f"Loading config: {config_path}")
    config = Config.from_file(config_path)

    if not config.is_experiment_matrix():
        print("❌ Error: Config is not experiment matrix format")
        print("   Add 'experiment:' top-level key with benchmarks, models, checks")
        return 1

    # Setup output directory
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT MATRIX")
    print(f"{'=' * 70}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Benchmarks: {len(config.experiment.get('benchmarks', []))}")
    print(f"Models: {len(config.experiment.get('models', []))}")
    print(f"Scoring modes: {config.scoring_modes}")
    print(f"Runs per combination: {config.n_runs}")
    print(f"Seed: {config.seed}")
    print(f"Progress level: {progress_level}")

    # Generate combinations
    combinations = config.generate_experiment_combinations()
    total = len(combinations)
    print(f"\nTotal runs to execute: {total}")
    print(f"{'=' * 70}\n")

    if dry_run:
        print("DRY RUN - showing what would be executed:\n")
        for idx, combo in enumerate(combinations[:10], 1):  # Show first 10
            print(
                f"{idx}. {combo['benchmark_name']} + {combo['model_name']} + "
                f"{combo['scoring_mode']} (run {combo['run_index']})"
            )
        if total > 10:
            print(f"... and {total - 10} more")
        return 0

    # Group combinations by (benchmark, scoring_mode) for aggregation (flat structure)
    # Cache for model-free and scoring-independent checks
    model_free_cache: Dict[str, Dict[str, Any]] = {}  # {benchmark_name: {check_name: metrics}}
    scoring_independent_cache: Dict[str, Dict[str, Any]] = {}  # {(benchmark, model): {check_name: metrics}}

    # Track which datasets have questions_base.json saved
    questions_base_saved: Dict[str, bool] = {}

    # Track scoring-dependent check runs for aggregation
    # {(benchmark, model, check_name): {mode: {runs: [], per_question_runs: []}}}
    scoring_dependent_runs: Dict[tuple, Dict[str, Dict[str, List]]] = {}

    # Execute each combination
    completed = 0
    failed = 0

    for idx, combo in enumerate(combinations, 1):
        benchmark_name = combo["benchmark_name"]
        model_name = combo["model_name"]
        scoring_mode = combo.get("scoring_mode", "none")
        run_idx = combo["run_index"]

        print(
            f"\n[{idx}/{total}] {benchmark_name} + {model_name} + {scoring_mode} (run {run_idx + 1}/{config.n_runs})"
        )
        print(f"{'─' * 70}")

        try:
            # === Prepare output directory for this benchmark ===
            benchmark_dir = output_dir / f"{benchmark_name}_"
            benchmark_dir.mkdir(parents=True, exist_ok=True)

            # Create adapter
            adapter = config.get_adapter(combo["benchmark_config"])

            # Create model
            model = config._create_model_from_config(combo["model_config"])

            # Separate checks by type
            model_free_checks = []
            scoring_independent_checks = []
            scoring_dependent_checks = []

            for check_cfg in config.checks:
                check_name = check_cfg.get("name")
                metadata = _get_check_metadata(check_name, model_name, scoring_mode)

                if metadata["check_type"] == "model_free":
                    model_free_checks.append((check_name, check_cfg))
                elif metadata["check_type"] == "scoring_independent":
                    scoring_independent_checks.append((check_name, check_cfg))
                else:
                    scoring_dependent_checks.append((check_name, check_cfg))

            # Build final check instances list
            check_instances = []
            final_metrics = {}

            # 1. Model-free checks - run once per dataset, cache results
            if benchmark_name not in model_free_cache:
                model_free_cache[benchmark_name] = {}

            for check_class_name, check_cfg in model_free_checks:
                if check_class_name not in model_free_cache[benchmark_name]:
                    # Run model-free check (first time for this dataset)
                    check_class = getattr(bencheck_checks, check_class_name)
                    params = check_cfg.get("params", {}).copy()
                    check_inst = check_class(**params)
                    check_instances.append(check_inst)
                    print(f"   🔵 Running model-free check: {check_class_name}")
                else:
                    # Use cached result - need to get check.name for final_metrics
                    check_class = getattr(bencheck_checks, check_class_name)
                    check_inst_name = check_class.name if hasattr(check_class, 'name') else check_class_name.lower()
                    final_metrics[check_inst_name] = model_free_cache[benchmark_name][check_class_name]
                    print(f"   ♻️  Using cached model-free check: {check_class_name}")

            # 2. Scoring-independent checks - run once per (dataset, model), cache results
            cache_key_si = (benchmark_name, model_name)
            if cache_key_si not in scoring_independent_cache:
                scoring_independent_cache[cache_key_si] = {}

            for check_class_name, check_cfg in scoring_independent_checks:
                if check_class_name not in scoring_independent_cache[cache_key_si]:
                    # Run scoring-independent check (first time for this dataset+model)
                    check_class = getattr(bencheck_checks, check_class_name)
                    params = check_cfg.get("params", {}).copy()

                    if "model" not in check_cfg:
                        params["model"] = model
                    elif check_cfg.get("model") is not None:
                        params["model"] = config._create_model_from_config(check_cfg["model"])

                    # Inject output_dir for checks that support checkpointing (e.g., GrammarQualityCheck)
                    accepts_output_dir = "output_dir" in inspect.signature(check_class).parameters
                    if accepts_output_dir and "output_dir" not in params:
                        params["output_dir"] = str(benchmark_dir)

                    check_inst = check_class(**params)
                    check_instances.append(check_inst)
                    print(f"   🔵 Running scoring-independent check: {check_class_name}")
                else:
                    # Use cached result - need to get check.name for final_metrics
                    check_class = getattr(bencheck_checks, check_class_name)
                    check_inst_name = check_class.name if hasattr(check_class, 'name') else check_class_name.lower()
                    final_metrics[check_inst_name] = scoring_independent_cache[cache_key_si][check_class_name]
                    print(f"   ♻️  Using cached scoring-independent check: {check_class_name}")

            # 3. Scoring-dependent checks - always run
            for check_name, check_cfg in scoring_dependent_checks:
                check_class = getattr(bencheck_checks, check_name)
                params = check_cfg.get("params", {}).copy()

                if "model" not in check_cfg:
                    params["model"] = model
                elif check_cfg.get("model") is not None:
                    params["model"] = config._create_model_from_config(check_cfg["model"])

                # Override scoring_mode for scoring-dependent checks
                if check_name == "ContextRequirementCheck" and scoring_mode != "none":
                    params["scoring_mode"] = scoring_mode

                check_inst = check_class(**params)
                check_instances.append(check_inst)
                print(f"   🔵 Running scoring-dependent check: {check_name}")

            # Create runner (single run, we handle multi-run at experiment level)
            runner = BencheckRunner(
                adapter=adapter,
                checks=check_instances,
                benchmark_name=benchmark_name,
                n_runs=1,  # Single run here
                seed=combo["seed"],
                scoring_mode=scoring_mode if scoring_mode != "none" else None,
            )

            # Execute
            result = runner.run()

            # Merge cached metrics with newly computed ones (before adding metadata)
            result.metrics = {**final_metrics, **result.metrics}

            # Add metadata to all metrics
            result.metrics = _add_metadata_to_metrics(result.metrics, model_name, scoring_mode)

            # === NEW OUTPUT STRUCTURE ===
            # benchmark_dir already created at start of loop

            # 1. Save questions_base.json (once per dataset)
            if benchmark_name not in questions_base_saved:
                questions_base_file = benchmark_dir / "questions_base.json"
                # Extract questions list from adapter (re-load to get clean data)
                questions_list = adapter.load()
                save_questions_base(questions_base_file, questions_list)
                questions_base_saved[benchmark_name] = True
                print(f"   💾 Saved questions_base.json")

            # 2. Save model-free check files (if newly computed)
            for check_class_name, check_cfg in model_free_checks:
                if check_class_name not in model_free_cache[benchmark_name]:
                    # This check was just computed, save it
                    check_class = getattr(bencheck_checks, check_class_name)
                    check_inst_name = check_class.name if hasattr(check_class, 'name') else check_class_name.lower()

                    if check_inst_name in result.metrics:
                        check_metrics = result.metrics[check_inst_name]
                        # Extract metadata
                        metadata = check_metrics.get("_metadata", {})
                        # Remove metadata from metrics for saving
                        metrics_without_metadata = {k: v for k, v in check_metrics.items() if k != "_metadata"}
                        # Extract per-question results
                        per_question_results = _extract_per_question_results(result, check_inst_name)
                        # Save files
                        _save_check_files(benchmark_dir, check_inst_name, metrics_without_metadata, per_question_results, metadata)
                        print(f"   💾 Saved {check_inst_name}.json + per_questions/")

            # 3. Save scoring-independent check files (if newly computed)
            for check_class_name, check_cfg in scoring_independent_checks:
                if check_class_name not in scoring_independent_cache[cache_key_si]:
                    check_class = getattr(bencheck_checks, check_class_name)
                    check_inst_name = check_class.name if hasattr(check_class, 'name') else check_class_name.lower()

                    if check_inst_name in result.metrics:
                        check_metrics = result.metrics[check_inst_name]
                        metadata = check_metrics.get("_metadata", {})
                        metrics_without_metadata = {k: v for k, v in check_metrics.items() if k != "_metadata"}
                        per_question_results = _extract_per_question_results(result, check_inst_name)
                        _save_check_files(benchmark_dir, check_inst_name, metrics_without_metadata, per_question_results, metadata)
                        print(f"   💾 Saved {check_inst_name}.json + per_questions/")

            # Update caches with newly computed checks (AFTER saving files)
            # Note: cache key is check class name, but result.metrics uses check.name
            # So we need to find check.name for each class
            for check_class_name, check_cfg in model_free_checks:
                if check_class_name not in model_free_cache[benchmark_name]:
                    # Find the actual check.name in result.metrics
                    check_class = getattr(bencheck_checks, check_class_name)
                    check_inst_name = check_class.name if hasattr(check_class, 'name') else check_class_name.lower()
                    if check_inst_name in result.metrics:
                        model_free_cache[benchmark_name][check_class_name] = result.metrics[check_inst_name]

            for check_class_name, check_cfg in scoring_independent_checks:
                if check_class_name not in scoring_independent_cache[cache_key_si]:
                    check_class = getattr(bencheck_checks, check_class_name)
                    check_inst_name = check_class.name if hasattr(check_class, 'name') else check_class_name.lower()
                    if check_inst_name in result.metrics:
                        scoring_independent_cache[cache_key_si][check_class_name] = result.metrics[check_inst_name]

            # 4. Handle scoring-dependent checks (accumulate runs for aggregation)
            for check_name, check_cfg in scoring_dependent_checks:
                check_class = getattr(bencheck_checks, check_name)
                check_inst_name = check_class.name if hasattr(check_class, 'name') else check_name.lower()

                if check_inst_name in result.metrics:
                    # Create tracking key
                    tracking_key = (benchmark_name, model_name, check_inst_name)
                    if tracking_key not in scoring_dependent_runs:
                        scoring_dependent_runs[tracking_key] = {}
                    if scoring_mode not in scoring_dependent_runs[tracking_key]:
                        scoring_dependent_runs[tracking_key][scoring_mode] = {
                            "runs": [],
                            "per_question_runs": [],
                            "metadata": None
                        }

                    check_metrics = result.metrics[check_inst_name]
                    metadata = check_metrics.get("_metadata", {})
                    metrics_without_metadata = {k: v for k, v in check_metrics.items() if k != "_metadata"}

                    # Store run data
                    run_data = {"run_index": run_idx, **metrics_without_metadata}
                    scoring_dependent_runs[tracking_key][scoring_mode]["runs"].append(run_data)
                    scoring_dependent_runs[tracking_key][scoring_mode]["metadata"] = metadata

                    # Save per-question file for this run
                    per_question_results = _extract_per_question_results(result, check_inst_name)
                    per_questions_dir = benchmark_dir / "per_questions"
                    per_question_file = per_questions_dir / f"{check_inst_name}_{scoring_mode}_run_{run_idx}.json"
                    save_per_question(per_question_file, per_question_results)
                    print(f"   💾 Saved per_questions/{check_inst_name}_{scoring_mode}_run_{run_idx}.json")

            completed += 1

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
            continue

    # === Aggregate scoring-dependent checks ===
    print(f"\n{'─' * 70}")
    print("Aggregating scoring-dependent checks...")
    print(f"{'─' * 70}")

    for tracking_key, modes_data in scoring_dependent_runs.items():
        benchmark_name, model_name, check_name = tracking_key
        benchmark_dir = output_dir / f"{benchmark_name}_"

        # Build full check data structure with all modes
        check_data = {
            "_metadata": {
                "model": model_name,
                "mode": None,  # Multiple modes
                "check_type": "scoring_dependent"
            }
        }

        for mode, mode_data in modes_data.items():
            runs_list = mode_data["runs"]

            # Aggregate runs
            aggregated = aggregate_runs(runs_list)

            check_data[mode] = {
                "runs": runs_list,
                "aggregated": aggregated
            }

        # Save aggregated file
        check_file = benchmark_dir / f"{check_name}.json"
        save_check_result(check_file, check_data, None)  # metadata already in check_data
        print(f"   ✅ Aggregated {check_name}.json ({', '.join(modes_data.keys())})")

    # === Create experiment summary ===
    print(f"\n{'─' * 70}")
    print("Creating experiment summary...")
    print(f"{'─' * 70}")

    create_experiment_summary(output_dir, config_path)
    print(f"   ✅ Created experiment_summary.json")

    # === Generate Markdown report ===
    print(f"\n{'─' * 70}")
    print("Generating Markdown report...")
    print(f"{'─' * 70}")

    try:
        import subprocess
        report_path = output_dir / "REPORT.md"
        subprocess.run(
            [sys.executable, "scripts/generate_report.py", str(output_dir), "-o", str(report_path)],
            check=True,
            capture_output=True
        )
        print(f"   ✅ Created REPORT.md")
    except Exception as e:
        print(f"   ⚠️  Failed to generate report: {e}")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total runs: {total}")
    print(f"Completed: {completed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {completed / total * 100:.1f}%")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'=' * 70}\n")

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Run BenCheck experiment matrix with incremental saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python scripts/run_experiment.py configs/examples/quickstart.yaml

  # Dry run to see what would execute
  python scripts/run_experiment.py configs/examples/quickstart.yaml --dry-run

  # Verbose progress for debugging
  python scripts/run_experiment.py configs/examples/quickstart.yaml --progress verbose

  # Quiet mode for batch jobs
  python scripts/run_experiment.py configs/examples/quickstart.yaml --progress quiet
        """,
    )

    parser.add_argument("config", help="Path to experiment matrix config file (YAML/JSON)")

    parser.add_argument(
        "--progress",
        choices=["quiet", "normal", "verbose"],
        default="normal",
        help="Progress bar verbosity (default: normal)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be executed without running"
    )

    args = parser.parse_args()

    try:
        exit_code = run_experiment(args.config, args.progress, args.dry_run)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
