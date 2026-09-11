#!/usr/bin/env python3
"""
Run BenCheck checks from configuration file.

Usage:
    python scripts/run_from_config.py configs/examples/grammar_quality_gector.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bencheck import BencheckRunner
from bencheck.core.progress import ProgressLevel, set_progress_level
from bencheck.types import MultiRunEvaluation
from bencheck.utils.config import Config
from bencheck.utils.summary import create_summary_table


def display_question_results(results, show_count: int = 0, show_errors_only: bool = False):
    """
    Display per-question results in console.

    Args:
        results: BenchmarkEvaluation results
        show_count: Number of questions to show (0 = none)
        show_errors_only: Show only incorrect predictions
    """
    if show_count == 0 and not show_errors_only:
        return

    print("\n" + "=" * 60)
    print("Per-Question Results")
    print("=" * 60)

    questions_to_show = []

    for q_id, q_data in results.questions.items():
        # Check if any check has incorrect prediction
        has_error = False
        for check_name, check_result in q_data["checks"].items():
            if check_result.get("is_correct") is False:
                has_error = True
                break

        if show_errors_only and not has_error:
            continue

        questions_to_show.append((q_id, q_data))

        if not show_errors_only and len(questions_to_show) >= show_count:
            break

    if not questions_to_show:
        print("\n✅ No errors found!" if show_errors_only else "\nNo questions to display.")
        return

    for i, (q_id, q_data) in enumerate(questions_to_show, 1):
        print(f"\n[{i}] Question ID: {q_id}")
        print(f"Q: {q_data['question'][:100]}{'...' if len(q_data['question']) > 100 else ''}")
        print(f"Choices: {q_data['choices']}")
        print(f"Correct: {q_data['correct']}")

        for check_name, check_result in q_data["checks"].items():
            print(f"\n  {check_name}:")

            # Show prediction if available
            if "predicted_position" in check_result:
                pred_pos = check_result["predicted_position"]
                is_correct = check_result.get("is_correct", False)
                status = "✅" if is_correct else "❌"
                print(f"    Predicted: {pred_pos} {status}")

            # Show scores if available
            if "scores" in check_result:
                scores = check_result["scores"]
                print(f"    Scores: {[f'{s:.3f}' for s in scores]}")

            # Show other metrics
            for key, value in check_result.items():
                if key not in ["predicted_position", "is_correct", "scores", "gold_position"]:
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    elif isinstance(value, list) and len(value) < 10:
                        print(f"    {key}: {value}")

    if show_errors_only:
        total_errors = len(questions_to_show)
        total_questions = len(results.questions)
        print(
            f"\n📊 Total errors: {total_errors}/{total_questions} ({total_errors / total_questions * 100:.1f}%)"
        )
    elif len(questions_to_show) < len(results.questions):
        print(f"\n... and {len(results.questions) - len(questions_to_show)} more questions")
        print(f"💡 Use --show-questions {len(results.questions)} to see all")


def save_results(results, config: Config):
    """
    Save results to file.

    Args:
        results: BenchmarkEvaluation or MultiRunEvaluation results
        config: Config instance
    """
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_format = config.get_output_format()

    # Handle MultiRunEvaluation differently
    if isinstance(results, MultiRunEvaluation):
        # Save aggregated statistics
        if output_format == "json":
            stats_file = output_dir / f"stats_{results.benchmark_name}_{timestamp}.json"

            stats_dict = {
                "benchmark_name": results.benchmark_name,
                "timestamp": timestamp,
                "n_runs": results.n_runs,
                "seed": results.seed,
                "scoring_mode": results.scoring_mode,
                "metrics_stats": results.metrics_stats,
                "summary": results.summary,
            }

            with open(stats_file, "w") as f:
                json.dump(stats_dict, f, indent=2)

            print(f"\n✅ Multi-run stats saved to: {stats_file}")

            # Optionally save individual runs
            if config.output.get("save_individual_runs", False):
                for idx, run in enumerate(results.individual_runs):
                    run_file = output_dir / f"run_{idx}_{results.benchmark_name}_{timestamp}.json"
                    run_dict = {
                        "benchmark_name": run.benchmark_name,
                        "run_index": idx,
                        "timestamp": timestamp,
                        "metrics": run.metrics,
                    }
                    if config.should_save_questions():
                        run_dict["questions"] = run.questions

                    with open(run_file, "w") as f:
                        json.dump(run_dict, f, indent=2)

                print(f"✅ Individual runs saved: {results.n_runs} files")

        return

    # Original single-run logic
    if output_format == "json":
        output_file = output_dir / f"results_{timestamp}.json"

        # Prepare results dict
        results_dict = {
            "benchmark_name": results.benchmark_name,
            "timestamp": timestamp,
            "metrics": results.metrics,
        }

        if config.should_save_questions():
            results_dict["questions"] = results.questions

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")

    elif output_format == "csv":
        # Save metrics as CSV
        import csv

        metrics_file = output_dir / f"metrics_{timestamp}.csv"

        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["check", "metric", "value"])

            for check_name, metrics in results.metrics.items():
                for metric_name, value in metrics.items():
                    writer.writerow([check_name, metric_name, value])

        print(f"\n✅ Metrics saved to: {metrics_file}")

        if config.should_save_questions():
            questions_file = output_dir / f"questions_{timestamp}.json"
            with open(questions_file, "w") as f:
                json.dump(results.questions, f, indent=2)
            print(f"✅ Questions saved to: {questions_file}")

    else:
        raise ValueError(f"Unknown output format: {output_format}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run BenCheck checks from configuration file")
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file (JSON or YAML)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and show plan without running",
    )
    parser.add_argument(
        "--show-questions",
        type=int,
        metavar="N",
        help="Show first N question results in console (default: 0)",
        default=0,
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show only incorrect predictions in console",
    )
    parser.add_argument(
        "--progress",
        choices=["quiet", "normal", "verbose"],
        default="normal",
        help="Progress bar verbosity (default: normal)",
    )

    args = parser.parse_args()

    # Set progress level
    level_map = {
        "quiet": ProgressLevel.QUIET,
        "normal": ProgressLevel.NORMAL,
        "verbose": ProgressLevel.VERBOSE,
    }
    set_progress_level(level_map[args.progress])

    # Load configuration
    print("=" * 60)
    print("BenCheck Configuration Runner")
    print("=" * 60)
    print(f"\n📄 Loading config: {args.config}")

    try:
        config = Config.from_file(args.config)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1

    # Check if this is an experiment matrix config
    if config.is_experiment_matrix():
        print("\n⚠️  Detected experiment matrix config!")
        print("   Please use run_experiment.py for experiment matrix configs:")
        print(f"   python scripts/run_experiment.py {args.config}")
        print("\n   Experiment matrix configs should be run with the dedicated script")
        print("   to enable incremental saving and proper result organization.")
        return 1

    # Display configuration
    print("\n📋 Configuration:")

    if config.is_multi_benchmark():
        print(f"  Mode: Multi-benchmark ({len(config.benchmarks)} benchmarks)")
        for i, bench in enumerate(config.benchmarks, 1):
            print(f"    {i}. {bench.get('name', 'N/A')} ({bench.get('adapter', 'N/A')})")
    else:
        print(f"  Benchmark: {config.benchmark.get('name', 'N/A')}")
        print(f"  Adapter: {config.benchmark.get('adapter', 'N/A')}")

    if config.model:
        print(f"  Model: {config.model.get('type')} - {config.model.get('name')}")
    else:
        print("  Model: None (model-free checks)")

    print(f"  Checks: {', '.join(c['name'] for c in config.checks)}")
    print(
        f"  Runs: {config.n_runs}" + (" (multi-run with statistics)" if config.n_runs > 1 else "")
    )
    if config.seed is not None:
        print(f"  Seed: {config.seed}")
    print(f"  Output: {config.get_output_dir()}")
    print(f"  Progress: {args.progress}")

    if args.dry_run:
        print("\n✅ Dry run complete (config valid)")
        return 0

    # Create global model and checks (shared across all benchmarks)
    print("\n🔧 Initializing components...")

    try:
        default_model = config.get_model()
        if default_model:
            print(f"  ✅ Default Model: {default_model.__class__.__name__}")
        else:
            print("  ✅ Default Model: None")
    except Exception as e:
        print(f"  ❌ Error creating default model: {e}")
        return 1

    try:
        checks = config.get_checks(default_model=default_model)
        for check in checks:
            # Show which model each check uses
            check_model = getattr(check, "model", None) if hasattr(check, "__dict__") else None
            if check_model:
                model_name = check_model.__class__.__name__
            else:
                model_name = "None (model-free)"
            print(f"  ✅ Check: {check.__class__.__name__} [model: {model_name}]")
    except Exception as e:
        print(f"  ❌ Error creating checks: {e}")
        return 1

    # Run evaluation(s)
    print("\n🚀 Running evaluation...")
    print("-" * 60)

    all_results = {}

    try:
        # Get list of benchmarks (single or multiple)
        benchmarks_list = config.get_benchmarks_list()

        for bench_idx, bench_cfg in enumerate(benchmarks_list, 1):
            bench_name = bench_cfg.get("name", f"benchmark_{bench_idx}")

            if config.is_multi_benchmark():
                print(f"\n[{bench_idx}/{len(benchmarks_list)}] Running: {bench_name}")
                print("-" * 60)

            # Create adapter for this benchmark
            try:
                adapter = config.get_adapter(bench_cfg)
                print(f"  ✅ Adapter: {adapter.__class__.__name__}")
            except Exception as e:
                print(f"  ❌ Error creating adapter for {bench_name}: {e}")
                continue

            # Run checks on this benchmark
            runner = BencheckRunner(
                adapter=adapter,
                checks=checks,
                benchmark_name=bench_name,
                n_runs=config.n_runs,
                seed=config.seed,
            )

            results = runner.run()
            all_results[bench_name] = results

            # Display results summary for this benchmark
            if config.is_multi_benchmark():
                print(f"\n  Results for {bench_name}:")
                # Handle both single-run and multi-run results
                if isinstance(results, MultiRunEvaluation):
                    print(f"    (Multi-run: {results.n_runs} runs, showing aggregated stats)")
                    for check_name, check_stats in results.metrics_stats.items():
                        print(f"    {check_name}:")
                        for metric_name, stats in check_stats.items():
                            if isinstance(stats, dict) and "mean" in stats:
                                print(
                                    f"      {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}"
                                )
                            else:
                                print(f"      {metric_name}: {stats}")
                else:
                    for check_name, metrics in results.metrics.items():
                        print(f"    {check_name}:")
                        for metric_name, value in metrics.items():
                            if isinstance(value, float):
                                print(f"      {metric_name}: {value:.4f}")
                            else:
                                print(f"      {metric_name}: {value}")

        # Display final results summary
        print("\n" + "=" * 60)
        if config.is_multi_benchmark():
            print(f"Results Summary (All {len(all_results)} Benchmarks)")
        else:
            print("Results Summary")
        print("=" * 60)

        # Show aggregated table for multi-benchmark runs
        if config.is_multi_benchmark() and len(all_results) > 1:
            print("\n📊 Aggregated Summary Table:\n")
            summary_table = create_summary_table(all_results, format="markdown")
            print(summary_table)
            print("\n" + "-" * 60 + "\n")

        # Show detailed results for each benchmark
        for bench_name, results in all_results.items():
            if config.is_multi_benchmark():
                print(f"\n[{bench_name}]")

            # Handle both single-run and multi-run results
            if isinstance(results, MultiRunEvaluation):
                print(f"\n📊 Multi-Run Statistics ({results.n_runs} runs):")
                for check_name, check_stats in results.metrics_stats.items():
                    print(f"\n{check_name}:")
                    for metric_name, stats in check_stats.items():
                        if isinstance(stats, dict) and "mean" in stats:
                            print(f"  {metric_name}:")
                            print(f"    mean: {stats['mean']:.4f}")
                            print(f"    std:  {stats['std']:.4f}")
                            print(f"    min:  {stats['min']:.4f}")
                            print(f"    max:  {stats['max']:.4f}")
                            print(
                                f"    95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]"
                            )
                        else:
                            print(f"  {metric_name}: {stats}")
            else:
                for check_name, metrics in results.metrics.items():
                    print(f"\n{check_name}:")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            print(f"  {metric_name}: {value:.4f}")
                        else:
                            print(f"  {metric_name}: {value}")

                # Display per-question results if requested (only for single benchmark, single-run)
                if not config.is_multi_benchmark():
                    display_question_results(
                        results, show_count=args.show_questions, show_errors_only=args.show_errors
                    )

            # Save results for this benchmark
            save_results(results, config)

        print("\n" + "=" * 60)
        print("✅ Evaluation complete!")
        if config.is_multi_benchmark():
            print(f"   Processed {len(all_results)} benchmarks")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
