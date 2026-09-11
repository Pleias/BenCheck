"""Summary table utilities for aggregating multi-benchmark results."""

from typing import Any, Dict, List

from ..types import BenchmarkEvaluation


def create_summary_table(results: Dict[str, BenchmarkEvaluation], format: str = "markdown") -> str:
    """
    Create summary table from multiple benchmark results.

    Args:
        results: Dict mapping benchmark names to BenchmarkEvaluation objects
        format: Output format - "markdown", "csv", or "dict"

    Returns:
        Formatted summary table

    Example:
        >>> results = {
        ...     "hellaswag": hellaswag_eval,
        ...     "mmlu": mmlu_eval
        ... }
        >>> print(create_summary_table(results))
    """
    if not results:
        return "No results to summarize"

    # Collect all check names
    all_checks = set()
    for result in results.values():
        all_checks.update(result.metrics.keys())

    all_checks = sorted(all_checks)

    # Build summary data
    summary_data = []
    for bench_name, result in sorted(results.items()):
        row = {"benchmark": bench_name}

        for check_name in all_checks:
            if check_name in result.metrics:
                metrics = result.metrics[check_name]

                # Extract key metrics for each check
                if check_name == "length_bias":
                    row["length_corr"] = metrics.get("length_correlation", "N/A")
                    row["mean_length"] = metrics.get("mean_length", "N/A")

                elif check_name == "enumeration_bias":
                    row["accuracy"] = metrics.get("overall_accuracy", "N/A")
                    row["chi2_p"] = metrics.get("chi2_p_value", "N/A")

                elif check_name == "none_of_above":
                    row["nota_acc"] = metrics.get("accuracy", "N/A")

                elif check_name == "context_requirement":
                    row["ctx_req"] = metrics.get("accuracy_empty", "N/A")
                    row["ctx_delta"] = metrics.get("accuracy_delta", "N/A")

            else:
                # Check not run for this benchmark
                if check_name == "length_bias":
                    row["length_corr"] = "N/A"
                    row["mean_length"] = "N/A"
                elif check_name == "enumeration_bias":
                    row["accuracy"] = "N/A"
                    row["chi2_p"] = "N/A"
                elif check_name == "none_of_above":
                    row["nota_acc"] = "N/A"
                elif check_name == "context_requirement":
                    row["ctx_req"] = "N/A"
                    row["ctx_delta"] = "N/A"

        summary_data.append(row)

    # Format output
    if format == "markdown":
        return _format_markdown(summary_data)
    elif format == "csv":
        return _format_csv(summary_data)
    elif format == "dict":
        return summary_data
    else:
        raise ValueError(f"Unknown format: {format}")


def _format_markdown(data: List[Dict[str, Any]]) -> str:
    """Format summary as markdown table."""
    if not data:
        return "No data"

    # Get all column names
    all_cols = set()
    for row in data:
        all_cols.update(row.keys())

    # Order columns
    col_order = [
        "benchmark",
        "length_corr",
        "mean_length",
        "accuracy",
        "chi2_p",
        "nota_acc",
        "ctx_req",
        "ctx_delta",
    ]
    cols = [c for c in col_order if c in all_cols]

    # Rename for display
    display_names = {
        "benchmark": "Benchmark",
        "length_corr": "Length Corr",
        "mean_length": "Mean Length",
        "accuracy": "Accuracy (%)",
        "chi2_p": "Chi² p-value",
        "nota_acc": "NOTA Acc (%)",
        "ctx_req": "Context Req (%)",
        "ctx_delta": "Ctx Delta (%)",
    }

    # Build table
    lines = []

    # Header
    header = "| " + " | ".join(display_names.get(c, c) for c in cols) + " |"
    separator = "|" + "|".join("-" * (len(display_names.get(c, c)) + 2) for c in cols) + "|"

    lines.append(header)
    lines.append(separator)

    # Rows
    for row in data:
        values = []
        for col in cols:
            val = row.get(col, "N/A")

            # Format values
            if col in ["accuracy", "nota_acc", "ctx_req"]:
                if isinstance(val, (int, float)):
                    val = f"{val * 100:.1f}"
            elif col in ["length_corr", "chi2_p", "ctx_delta"]:
                if isinstance(val, float):
                    val = f"{val:.3f}"
            elif col == "mean_length":
                if isinstance(val, float):
                    val = f"{val:.1f}"

            values.append(str(val))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _format_csv(data: List[Dict[str, Any]]) -> str:
    """Format summary as CSV."""
    if not data:
        return ""

    # Get all columns
    all_cols = set()
    for row in data:
        all_cols.update(row.keys())

    cols = sorted(all_cols)
    if "benchmark" in cols:
        cols.remove("benchmark")
        cols = ["benchmark"] + cols

    # Header
    lines = [",".join(cols)]

    # Rows
    for row in data:
        values = [str(row.get(col, "N/A")) for col in cols]
        lines.append(",".join(values))

    return "\n".join(lines)


def create_compact_summary(results: Dict[str, BenchmarkEvaluation]) -> str:
    """
    Create very compact summary - one line per benchmark.

    Args:
        results: Dict mapping benchmark names to results

    Returns:
        Compact summary string

    Example output:
        HellaSwag: LengthCorr=0.34, Acc=45.2%, Chi²=0.023, CtxReq=78.3%
        MMLU: LengthCorr=0.12, Acc=52.1%, Chi²=0.451, CtxReq=45.1%
    """
    lines = []

    for bench_name, result in sorted(results.items()):
        parts = [f"{bench_name}:"]

        for check_name, metrics in result.metrics.items():
            if check_name == "length_bias":
                corr = metrics.get("length_correlation")
                if corr is not None:
                    parts.append(f"LengthCorr={corr:.2f}")

            elif check_name == "enumeration_bias":
                acc = metrics.get("overall_accuracy")
                chi2 = metrics.get("chi2_p_value")
                if acc is not None:
                    parts.append(f"Acc={acc * 100:.1f}%")
                if chi2 is not None:
                    parts.append(f"Chi²={chi2:.3f}")

            elif check_name == "context_requirement":
                ctx = metrics.get("accuracy_empty")
                if ctx is not None:
                    parts.append(f"CtxReq={ctx * 100:.1f}%")

        lines.append(" ".join(parts))

    return "\n".join(lines)


def print_summary(results: Dict[str, BenchmarkEvaluation], style: str = "table"):
    """
    Print formatted summary to console.

    Args:
        results: Dict mapping benchmark names to results
        style: "table" (markdown table), "compact" (one-liners), or "full" (detailed)
    """
    if style == "table":
        print(create_summary_table(results, format="markdown"))
    elif style == "compact":
        print(create_compact_summary(results))
    elif style == "full":
        for bench_name, result in sorted(results.items()):
            print(f"\n{'=' * 60}")
            print(f"{bench_name}")
            print("=" * 60)

            for check_name, metrics in result.metrics.items():
                print(f"\n{check_name}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
    else:
        raise ValueError(f"Unknown style: {style}")
