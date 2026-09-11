"""Statistical utilities for multi-run aggregation."""

from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def compute_metric_statistics(values: List[float], epsilon: float = 1e-6) -> Any:
    """
    Compute comprehensive statistics for a metric across runs.

    If all values are identical (std < epsilon), returns just the constant value
    to avoid cluttering output with redundant statistics.

    Args:
        values: List of metric values from different runs
        epsilon: Threshold for considering values constant (default: 1e-6)

    Returns:
        If values are constant (std < epsilon):
            The constant value (float or int)
        Otherwise:
            Dictionary with statistics:
            - mean: Mean value
            - std: Standard deviation
            - median: Median value
            - min: Minimum value
            - max: Maximum value
            - ci_95_lower: Lower bound of 95% confidence interval
            - ci_95_upper: Upper bound of 95% confidence interval
            - raw_values: Original values
            - n_runs: Number of runs
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
            "raw_values": [],
            "n_runs": 0,
        }

    n = len(values)

    if HAS_NUMPY:
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        median = float(np.median(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
    else:
        # Pure Python fallback
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = variance**0.5
        else:
            std = 0.0
        sorted_vals = sorted(values)
        median = (
            sorted_vals[n // 2]
            if n % 2 == 1
            else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        )
        min_val = min(values)
        max_val = max(values)

    # If std < epsilon, all values are essentially identical - return just the value
    if std < epsilon:
        # Return as int if it's a whole number, otherwise float
        rounded_val = round(mean, 4)
        if abs(rounded_val - round(rounded_val)) < epsilon:
            return int(round(rounded_val))
        return rounded_val

    # Compute bootstrap confidence interval
    ci_lower, ci_upper = bootstrap_confidence_interval(values)

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(median, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "raw_values": [round(v, 4) for v in values],
        "n_runs": n,
    }


def bootstrap_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval using resampling.

    Args:
        values: Original data values
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        # Can't compute CI with < 2 values
        val = values[0] if values else 0.0
        return val, val

    if HAS_NUMPY:
        rng = np.random.default_rng(seed)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            resample = rng.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(resample))

        # Compute percentiles
        alpha = (1 - confidence) / 2
        lower = float(np.percentile(bootstrap_means, alpha * 100))
        upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))

        return lower, upper
    else:
        # Pure Python fallback - use approximate CI based on t-distribution
        import math

        n = len(values)
        mean = sum(values) / n

        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = variance**0.5

            # Approximate t-value for 95% CI (good enough for n > 5)
            t_value = 2.0 if n >= 30 else 2.5
            margin = t_value * std / math.sqrt(n)

            return mean - margin, mean + margin
        else:
            return mean, mean


def aggregate_check_results_across_runs(
    run_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate all metrics from multiple run results.

    Args:
        run_results: List of metrics dicts from each run
            Structure: [{check_name: {metric_name: value, ...}, ...}, ...]

    Returns:
        Aggregated statistics
        Structure: {check_name: {metric_name: {stats}, ...}, ...}
    """
    if not run_results:
        return {}

    aggregated = {}

    # Get all check names from first run
    check_names = list(run_results[0].keys())

    for check_name in check_names:
        aggregated[check_name] = {}

        # Get all metric names for this check
        if check_name in run_results[0]:
            metric_names = list(run_results[0][check_name].keys())

            for metric_name in metric_names:
                # Collect values across runs
                values = []
                for run_result in run_results:
                    if check_name in run_result and metric_name in run_result[check_name]:
                        value = run_result[check_name][metric_name]

                        # Handle different value types
                        if isinstance(value, (int, float)):
                            values.append(float(value))
                        elif isinstance(value, list):
                            # For distributions, skip aggregation (too complex)
                            # Could implement element-wise aggregation if needed
                            continue
                        elif isinstance(value, dict):
                            # For nested dicts, skip or recurse
                            continue

                # Compute statistics if we have numeric values
                if values:
                    aggregated[check_name][metric_name] = compute_metric_statistics(values)

    return aggregated


def aggregate_metric_list(
    metric_values: List[List[Any]],
) -> Dict[str, Any]:
    """
    Aggregate metrics that are lists (e.g., distributions).

    For distribution metrics like [10, 20, 30, 40] representing counts per position,
    compute element-wise statistics.

    Args:
        metric_values: List of list values from each run

    Returns:
        Dict with element-wise statistics or original if not aggregatable
    """
    if not metric_values or not all(isinstance(v, list) for v in metric_values):
        return {"note": "Cannot aggregate non-list values"}

    # Check all lists have same length
    lengths = [len(v) for v in metric_values]
    if len(set(lengths)) > 1:
        return {"note": "Lists have different lengths, cannot aggregate"}

    if not lengths:
        return {"note": "Empty lists"}

    length = lengths[0]
    aggregated_list = []

    # Aggregate element-wise
    for idx in range(length):
        element_values = [
            run_val[idx] for run_val in metric_values if isinstance(run_val[idx], (int, float))
        ]
        if element_values:
            stats = compute_metric_statistics(element_values)
            aggregated_list.append(
                {"mean": stats["mean"], "std": stats["std"], "raw": element_values}
            )
        else:
            aggregated_list.append(None)

    return {"element_wise_stats": aggregated_list}
