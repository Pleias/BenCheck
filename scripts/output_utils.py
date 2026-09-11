"""
Utilities for saving experiment results in the new output structure.

This module provides functions to save results in the new flat structure:
- questions_base.json: Base question info
- {check_name}.json: Summary metrics for each check
- per_questions/{check_name}.json: Per-question details
- experiment_summary.json: Aggregated summary across all datasets
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bencheck.types import BencheckQuestion
from bencheck.utils.statistics import compute_metric_statistics


def save_json(path: Path, data: Any) -> None:
    """Save data to JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


def save_questions_base(path: Path, questions: List[BencheckQuestion]) -> None:
    """
    Save base question info without check results.

    Args:
        path: Path to save questions_base.json
        questions: List of BencheckQuestion objects

    Output format:
        {
            "0": {
                "question": "...",
                "choices": ["...", "..."],
                "correct": 1,
                "question_type": "single_choice",
                "metadata": {...}
            },
            ...
        }
    """
    data = {}
    for i, q in enumerate(questions):
        data[str(i)] = {
            "question": q.question,
            "choices": q.choices,
            "correct": q.correct,
            "question_type": q.question_type.value,
            "metadata": q.metadata
        }
    save_json(path, data)


def save_check_result(
    path: Path,
    check_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]]
) -> None:
    """
    Save check summary metrics with metadata.

    Args:
        path: Path to save check result
        check_data: Check metrics dict
        metadata: Metadata dict with model/mode/check_type, or None if already in check_data

    Output format for model-free/scoring-independent:
        {
            "_metadata": {
                "model": null or "/models/...",
                "mode": null,
                "check_type": "model_free" | "scoring_independent"
            },
            "n_questions": 1838,
            "metric1": value1,
            ...
        }

    Output format for scoring-dependent:
        {
            "_metadata": {
                "model": "/models/...",
                "mode": null,
                "check_type": "scoring_dependent"
            },
            "log_likelihood": {
                "runs": [...],
                "aggregated": {...}
            },
            "generation": {
                "runs": [...],
                "aggregated": {...}
            }
        }
    """
    if metadata is not None:
        # Add metadata to check_data
        data = {
            "_metadata": metadata,
            **check_data
        }
    else:
        # Metadata already in check_data (for scoring-dependent)
        data = check_data

    save_json(path, data)


def save_per_question(
    path: Path,
    question_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Save per-question check results.

    Args:
        path: Path to save per-question results
        question_results: Dict mapping question_id to check-specific results

    Output format:
        {
            "0": {
                "metric1": value1,
                "metric2": value2,
                ...
            },
            "1": {...},
            ...
        }
    """
    save_json(path, question_results)


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple runs into summary statistics.

    Uses compute_metric_statistics() which returns:
    - Simple value if std < 1e-6 (constant across runs)
    - Stats dict with mean/std/ci/etc otherwise

    Args:
        runs: List of dicts containing run results
              Each dict should have run_index and metric values

    Returns:
        Aggregated dict with same keys as runs (except run_index)

    Example:
        Input: [
            {"run_index": 0, "n_questions": 1838, "accuracy": 75.2},
            {"run_index": 1, "n_questions": 1838, "accuracy": 75.3},
            {"run_index": 2, "n_questions": 1838, "accuracy": 75.1}
        ]
        Output: {
            "n_questions": 1838,  # Simplified (constant)
            "accuracy": 75.2  # Or stats dict if variance is significant
        }
    """
    if not runs:
        return {}

    aggregated = {}

    # Get all metric names (exclude run_index and internal fields)
    metric_names = set()
    for run in runs:
        metric_names.update(run.keys())

    # Remove run_index and any internal fields
    metric_names.discard("run_index")
    metric_names.discard("_metadata")

    for metric_name in metric_names:
        values = [run[metric_name] for run in runs if metric_name in run]

        if not values:
            continue

        # Only aggregate numeric values
        if all(isinstance(v, (int, float)) for v in values):
            # Use compute_metric_statistics which handles constants
            aggregated[metric_name] = compute_metric_statistics(values)
        else:
            # For non-numeric values, take the first one
            # (should be the same across all runs for things like scoring_mode)
            aggregated[metric_name] = values[0]

    return aggregated


def create_experiment_summary(
    output_dir: Path,
    config_path: str
) -> None:
    """
    Create experiment_summary.json aggregating all datasets.

    Args:
        output_dir: Directory containing dataset subdirectories
        config_path: Path to experiment config file

    Output structure:
        {
            "experiment_config": "configs/...",
            "timestamp": "2025-12-27T...",
            "datasets": {
                "piqa_validation": {
                    "n_questions": 1838,
                    "checks": {
                        "length_bias": {...},
                        "enumeration_bias": {...},
                        ...
                    }
                },
                ...
            }
        }
    """
    summary = {
        "experiment_config": str(config_path),
        "timestamp": datetime.now().isoformat(),
        "datasets": {}
    }

    # Find all dataset directories (end with _)
    for dataset_dir in sorted(output_dir.glob("*_/")):
        dataset_name = dataset_dir.name.rstrip("_")

        # Load questions_base for n_questions
        questions_base_file = dataset_dir / "questions_base.json"
        if not questions_base_file.exists():
            continue

        questions_base = load_json(questions_base_file)

        # Load all check files
        checks_data = {}
        for check_file in sorted(dataset_dir.glob("*.json")):
            if check_file.name != "questions_base.json":
                check_name = check_file.stem
                checks_data[check_name] = load_json(check_file)

        summary["datasets"][dataset_name] = {
            "n_questions": len(questions_base),
            "checks": checks_data
        }

    # Save summary
    save_json(output_dir / "experiment_summary.json", summary)


def load_or_create_check_file(
    path: Path,
    default_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Load existing check file or create new one with metadata.

    Args:
        path: Path to check file
        default_metadata: Default metadata to use if file doesn't exist

    Returns:
        Loaded or newly created data dict
    """
    if path.exists():
        return load_json(path)
    else:
        return {"_metadata": default_metadata}


def check_metadata_matches(
    existing_metadata: Dict[str, Any],
    expected_model: str,
    expected_mode: Optional[str] = None
) -> bool:
    """
    Check if existing check metadata matches expected model/mode.

    Args:
        existing_metadata: Metadata from existing check file
        expected_model: Expected model name
        expected_mode: Expected mode (or None for scoring-independent)

    Returns:
        True if metadata matches, False otherwise
    """
    if existing_metadata.get("model") != expected_model:
        return False

    if expected_mode is not None and existing_metadata.get("mode") != expected_mode:
        return False

    return True
