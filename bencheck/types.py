"""Core data structures for BenCheck."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class QuestionType(Enum):
    """Type of question format."""

    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_ENDED = "open_ended"


@dataclass
class BencheckQuestion:
    """
    Standardized question representation.

    The `correct` field is flexible to support different benchmark formats:
    - Single choice: correct=1 or correct=[1] or correct="4"
    - Multiple choice: correct=[0, 2, 3] or correct=["1", "3", "4"]
    - Open-ended (future): correct="some text answer"

    **Important**: Adapters are responsible for loading data in the
    benchmark's native format. No automatic conversion of indices
    (0-based vs 1-based) - adapters handle this explicitly.
    """

    id: str
    question: str
    choices: List[str]
    correct: Union[int, List[int], str, List[str]]
    question_type: QuestionType = QuestionType.SINGLE_CHOICE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    """
    Output from running a diagnostic check on a dataset.

    Attributes:
        check_name: Name of the diagnostic check
        question_results: Per-question diagnostic data (question_id → diagnostics)
        metrics: Dataset-level metrics. Values can be:
            - float: single value (e.g., accuracy: 85.0)
            - tuple: multiple values (e.g., mean±std: (85.0, 3.2))
            - dict: nested metrics (e.g., by_category: {"A": 90.0, "B": 80.0})
    """

    check_name: str
    question_results: Dict[str, Dict[str, Any]]  # question_id → diagnostics
    metrics: Dict[str, Union[float, tuple, Dict[str, Any]]]  # metric_name → value(s)


@dataclass
class BenchmarkEvaluation:
    """
    Evaluation results for ONE benchmark + ONE model + multiple checks.

    This is the primary output structure. For comparing multiple benchmarks
    or models, use lists of BenchmarkEvaluation objects with utilities in
    bencheck.utils.multi_eval.

    Attributes:
        benchmark_name: Name of the benchmark
        model_config: Model metadata (name, class, etc.)
        questions: Per-question data including all check results
        metrics: Dataset-level metrics from all checks.
            Structure: check_name → {metric_name → value(s)}
            Values can be float, tuple, or dict for complex metrics.
    """

    benchmark_name: str
    model_config: Dict[str, Any]
    questions: Dict[str, Any]  # question_id → {question, choices, correct, checks}
    metrics: Dict[str, Dict[str, Any]]  # check_name → {metric: value(s)}


@dataclass
class MultiRunEvaluation:
    """
    Results from N runs of the same benchmark configuration with aggregated statistics.

    This type is returned when BencheckRunner is executed with n_runs > 1.
    It contains both individual run results and aggregated statistics across runs.

    Attributes:
        benchmark_name: Name of the benchmark
        model_config: Model configuration metadata
        scoring_mode: Scoring mode used ("log_likelihood", "generation", or None for model-free)
        n_runs: Number of runs executed
        seed: Initial random seed used (each run uses seed + run_index)
        individual_runs: List of BenchmarkEvaluation objects, one per run
        metrics_stats: Aggregated statistics across runs
            Structure: check_name → metric_name → {mean, std, ci_95_lower, ci_95_upper, min, max, median, raw_values}
        summary: High-level summary statistics
    """

    benchmark_name: str
    model_config: Dict[str, Any]
    scoring_mode: Optional[str]
    n_runs: int
    seed: Optional[int]
    individual_runs: List["BenchmarkEvaluation"]
    metrics_stats: Dict[str, Dict[str, Dict[str, Any]]]  # check → metric → stats
    summary: Dict[str, Any]
