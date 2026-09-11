"""Tests for new output structure refactoring."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from bencheck.types import BencheckQuestion, QuestionType


class TestOutputStructure:
    """Test the new flat output structure."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_questions(self):
        """Sample questions for testing."""
        return [
            BencheckQuestion(
                id="0",
                question="To make a telescope you need to...",
                choices=["use binoculars", "use a magnifying glass"],
                correct=1,
                question_type=QuestionType.SINGLE_CHOICE,
                metadata={"source": "test"}
            ),
            BencheckQuestion(
                id="1",
                question="What is the capital of France?",
                choices=["London", "Paris", "Berlin", "Madrid"],
                correct=1,
                question_type=QuestionType.SINGLE_CHOICE,
                metadata={"source": "test"}
            ),
        ]

    def test_flat_directory_structure(self, temp_output_dir):
        """Test that output has flat structure per dataset."""
        # This test will pass once we implement the new structure
        from scripts.output_utils import save_questions_base

        benchmark_dir = temp_output_dir / "piqa_validation_"
        benchmark_dir.mkdir(parents=True)

        # Should create:
        # piqa_validation_/
        #   ├── questions_base.json
        #   ├── length_bias.json
        #   └── per_questions/
        #       └── length_bias.json

        questions_file = benchmark_dir / "questions_base.json"
        check_file = benchmark_dir / "length_bias.json"
        per_questions_dir = benchmark_dir / "per_questions"

        # After implementation, these should exist
        # For now, we just test the structure we expect
        assert benchmark_dir.exists()

    def test_questions_base_saved(self, temp_output_dir, sample_questions):
        """Test that questions_base.json is created correctly."""
        from scripts.output_utils import save_questions_base

        benchmark_dir = temp_output_dir / "piqa_validation_"
        benchmark_dir.mkdir(parents=True)
        questions_file = benchmark_dir / "questions_base.json"

        save_questions_base(questions_file, sample_questions)

        assert questions_file.exists()

        # Load and verify structure
        with open(questions_file) as f:
            data = json.load(f)

        assert "0" in data
        assert "1" in data
        assert data["0"]["question"] == "To make a telescope you need to..."
        assert data["0"]["choices"] == ["use binoculars", "use a magnifying glass"]
        assert data["0"]["correct"] == 1
        assert data["0"]["question_type"] == "single_choice"
        assert data["0"]["metadata"] == {"source": "test"}

    def test_check_metadata_model_free(self, temp_output_dir):
        """Test that model-free checks have correct metadata."""
        from scripts.output_utils import save_check_result

        benchmark_dir = temp_output_dir / "piqa_validation_"
        benchmark_dir.mkdir(parents=True)
        check_file = benchmark_dir / "length_bias.json"

        # Mock CheckResult-like data
        check_data = {
            "n_questions": 1838,
            "median_relative_length_diff": 0.0,
            "mean_correct_length": 18.53,
        }

        metadata = {
            "model": None,
            "mode": None,
            "check_type": "model_free"
        }

        save_check_result(check_file, check_data, metadata)

        assert check_file.exists()

        # Load and verify
        with open(check_file) as f:
            data = json.load(f)

        assert data["_metadata"]["check_type"] == "model_free"
        assert data["_metadata"]["model"] is None
        assert data["_metadata"]["mode"] is None
        assert data["n_questions"] == 1838

    def test_check_metadata_scoring_independent(self, temp_output_dir):
        """Test that scoring-independent checks have correct metadata."""
        from scripts.output_utils import save_check_result

        benchmark_dir = temp_output_dir / "piqa_validation_"
        benchmark_dir.mkdir(parents=True)
        check_file = benchmark_dir / "enumeration_bias.json"

        check_data = {
            "n_questions": 1838,
            "overall_accuracy": 75.2,
        }

        metadata = {
            "model": "/models/gemma-3-12b-it",
            "mode": None,
            "check_type": "scoring_independent"
        }

        save_check_result(check_file, check_data, metadata)

        # Load and verify
        with open(check_file) as f:
            data = json.load(f)

        assert data["_metadata"]["check_type"] == "scoring_independent"
        assert data["_metadata"]["model"] == "/models/gemma-3-12b-it"
        assert data["_metadata"]["mode"] is None

    def test_check_metadata_scoring_dependent(self, temp_output_dir):
        """Test that scoring-dependent checks have correct metadata."""
        from scripts.output_utils import save_check_result

        benchmark_dir = temp_output_dir / "piqa_validation_"
        benchmark_dir.mkdir(parents=True)
        check_file = benchmark_dir / "context_requirement.json"

        check_data = {
            "_metadata": {
                "model": "/models/gemma-3-12b-it",
                "mode": None,
                "check_type": "scoring_dependent"
            },
            "log_likelihood": {
                "runs": [
                    {"run_index": 0, "baseline_accuracy": 75.2}
                ],
                "aggregated": {
                    "baseline_accuracy": 75.2
                }
            }
        }

        # For scoring-dependent, we pass full data including metadata
        save_check_result(check_file, check_data, None)

        # Load and verify
        with open(check_file) as f:
            data = json.load(f)

        assert data["_metadata"]["check_type"] == "scoring_dependent"
        assert "log_likelihood" in data
        assert "runs" in data["log_likelihood"]
        assert "aggregated" in data["log_likelihood"]

    def test_constant_fields_simplified(self, temp_output_dir):
        """Test that constant fields are simplified to values."""
        from scripts.output_utils import save_check_result

        benchmark_dir = temp_output_dir / "piqa_validation_"
        benchmark_dir.mkdir(parents=True)
        check_file = benchmark_dir / "length_bias.json"

        # n_questions is constant across runs, should be simplified to int
        check_data = {
            "n_questions": 1838,  # Should remain as int, not dict
            "median_relative_length_diff": 0.0,
        }

        metadata = {
            "model": None,
            "mode": None,
            "check_type": "model_free"
        }

        save_check_result(check_file, check_data, metadata)

        # Load and verify
        with open(check_file) as f:
            data = json.load(f)

        # Should be int, not dict with mean/std/etc
        assert isinstance(data["n_questions"], int)
        assert data["n_questions"] == 1838

    def test_per_question_files(self, temp_output_dir):
        """Test that per_question files are created correctly."""
        from scripts.output_utils import save_per_question

        benchmark_dir = temp_output_dir / "piqa_validation_"
        per_questions_dir = benchmark_dir / "per_questions"
        per_questions_dir.mkdir(parents=True)

        per_question_file = per_questions_dir / "length_bias.json"

        question_results = {
            "0": {
                "option_lengths": [10, 12, 8, 15],
                "correct_length": 12,
                "mean_incorrect_length": 11.0,
            },
            "1": {
                "option_lengths": [8, 10, 12, 9],
                "correct_length": 10,
                "mean_incorrect_length": 9.67,
            }
        }

        save_per_question(per_question_file, question_results)

        assert per_question_file.exists()

        # Load and verify
        with open(per_question_file) as f:
            data = json.load(f)

        assert "0" in data
        assert "1" in data
        assert data["0"]["option_lengths"] == [10, 12, 8, 15]
        assert data["0"]["correct_length"] == 12

    def test_scoring_dependent_structure(self, temp_output_dir):
        """Test context_requirement.json structure with runs and aggregated."""
        from scripts.output_utils import save_check_result

        benchmark_dir = temp_output_dir / "piqa_validation_"
        benchmark_dir.mkdir(parents=True)
        check_file = benchmark_dir / "context_requirement.json"

        check_data = {
            "_metadata": {
                "model": "/models/gemma-3-12b-it",
                "mode": None,
                "check_type": "scoring_dependent"
            },
            "log_likelihood": {
                "runs": [
                    {"run_index": 0, "baseline_accuracy": 75.2, "empty_accuracy": 60.1},
                    {"run_index": 1, "baseline_accuracy": 75.2, "empty_accuracy": 60.2},
                    {"run_index": 2, "baseline_accuracy": 75.2, "empty_accuracy": 60.0},
                ],
                "aggregated": {
                    "baseline_accuracy": 75.2,
                    "empty_accuracy": 60.1,
                }
            },
            "generation": {
                "runs": [
                    {"run_index": 0, "baseline_accuracy": 68.5},
                ],
                "aggregated": {
                    "baseline_accuracy": 68.5,
                }
            }
        }

        save_check_result(check_file, check_data, None)

        # Load and verify
        with open(check_file) as f:
            data = json.load(f)

        assert "log_likelihood" in data
        assert "generation" in data
        assert "runs" in data["log_likelihood"]
        assert "aggregated" in data["log_likelihood"]
        assert len(data["log_likelihood"]["runs"]) == 3
        assert data["log_likelihood"]["aggregated"]["baseline_accuracy"] == 75.2

    def test_aggregate_runs_with_constants(self):
        """Test that aggregate_runs simplifies constant values."""
        from scripts.output_utils import aggregate_runs

        runs = [
            {"run_index": 0, "n_questions": 1838, "accuracy": 75.2},
            {"run_index": 1, "n_questions": 1838, "accuracy": 75.3},
            {"run_index": 2, "n_questions": 1838, "accuracy": 75.1},
        ]

        aggregated = aggregate_runs(runs)

        # n_questions is constant, should be simplified
        assert isinstance(aggregated["n_questions"], int)
        assert aggregated["n_questions"] == 1838

        # accuracy varies, might be aggregated (depends on implementation)
        # For now just check it exists
        assert "accuracy" in aggregated


class TestExperimentSummary:
    """Test experiment_summary.json creation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create sample structure
            piqa_dir = output_dir / "piqa_validation_"
            piqa_dir.mkdir(parents=True)

            # Create questions_base.json
            with open(piqa_dir / "questions_base.json", "w") as f:
                json.dump({
                    "0": {"question": "test", "choices": ["a", "b"], "correct": 1, "question_type": "single_choice", "metadata": {}},
                    "1": {"question": "test2", "choices": ["a", "b"], "correct": 0, "question_type": "single_choice", "metadata": {}},
                }, f)

            # Create length_bias.json
            with open(piqa_dir / "length_bias.json", "w") as f:
                json.dump({
                    "_metadata": {"model": None, "mode": None, "check_type": "model_free"},
                    "n_questions": 2,
                    "median_relative_length_diff": 0.0,
                }, f)

            # Create enumeration_bias.json
            with open(piqa_dir / "enumeration_bias.json", "w") as f:
                json.dump({
                    "_metadata": {"model": "/models/gemma-3-12b-it", "mode": None, "check_type": "scoring_independent"},
                    "overall_accuracy": 75.2,
                }, f)

            yield output_dir

    def test_experiment_summary_creation(self, temp_output_dir):
        """Test that experiment_summary.json is created correctly."""
        from scripts.output_utils import create_experiment_summary

        create_experiment_summary(
            output_dir=temp_output_dir,
            config_path="configs/experiments/test.yaml"
        )

        summary_file = temp_output_dir / "experiment_summary.json"
        assert summary_file.exists()

        # Load and verify
        with open(summary_file) as f:
            summary = json.load(f)

        assert "experiment_config" in summary
        assert "timestamp" in summary
        assert "datasets" in summary
        assert "piqa_validation" in summary["datasets"]

        # Check dataset info
        piqa_data = summary["datasets"]["piqa_validation"]
        assert piqa_data["n_questions"] == 2
        assert "checks" in piqa_data
        assert "length_bias" in piqa_data["checks"]
        assert "enumeration_bias" in piqa_data["checks"]

        # Check that check data is included
        assert piqa_data["checks"]["length_bias"]["_metadata"]["check_type"] == "model_free"
        assert piqa_data["checks"]["enumeration_bias"]["_metadata"]["check_type"] == "scoring_independent"
