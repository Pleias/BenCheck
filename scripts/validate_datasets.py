#!/usr/bin/env python3
"""Validate dataset availability and format compatibility.

This script checks that all configured benchmark datasets:
1. Are accessible from HuggingFace
2. Have the expected format for their adapters
3. Can be loaded and converted successfully

Usage:
    python scripts/validate_datasets.py
    python scripts/validate_datasets.py --datasets hellaswag piqa
    python scripts/validate_datasets.py --quick  # Load only 10 samples per dataset
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from bencheck.adapters import (
    ARC_Adapter,
    BigBenchAdapter,
    CommonsenseQAAdapter,
    GlobalPIQAAdapter,
    HellaSwagAdapter,
    MMLUAdapter,
    PIQAAdapter,
    TruthfulQAAdapter,
    WinograndeAdapter,
)


@dataclass
class DatasetConfig:
    """Configuration for a dataset to validate."""

    name: str
    adapter_class: Any
    adapter_kwargs: Dict[str, Any]
    dataset_name: str
    config_name: Optional[str] = None
    split: str = "validation"
    min_samples: int = 10
    description: str = ""


# Benchmark dataset configurations
DATASET_CONFIGS = [
    DatasetConfig(
        name="piqa",
        adapter_class=PIQAAdapter,
        adapter_kwargs={"split": "validation"},
        dataset_name="regisss/piqa",
        split="validation",
        min_samples=10,
        description="Physical Interaction QA - commonsense physical reasoning",
    ),
    DatasetConfig(
        name="global_piqa",
        adapter_class=GlobalPIQAAdapter,
        adapter_kwargs={"subset": "eng_Latn", "split": "validation"},
        dataset_name="mrlbenchmarks/global-piqa-nonparallel",
        config_name="eng_Latn",
        split="validation",
        min_samples=10,
        description="Global PIQA - multilingual PIQA (English subset only)",
    ),
    DatasetConfig(
        name="hellaswag",
        adapter_class=HellaSwagAdapter,
        adapter_kwargs={"split": "validation"},
        dataset_name="Rowan/hellaswag",
        split="validation",
        min_samples=10,
        description="HellaSwag - commonsense natural language inference",
    ),
    DatasetConfig(
        name="winogrande",
        adapter_class=WinograndeAdapter,
        adapter_kwargs={"split": "validation"},
        dataset_name="winogrande",
        config_name="winogrande_xl",
        split="validation",
        min_samples=10,
        description="Winogrande - pronoun resolution reasoning",
    ),
    DatasetConfig(
        name="mmlu",
        adapter_class=MMLUAdapter,
        adapter_kwargs={"split": "test", "config_name": "abstract_algebra"},
        dataset_name="cais/mmlu",
        config_name="abstract_algebra",
        split="test",
        min_samples=5,
        description="MMLU - massive multitask language understanding",
    ),
    DatasetConfig(
        name="arc_challenge",
        adapter_class=ARC_Adapter,
        adapter_kwargs={"challenge_type": "ARC-Challenge", "split": "validation"},
        dataset_name="ai2_arc",
        config_name="ARC-Challenge",
        split="validation",
        min_samples=10,
        description="ARC Challenge - science question answering",
    ),
    DatasetConfig(
        name="arc_easy",
        adapter_class=ARC_Adapter,
        adapter_kwargs={"challenge_type": "ARC-Easy", "split": "validation"},
        dataset_name="ai2_arc",
        config_name="ARC-Easy",
        split="validation",
        min_samples=10,
        description="ARC Easy - science question answering",
    ),
    DatasetConfig(
        name="commonsense_qa",
        adapter_class=CommonsenseQAAdapter,
        adapter_kwargs={"split": "validation"},
        dataset_name="commonsense_qa",
        split="validation",
        min_samples=10,
        description="CommonsenseQA - commonsense reasoning",
    ),
    DatasetConfig(
        name="truthful_qa_mc1",
        adapter_class=TruthfulQAAdapter,
        adapter_kwargs={"split": "validation", "mc_type": "mc1"},
        dataset_name="truthful_qa",
        config_name="multiple_choice",
        split="validation",
        min_samples=5,
        description="TruthfulQA MC1 - truthfulness evaluation (single correct)",
    ),
    DatasetConfig(
        name="truthful_qa_mc2",
        adapter_class=TruthfulQAAdapter,
        adapter_kwargs={"split": "validation", "mc_type": "mc2"},
        dataset_name="truthful_qa",
        config_name="multiple_choice",
        split="validation",
        min_samples=5,
        description="TruthfulQA MC2 - truthfulness evaluation (multiple correct)",
    ),
    DatasetConfig(
        name="bigbench_emoji_movie",
        adapter_class=BigBenchAdapter,
        adapter_kwargs={"task_name": "emoji_movie", "split": "default"},
        dataset_name="google/bigbench",
        config_name="emoji_movie",
        split="default",
        min_samples=5,
        description="BigBench - emoji movie task (example)",
    ),
]


class DatasetValidator:
    """Validate dataset availability and format."""

    def __init__(self, quick: bool = False):
        """
        Initialize validator.

        Args:
            quick: If True, load only small samples for quick validation
        """
        self.quick = quick
        self.results: List[Dict[str, Any]] = []

    def validate_dataset(self, config: DatasetConfig) -> Dict[str, Any]:
        """
        Validate a single dataset.

        Returns:
            Dict with validation results (status, message, details)
        """
        result = {
            "name": config.name,
            "dataset": config.dataset_name,
            "status": "unknown",
            "message": "",
            "details": {},
        }

        try:
            # Step 1: Load using adapter (handles HuggingFace internally)
            print(f"\n{'=' * 60}")
            print(f"Validating: {config.name}")
            print(f"Dataset: {config.dataset_name}")
            if config.config_name:
                print(f"Config: {config.config_name}")
            print(f"Split: {config.split}")
            print(f"Description: {config.description}")
            print(f"{'=' * 60}")

            print(f"Loading dataset via adapter...")

            # Create adapter with quick mode if requested
            if self.quick:
                # Modify adapter kwargs to load only first 10 samples
                adapter_kwargs = config.adapter_kwargs.copy()
                # Override split to load only 10 samples
                if "split" in adapter_kwargs:
                    original_split = adapter_kwargs["split"]
                    adapter_kwargs["split"] = f"{original_split}[:10]"
                    print(f"Quick mode: loading {adapter_kwargs['split']}")
            else:
                adapter_kwargs = config.adapter_kwargs

            adapter = config.adapter_class(**adapter_kwargs)

            # Load questions via adapter
            questions = adapter.load()
            num_converted = len(questions)

            print(f"✓ Loaded and converted {num_converted} examples to BencheckQuestion")
            result["details"]["num_converted"] = num_converted

            # Check conversion quality
            if num_converted == 0:
                result["status"] = "error"
                result["message"] = "No examples could be converted (format mismatch?)"
                print(f"✗ {result['message']}")
                return result

            # Check minimum samples
            if num_converted < config.min_samples and not self.quick:
                result["status"] = "warning"
                result["message"] = (
                    f"Only {num_converted} samples loaded (expected at least {config.min_samples})"
                )
                print(f"⚠ {result['message']}")
            else:
                print(f"✓ Sample count OK ({num_converted} examples)")

            # Step 3: Validate question structure
            print("\nValidating question structure...")
            sample_q = questions[0]

            assert sample_q.question, "Question text is empty"
            assert sample_q.choices, "Choices list is empty"
            assert len(sample_q.choices) >= 2, "Need at least 2 choices"
            assert sample_q.correct is not None, "Correct answer is None"
            assert sample_q.question_type is not None, "Question type is None"

            print(f"✓ Question structure valid")
            print(f"  - Question: {sample_q.question[:60]}...")
            print(f"  - Num choices: {len(sample_q.choices)}")
            print(f"  - Correct: {sample_q.correct}")
            print(f"  - Type: {sample_q.question_type.value}")

            result["details"]["sample_question"] = {
                "id": sample_q.id,
                "question_preview": sample_q.question[:100],
                "num_choices": len(sample_q.choices),
                "correct": sample_q.correct,
                "question_type": sample_q.question_type.value,
            }

            # Success
            if result["status"] != "warning":
                result["status"] = "success"
                result["message"] = f"Dataset validated successfully ({num_converted} examples)"

            print(f"\n✓ Validation complete: {result['status'].upper()}")
            return result

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__

            # Handle specific known issues
            if "LocalFileSystem is not supported" in error_msg:
                result["status"] = "warning"
                result["message"] = (
                    f"Dataset cached locally in unsupported format. "
                    f"Try: `rm -rf ~/.cache/huggingface/datasets/{config.dataset_name.replace('/', '___')}`"
                )
                print(f"\n⚠ {result['message']}")
                print(f"   Original error: {error_msg}")
            else:
                result["status"] = "error"
                result["message"] = f"Validation failed: {error_msg}"
                print(f"\n✗ Validation failed: {e}")
                import traceback

                traceback.print_exc()

            result["details"]["error_type"] = error_type
            return result

    def validate_all(self, dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate all configured datasets.

        Args:
            dataset_names: Optional list of specific datasets to validate.
                          If None, validates all datasets.

        Returns:
            Summary dict with overall results
        """
        # Filter configs if specific datasets requested
        configs = DATASET_CONFIGS
        if dataset_names:
            configs = [c for c in configs if c.name in dataset_names]
            if not configs:
                print(f"Error: No datasets found matching: {dataset_names}")
                print(f"Available datasets: {[c.name for c in DATASET_CONFIGS]}")
                return {"status": "error", "message": "No datasets to validate"}

        print(f"\nValidating {len(configs)} dataset(s)...\n")

        # Validate each dataset
        for config in configs:
            result = self.validate_dataset(config)
            self.results.append(result)

        # Generate summary
        summary = self._generate_summary()
        return summary

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total = len(self.results)
        success = sum(1 for r in self.results if r["status"] == "success")
        warnings = sum(1 for r in self.results if r["status"] == "warning")
        errors = sum(1 for r in self.results if r["status"] == "error")

        summary = {
            "total": total,
            "success": success,
            "warnings": warnings,
            "errors": errors,
            "results": self.results,
        }

        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total datasets: {total}")
        print(f"✓ Success: {success}")
        if warnings > 0:
            print(f"⚠ Warnings: {warnings}")
        if errors > 0:
            print(f"✗ Errors: {errors}")

        # Print details for warnings and errors
        if warnings > 0 or errors > 0:
            print("\nIssues found:")
            for r in self.results:
                if r["status"] in ["warning", "error"]:
                    symbol = "⚠" if r["status"] == "warning" else "✗"
                    print(f"  {symbol} {r['name']}: {r['message']}")

        print("=" * 60 + "\n")

        return summary


def main():
    """Run dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate BenCheck benchmark dataset availability and format"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to validate (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation (load only 10 samples per dataset)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    # List datasets if requested
    if args.list:
        print("\nAvailable datasets:")
        print("=" * 60)
        for config in DATASET_CONFIGS:
            print(f"\n{config.name}")
            print(f"  Dataset: {config.dataset_name}")
            if config.config_name:
                print(f"  Config: {config.config_name}")
            print(f"  Split: {config.split}")
            print(f"  Description: {config.description}")
        print("\n" + "=" * 60)
        return 0

    # Run validation
    validator = DatasetValidator(quick=args.quick)
    summary = validator.validate_all(dataset_names=args.datasets)

    # Exit with appropriate code
    if summary["errors"] > 0:
        return 1
    elif summary["warnings"] > 0:
        return 0  # Warnings don't fail the script
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
