#!/usr/bin/env python3
"""Validate model availability and functionality.

This script checks that configured models:
1. Can be initialized successfully
2. Support required interfaces (GenerativeModel, ScoringModel)
3. Can perform basic operations (generation, scoring)
4. Report proper error messages when unavailable

Usage:
    python scripts/validate_models.py
    python scripts/validate_models.py --models dummy
    python scripts/validate_models.py --skip-slow  # Skip time-consuming model tests
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bencheck.models import DummyModel


@dataclass
class ModelConfig:
    """Configuration for a model to validate."""

    name: str
    model_class: Any
    init_kwargs: Dict[str, Any]
    supports_generation: bool
    supports_scoring: bool
    requires_api_key: bool = False
    requires_gpu: bool = False
    is_slow: bool = False
    description: str = ""


# Model configurations
MODEL_CONFIGS = [
    ModelConfig(
        name="dummy",
        model_class=DummyModel,
        init_kwargs={},
        supports_generation=True,
        supports_scoring=True,
        requires_api_key=False,
        requires_gpu=False,
        is_slow=False,
        description="Dummy model for testing - always available",
    ),
    # Note: Real model configs would go here, but they require
    # API keys or GPU setup. The dummy model is always testable.
]


class ModelValidator:
    """Validate model availability and functionality."""

    def __init__(self, skip_slow: bool = False):
        """
        Initialize validator.

        Args:
            skip_slow: If True, skip slow model tests (GPU models, API calls)
        """
        self.skip_slow = skip_slow
        self.results: List[Dict[str, Any]] = []

    def validate_model(self, config: ModelConfig) -> Dict[str, Any]:
        """
        Validate a single model.

        Returns:
            Dict with validation results (status, message, details)
        """
        result = {
            "name": config.name,
            "status": "unknown",
            "message": "",
            "details": {},
        }

        try:
            print(f"\n{'=' * 60}")
            print(f"Validating: {config.name}")
            print(f"Class: {config.model_class.__name__}")
            print(f"Description: {config.description}")
            print(f"{'=' * 60}")

            # Check if we should skip
            if self.skip_slow and config.is_slow:
                result["status"] = "skipped"
                result["message"] = "Skipped (slow test)"
                print(f"⊘ Skipped: Model requires slow operations")
                return result

            # Step 1: Test initialization
            print("\nTesting model initialization...")
            try:
                model = config.model_class(**config.init_kwargs)
                print(f"✓ Model initialized successfully")
                result["details"]["initialization"] = "success"
            except Exception as e:
                error_msg = str(e)
                if config.requires_api_key and ("API" in error_msg or "key" in error_msg.lower()):
                    result["status"] = "warning"
                    result["message"] = f"Requires API key: {error_msg}"
                    print(f"⚠ {result['message']}")
                    return result
                elif config.requires_gpu and ("CUDA" in error_msg or "GPU" in error_msg):
                    result["status"] = "warning"
                    result["message"] = f"Requires GPU: {error_msg}"
                    print(f"⚠ {result['message']}")
                    return result
                else:
                    raise  # Re-raise if not an expected requirement issue

            # Step 2: Test generation (if supported)
            if config.supports_generation:
                print("\nTesting generation capability...")
                try:
                    test_prompt = "The capital of France is"
                    response = model.generate(test_prompt, max_tokens=10)
                    print(f"✓ Generation works")
                    print(f"  Prompt: {test_prompt}")
                    print(f"  Response: {response[:100]}...")
                    result["details"]["generation"] = "success"
                    result["details"]["generation_sample"] = response[:100]
                except Exception as e:
                    result["status"] = "warning"
                    result["message"] = f"Generation failed: {str(e)}"
                    print(f"⚠ {result['message']}")
                    result["details"]["generation"] = "failed"

            # Step 3: Test scoring (if supported)
            if config.supports_scoring:
                print("\nTesting scoring capability...")
                try:
                    test_context = "The capital of France is"
                    test_continuation = " Paris"
                    score = model.score_continuation(test_context, test_continuation)
                    print(f"✓ Scoring works")
                    print(f"  Context: {test_context}")
                    print(f"  Continuation: {test_continuation}")
                    print(f"  Score: {score}")
                    result["details"]["scoring"] = "success"
                    result["details"]["scoring_sample"] = (
                        float(score) if isinstance(score, (int, float)) else "complex"
                    )
                except Exception as e:
                    result["status"] = "warning"
                    result["message"] = f"Scoring failed: {str(e)}"
                    print(f"⚠ {result['message']}")
                    result["details"]["scoring"] = "failed"

            # Step 4: Test batch scoring (if supported)
            if config.supports_scoring:
                print("\nTesting batch scoring capability...")
                try:
                    test_context = "The capital of France is"
                    test_continuations = [" Paris", " London", " Berlin"]
                    scores = model.score_continuations(test_context, test_continuations)
                    print(f"✓ Batch scoring works")
                    print(f"  Context: {test_context}")
                    print(f"  Continuations: {test_continuations}")
                    print(f"  Scores: {scores}")
                    result["details"]["batch_scoring"] = "success"
                except Exception as e:
                    # Batch scoring is optional
                    print(f"⊘ Batch scoring not supported: {str(e)}")
                    result["details"]["batch_scoring"] = "not_supported"

            # Success
            if result["status"] != "warning":
                result["status"] = "success"
                result["message"] = "Model validated successfully"

            print(f"\n✓ Validation complete: {result['status'].upper()}")
            return result

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Validation failed: {str(e)}"
            result["details"]["error_type"] = type(e).__name__
            print(f"\n✗ Validation failed: {e}")
            import traceback

            traceback.print_exc()
            return result

    def validate_all(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate all configured models.

        Args:
            model_names: Optional list of specific models to validate.
                        If None, validates all models.

        Returns:
            Summary dict with overall results
        """
        # Filter configs if specific models requested
        configs = MODEL_CONFIGS
        if model_names:
            configs = [c for c in configs if c.name in model_names]
            if not configs:
                print(f"Error: No models found matching: {model_names}")
                print(f"Available models: {[c.name for c in MODEL_CONFIGS]}")
                return {"status": "error", "message": "No models to validate"}

        print(f"\nValidating {len(configs)} model(s)...\n")

        # Validate each model
        for config in configs:
            result = self.validate_model(config)
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
        skipped = sum(1 for r in self.results if r["status"] == "skipped")

        summary = {
            "total": total,
            "success": success,
            "warnings": warnings,
            "errors": errors,
            "skipped": skipped,
            "results": self.results,
        }

        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total models: {total}")
        print(f"✓ Success: {success}")
        if warnings > 0:
            print(f"⚠ Warnings: {warnings}")
        if errors > 0:
            print(f"✗ Errors: {errors}")
        if skipped > 0:
            print(f"⊘ Skipped: {skipped}")

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
    """Run model validation."""
    parser = argparse.ArgumentParser(
        description="Validate BenCheck model availability and functionality"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to validate (default: all)",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow model tests (GPU models, API calls)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # List models if requested
    if args.list:
        print("\nAvailable models:")
        print("=" * 60)
        for config in MODEL_CONFIGS:
            print(f"\n{config.name}")
            print(f"  Class: {config.model_class.__name__}")
            print(f"  Generation: {'✓' if config.supports_generation else '✗'}")
            print(f"  Scoring: {'✓' if config.supports_scoring else '✗'}")
            if config.requires_api_key:
                print(f"  Requires: API key")
            if config.requires_gpu:
                print(f"  Requires: GPU")
            if config.is_slow:
                print(f"  Note: Slow test (use --skip-slow to skip)")
            print(f"  Description: {config.description}")
        print("\n" + "=" * 60)
        return 0

    # Run validation
    validator = ModelValidator(skip_slow=args.skip_slow)
    summary = validator.validate_all(model_names=args.models)

    # Exit with appropriate code
    if summary["errors"] > 0:
        return 1
    elif summary["warnings"] > 0:
        return 0  # Warnings don't fail the script
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
