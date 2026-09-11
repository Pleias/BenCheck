#!/usr/bin/env python3
"""
Verify BenCheck environment setup.

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --vllm
    python scripts/verify_setup.py --gemini
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print("=" * 60)


def check_item(name, success, details=""):
    """Print check result."""
    status = "✅" if success else "❌"
    print(f"{status} {name}")
    if details:
        print(f"   → {details}")
    return success


def verify_python():
    """Verify Python version."""
    print_header("Python Environment")

    version = sys.version_info
    success = version.major == 3 and version.minor >= 11

    check_item(
        "Python version",
        success,
        f"Python {version.major}.{version.minor}.{version.micro}"
        + ("" if success else " (requires 3.11+)"),
    )

    return success


def verify_core():
    """Verify core BenCheck installation."""
    print_header("Core Installation")

    all_success = True

    # Check bencheck import
    try:
        import bencheck

        version = getattr(bencheck, "__version__", "installed")
        check_item(
            "BenCheck package", True, f"v{version}" if version != "installed" else "Installed"
        )
    except ImportError as e:
        check_item("BenCheck package", False, f"Error: {e}")
        all_success = False
        return False

    # Check core modules
    modules = [
        ("bencheck.adapters", "Adapters"),
        ("bencheck.checks", "Checks"),
        ("bencheck.models", "Models"),
        ("bencheck.core.runner", "Runner"),
        ("bencheck.utils.config", "Config"),
    ]

    for module_name, display_name in modules:
        try:
            __import__(module_name)
            check_item(display_name, True)
        except ImportError as e:
            check_item(display_name, False, str(e))
            all_success = False

    return all_success


def verify_vllm():
    """Verify vLLM installation and GPU."""
    print_header("vLLM Setup")

    all_success = True

    # Check vLLM import
    try:
        import vllm

        check_item("vLLM package", True, f"v{vllm.__version__}")
    except ImportError:
        check_item("vLLM package", False, "Install with: pip install vllm")
        return False

    # Check PyTorch
    try:
        import torch

        check_item("PyTorch", True, f"v{torch.__version__}")
    except ImportError:
        check_item("PyTorch", False, "Should be installed with vLLM")
        all_success = False

    # Check CUDA
    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            check_item("CUDA GPU", True, f"{gpu_count} GPU(s) - {gpu_name}")

            # Check VRAM
            if gpu_count > 0:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                check_item("GPU Memory", True, f"{vram_gb:.1f} GB")
        else:
            check_item("CUDA GPU", False, "No GPU detected")
            all_success = False
    except Exception as e:
        check_item("CUDA GPU", False, str(e))
        all_success = False

    # Test vLLM model creation (dry run)
    try:
        from bencheck.models import VLLMModel

        check_item("VLLMModel class", True, "Can be imported")
    except ImportError as e:
        check_item("VLLMModel class", False, str(e))
        all_success = False

    return all_success


def verify_gemini():
    """Verify Gemini SDK and API key."""
    print_header("Gemini Setup")

    all_success = True

    # Check Gemini SDK
    try:
        from google import genai

        check_item("Gemini SDK", True, "google-genai installed")
    except ImportError:
        check_item("Gemini SDK", False, "Install with: pip install google-genai")
        return False

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if api_key:
        check_item("API Key", True, f"Set ({api_key[:10]}...)")

        # Test API connection
        try:
            from bencheck.models import GeminiModel

            print("   Testing API connection...")
            model = GeminiModel("gemini-2.0-flash-exp")
            response = model.generate("Hi", max_tokens=5)
            check_item("API Connection", True, "Successfully generated text")
        except Exception as e:
            check_item("API Connection", False, str(e))
            all_success = False
    else:
        check_item("API Key", False, "Set with: export GEMINI_API_KEY='your-key'")
        all_success = False

    return all_success


def verify_configs():
    """Verify configuration system."""
    print_header("Configuration System")

    all_success = True

    # Check YAML support
    try:
        import yaml

        check_item("PyYAML", True, "YAML configs supported")
    except ImportError:
        check_item("PyYAML", False, "Install with: pip install pyyaml (optional)")
        all_success = False

    # Check config files exist
    config_dir = Path(__file__).parent.parent / "configs" / "examples"

    if config_dir.exists():
        configs = list(config_dir.glob("*.json")) + list(config_dir.glob("*.yaml"))
        check_item("Example configs", True, f"{len(configs)} configs found")
    else:
        check_item("Example configs", False, "configs/examples/ not found")
        all_success = False

    # Check runner script
    runner_script = Path(__file__).parent / "run_from_config.py"
    check_item("Runner script", runner_script.exists(), str(runner_script))

    return all_success


def verify_tests():
    """Verify test suite."""
    print_header("Development Tools")

    all_success = True

    # Check pytest
    try:
        import pytest

        check_item("pytest", True, f"v{pytest.__version__}")
    except ImportError:
        check_item("pytest", False, "Install with: pip install -e .[dev]")
        all_success = False

    # Check ruff
    try:
        import ruff

        check_item("ruff", True, "Linter available")
    except ImportError:
        check_item("ruff", False, "Install with: pip install -e .[dev]")
        all_success = False

    # Check test directory
    test_dir = Path(__file__).parent.parent / "tests"
    if test_dir.exists():
        test_files = list(test_dir.glob("test_*.py"))
        check_item("Test suite", True, f"{len(test_files)} test files")
    else:
        check_item("Test suite", False, "tests/ directory not found")
        all_success = False

    return all_success


def main():
    """Main verification routine."""
    parser = argparse.ArgumentParser(description="Verify BenCheck environment setup")
    parser.add_argument("--vllm", action="store_true", help="Verify vLLM setup (GPU required)")
    parser.add_argument("--gemini", action="store_true", help="Verify Gemini API setup")
    parser.add_argument("--all", action="store_true", help="Verify everything (vLLM + Gemini)")

    args = parser.parse_args()

    # If no specific checks requested, do basic verification
    if not (args.vllm or args.gemini or args.all):
        args.vllm = False
        args.gemini = False

    if args.all:
        args.vllm = True
        args.gemini = True

    print("=" * 60)
    print("  BenCheck Environment Verification")
    print("=" * 60)

    results = {}

    # Always check Python and core
    results["python"] = verify_python()
    results["core"] = verify_core()

    # Optional checks
    if args.vllm:
        results["vllm"] = verify_vllm()

    if args.gemini:
        results["gemini"] = verify_gemini()

    # Always check configs
    results["configs"] = verify_configs()
    results["tests"] = verify_tests()

    # Summary
    print_header("Summary")

    all_passed = all(results.values())

    for name, passed in results.items():
        check_item(name.upper(), passed)

    if all_passed:
        print("\n✅ All checks passed! BenCheck is ready to use.")
        print("\nNext steps:")
        print("  1. Try: python scripts/run_from_config.py configs/examples/model_free.json")
        if args.vllm:
            print("  2. Try: python scripts/run_from_config.py configs/examples/vllm_full.yaml")
        if args.gemini:
            print("  2. Try: python scripts/run_from_config.py configs/examples/gemini_simple.yaml")
        print("  3. Read: docs/ARCHITECTURE.md for detailed documentation")
        return 0
    else:
        print("\n❌ Some checks failed. See details above.")
        print("\nFor help:")
        print("  - Read: docs/ARCHITECTURE.md")
        print("  - Read: docs/DEVELOPMENT.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
