#!/usr/bin/env python3
"""
Regenerate experiment_summary.json and REPORT.md for existing results.

Usage:
    python scripts/regenerate_summary_and_report.py results/reasoning_datasets
    python scripts/regenerate_summary_and_report.py results/full_benchmarks
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from output_utils import create_experiment_summary


def main():
    parser = argparse.ArgumentParser(description="Regenerate summary and report")
    parser.add_argument("results_dir", type=Path, help="Results directory")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to original config (optional, will use 'unknown' if not provided)"
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return 1

    # Determine config path
    config_path = args.config or "unknown"

    # Regenerate experiment_summary.json
    print(f"Regenerating experiment_summary.json for {results_dir}...")
    create_experiment_summary(results_dir, config_path)
    print(f"✅ Created {results_dir}/experiment_summary.json")

    # Regenerate REPORT.md
    print(f"Regenerating REPORT.md...")
    report_path = results_dir / "REPORT.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_report.py",
            str(results_dir),
            "-o",
            str(report_path)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✅ Created {report_path}")
    else:
        print(f"⚠️  Report generation failed: {result.stderr}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
