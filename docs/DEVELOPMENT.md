# Development Guide

Complete guide for getting started with BenCheck and contributing to development.

---

## Quick Start

Get up and running with BenCheck in 10 minutes.

### Prerequisites

- Python 3.10+
- Nothing else for the model-free quickstart; a GPU with CUDA (local vLLM) **or** API keys (Gemini/OpenAI) for model-dependent checks
- 16GB+ RAM recommended

### Installation

```bash
cd BenCheck

# Core package only
pip install -e .

# With vLLM (local GPU inference)
pip install -e .[vllm]

# With API support (Gemini/OpenAI)
pip install -e .[gemini]
pip install -e .[openai]

# Everything (development + all models)
pip install -e .[all]
```

### Verify Installation

```bash
python -c "import bencheck; print(bencheck.__version__)"
# Should print: 0.2.0

# CPU-only smoke test (~1 minute, no GPU or API key)
python scripts/run_experiment.py configs/examples/quickstart.yaml
# Results in: results/quickstart/
```

---

## Running Your First Experiment

### Option A: Local vLLM (GPU)

**Step 1: Start vLLM Server**

```bash
# Serve the model named in the example configs. If you serve a different model,
# set `models[].name` in your config to the same id.
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-12b-it \
  --host 0.0.0.0 \
  --port 8008
```

Wait for: `"Uvicorn running on http://0.0.0.0:8008"`

**Step 2: Test Connection**

```bash
curl http://localhost:8008/v1/models
# Should return: {"object":"list","data":[{"id":"google/gemma-3-12b-it",...}]}
```

**Step 3: Run an Example**

```bash
# Prompt-importance check (full / empty / Lorem-ipsum prompt) on
# PIQA, Global PIQA, HellaSwag, Winogrande. Long run: 8-16 hours.
nohup python scripts/run_experiment.py configs/examples/context_requirement_reasoning_datasets.yaml \
  > context_requirement.log 2>&1 &

# Monitor progress
tail -f context_requirement.log
```

All example configs live in `configs/examples/` (see the table in the README).

### Option B: API (Gemini/OpenAI)

**Step 1: Set API Key**

```bash
# For Gemini
export GEMINI_API_KEY="your-api-key-here"

# For OpenAI
export OPENAI_API_KEY="your-api-key-here"
```

**Step 2: Create Custom Config**

```yaml
# configs/my_experiment.yaml
experiment:
  n_runs: 3
  scoring_modes: ["log_likelihood"]  # APIs typically only support this
  seed: 42

  benchmarks:
    - name: "piqa_validation"
      adapter: "PIQAAdapter"
      params:
        split: "validation"

  models:
    - type: "gemini"  # or "openai"
      name: "gemini-1.5-flash"  # or "gpt-4o-mini"
      params:
        # Gemini needs no extra params
        # OpenAI: api_key loaded from env

  checks:
    - name: "LengthBiasCheck"
      model: null
    - name: "EnumerationBiasCheck"
    - name: "NoneOfTheAboveCheck"
    - name: "ContextRequirementCheck"
      params:
        transform_type: "both"

output:
  dir: "results/my_experiment"
  format: "json"
```

**Step 3: Run**

```bash
python scripts/run_experiment.py configs/my_experiment.yaml
```

---

## Understanding Results

### Output Structure

```
results/reasoning_datasets/
├── experiment_summary.json        ← High-level overview
├── piqa_validation_/
│   ├── questions_base.json        ← Question data (no check results)
│   ├── length_bias.json           ← Model-free check
│   ├── enumeration_bias.json      ← Scoring-independent check
│   ├── none_of_the_above.json     ← Scoring-independent check
│   ├── context_requirement.json   ← Scoring-dependent check (all runs)
│   └── per_questions/
│       ├── length_bias.json       ← Per-question details
│       ├── enumeration_bias.json
│       ├── none_of_the_above.json
│       ├── context_requirement_log_likelihood_run_0.json
│       ├── context_requirement_log_likelihood_run_1.json
│       └── ...
├── hellaswag_validation_/
│   └── ...
└── ...
```

### Key Files

**`experiment_summary.json`** - Overall summary
```json
{
  "experiment_config": "configs/examples/context_requirement_reasoning_datasets.yaml",
  "timestamp": "2025-12-27T14:30:00",
  "datasets": {
    "piqa_validation": {
      "n_questions": 1838,
      "checks": {
        "length_bias": {...},
        "enumeration_bias": {...},
        ...
      }
    }
  }
}
```

**`{check_name}.json`** - Summary metrics
```json
{
  "_metadata": {
    "model": "google/gemma-3-12b-it",
    "mode": null,
    "check_type": "scoring_independent"
  },
  "n_questions": 1838,
  "overall_accuracy": 75.2,
  ...
}
```

**`per_questions/{check_name}.json`** - Detailed per-question results

### Quick Red Flags

| Check | Metric | Red Flag | Interpretation |
|-------|--------|----------|----------------|
| **Length Bias** | `median_relative_length_diff` | > 0.3 | High length variability |
| **Enumeration Bias** | `gold_chi2_p_value` | < 0.05 | Non-uniform answer distribution |
| **Enumeration Bias** | `pred_chi2_p_value` | < 0.05 | Model has positional bias |
| **None of the Above** | `placeholder_selected_rate` | > 30% | Poor distractor quality |
| **Context Requirement** | `empty_accuracy` | > 30% | Questions don't need context |
| **Grammar Quality** | `Q_any_issue.rate` | > 0.1 | Grammar issues in 10%+ questions |

See [CHECKS.md](CHECKS.md) for detailed explanations of each check.

---

## Common Issues

### vLLM Server Not Starting

```bash
# Check GPU availability
nvidia-smi

# Check CUDA version
nvcc --version

# Try with smaller model
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8008
```

### Out of Memory (OOM)

```bash
# Reduce batch size
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8
```

### Import Errors

```bash
# Reinstall with specific dependencies
pip install -e .[vllm]

# Or reinstall everything
pip uninstall bencheck
pip install -e .[all]
```

### Progress Too Slow

Check your config:
- Reduce `n_runs` from 3 to 1 for testing
- Use `scoring_modes: ["log_likelihood"]` only (faster than generation)
- Test on a slice first, e.g. `split: "validation[:100]"` (see `configs/examples/quickstart.yaml`)

---

## Development Setup

For contributors and developers working on BenCheck.

### Install Development Dependencies

```bash
pip install -e .[dev]
```

This installs:
- `pytest` - Test framework
- `pytest-cov` - Coverage plugin
- `ruff` - Fast Python linter and formatter
- `mypy` - Static type checker
- `codespell` - Spell checker

---

## Testing

### Basic Test Run

```bash
pytest
```

### Verbose Output

```bash
pytest -v
```

### Run Specific Test File

```bash
pytest tests/test_length_bias.py
```

### Run Specific Test

```bash
pytest tests/test_length_bias.py::test_length_bias_check
```

### Skip Slow Tests

```bash
pytest -m "not slow"
```

### Test Coverage

Generate coverage report:
```bash
pytest --cov=bencheck --cov-report=term
```

Generate HTML coverage report:
```bash
pytest --cov=bencheck --cov-report=html
```

Open `htmlcov/index.html` in your browser to view the interactive report.

Generate coverage with missing lines:
```bash
pytest --cov=bencheck --cov-report=term-missing
```

Generate XML coverage (for CI):
```bash
pytest --cov=bencheck --cov-report=xml
```

---

## Code Quality

### Run Linter (Ruff)

Check for issues:
```bash
ruff check bencheck/ tests/ scripts/
```

Auto-fix issues:
```bash
ruff check --fix bencheck/ tests/ scripts/
```

### Run Formatter (Ruff)

Check formatting:
```bash
ruff format --check bencheck/ tests/ scripts/
```

Auto-format:
```bash
ruff format bencheck/ tests/ scripts/
```

### Run Type Checker (MyPy)

```bash
mypy bencheck/ --ignore-missing-imports
```

---

## Writing Tests

### Test Structure

```python
import pytest
from bencheck.checks import LengthBiasCheck

def test_length_bias_basic():
    """Test basic length bias functionality."""
    check = LengthBiasCheck()
    # Test implementation
    assert check.name == "length_bias"

@pytest.mark.slow
def test_length_bias_full_dataset():
    """Test with full dataset (slow)."""
    # This test will be skipped with -m "not slow"
    pass

@pytest.mark.integration
def test_end_to_end_pipeline():
    """Integration test for full pipeline."""
    pass
```

### Running Specific Markers

```bash
# Only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Slow tests only
pytest -m slow
```

---

## Coverage Goals

**Target**: 80%+ coverage for core functionality

**Priority areas**:
- `bencheck/checks/` - Diagnostic checks (target: 90%+)
- `bencheck/adapters/` - Dataset adapters (target: 85%+)
- `bencheck/core/` - Core runner and registry (target: 90%+)
- `bencheck/models/` - Model wrappers (target: 75%+)
- `bencheck/utils/` - Utilities (target: 80%+)

**Excluded from coverage**:
- Test files (`tests/`)
- Example scripts (`examples/`)
- Documentation (`docs/`)

---

## Best Practices

1. **Write tests for all new features** - Add tests before merging
2. **Maintain coverage** - Don't decrease coverage percentage
3. **Use meaningful test names** - `test_length_bias_calculates_median_correctly`
4. **Test edge cases** - Empty inputs, invalid data, etc.
5. **Use fixtures** - Share setup code between tests
6. **Mark slow tests** - Use `@pytest.mark.slow` for tests > 1 second
7. **Document complex tests** - Add docstrings explaining what's being tested

---

## Troubleshooting

### Tests Failing Locally

- Check Python version (3.10+ required)
- Ensure all dependencies are installed: `pip install -e .[dev]`
- Clean pytest cache: `pytest --cache-clear`

### Coverage Not Updating

- Delete `.coverage` file: `rm .coverage`
- Delete `htmlcov/` directory: `rm -rf htmlcov/`
- Re-run with coverage: `pytest --cov=bencheck`

### Ruff Errors

- Auto-fix: `ruff check --fix bencheck/`
- Check specific file: `ruff check bencheck/checks/length_bias.py`
- Ignore specific error: Add `# noqa: E501` to the line

### MyPy Errors

- Check specific file: `mypy bencheck/checks/length_bias.py`
- Ignore missing imports: `mypy bencheck/ --ignore-missing-imports`
- Add type ignores: `# type: ignore` on the line

---

## Quick Reference

```bash
# Run all tests with coverage
pytest --cov=bencheck --cov-report=html

# Run fast tests only
pytest -m "not slow"

# Lint and format
ruff check --fix bencheck/
ruff format bencheck/

# Type check
mypy bencheck/ --ignore-missing-imports

# Full check before opening a PR
pytest --cov=bencheck --cov-report=xml && \
ruff check bencheck/ tests/ scripts/ && \
ruff format --check bencheck/ tests/ scripts/ && \
mypy bencheck/ --ignore-missing-imports --no-strict-optional
```

---

## Next Steps

1. **Read Check Documentation**: [CHECKS.md](CHECKS.md)
2. **Understand Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Run the examples** in `configs/examples/`

---

## Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: See `configs/examples/` for example configurations
