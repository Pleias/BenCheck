# BenCheck Architecture

**Version: v0.2.0**

Technical architecture documentation for BenCheck (Benchmark quality auditing toolkit) - a benchmark quality auditing toolkit.

> ** For quick start and usage, see [DEVELOPMENT.md](DEVELOPMENT.md)**
> ** For diagnostic checks details, see [CHECKS.md](CHECKS.md)**

---

## Table of Contents

1. [Check Compatibility Matrix](#check-compatibility-matrix)
2. [Architecture](#architecture)
3. [MCQ vs Single-Choice Handling](#mcq-vs-single-choice-handling)
4. [Extension Guide](#extension-guide)
5. [Design Decisions](#design-decisions)

---

## Check Compatibility Matrix

| Check | Model Required | vLLM | Gemini | DummyModel |
|-------|----------------|------|--------|------------|
| LengthBiasCheck | No (model-free) |  |  |  |
| EnumerationBiasCheck | Yes (scoring) |  (log-likelihood) |  |  (lexical) |
| NoneOfTheAboveCheck | Yes (scoring) |  (log-likelihood) |  |  (lexical) |
| ContextRequirementCheck | Yes (scoring or generation) |  (both modes) |  (generation only) |  (lexical) |
| GrammarQualityCheck | Yes (GECToR grammar model) |  |  |  |

**Key:**
-  Full support
-  Not supported (Gemini lacks log-likelihood API)
- vLLM log-likelihood mode is recommended for most accurate MCQ evaluation


---

## Architecture

### Design Principles

1. **Dataset-centric**: Evaluates benchmark quality, not model performance
2. **Question-centric output**: Results organized by question, showing all diagnostics
3. **Per-check models**: Each check manages its own model (or is model-free)
4. **Protocol-based**: Duck typing for flexible model integration
5. **Extensible checks**: Easy to add new diagnostic tests
6. **Type-safe**: Heavy use of dataclasses and type hints
7. **Zero core dependencies**: Pure Python for portability

### Component Interaction

```
┌──────────────────────────────────────────────────────────────────────┐
│                          BencheckRunner                                │
│                     (Orchestrates evaluation)                         │
│                   NO GLOBAL MODEL (v0.2.0)                            │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             │ 1. Load dataset
             ▼
┌─────────────────────────┐
│    DatasetAdapter       │──────► List[BencheckQuestion]
│  (Load benchmark data)  │
└─────────────────────────┘
             │
             │ 2. For each Check:
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      BencheckCheck                                    │
│  Each check has its own model (or is model-free)                     │
│  ┌────────────────────┐          ┌──────────────────────┐            │
│  │  Model-Free Check  │          │  Model-Dependent     │            │
│  │  (e.g. Length Bias)│          │  Check               │            │
│  │  No model needed   │          │  ┌────────────────┐  │            │
│  └────────────────────┘          │  │ ScoringModel   │  │            │
│                                  │  │ or             │  │            │
│                                  │  │ GenerativeModel│  │            │
│                                  │  └────────────────┘  │            │
│                                  └──────────────────────┘            │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             │ run_on_question() → Dict[str, Any]
             ▼
      ┌──────────────────┐
      │ Per-Question     │       ScoreOutput can be:
      │ Diagnostics      │       • float (simple score)
      │ Dict[str, Any]   │       • List[float] (per-token)
      └──────────┬───────┘       • Dict[str, Any] (structured)
                 │
                 │ aggregate_metrics()
                 ▼
      ┌──────────────────┐
      │  CheckResult     │
      │ ┌──────────────┐ │
      │ │ check_name   │ │
      │ │ question_    │ │
      │ │   results    │ │
      │ │ metrics      │ │
      │ └──────────────┘ │
      └──────────┬───────┘
                 │
                 │ 3. Combine all checks
                 ▼
      ┌──────────────────────────┐
      │  BenchmarkEvaluation     │
      │ ┌──────────────────────┐ │
      │ │ benchmark_name       │ │
      │ │ questions (with all  │ │
      │ │   check results)     │ │
      │ │ metrics (all checks) │ │
      │ └──────────────────────┘ │
      └──────────────────────────┘
```

### Core Data Structures

**BencheckQuestion:**
```python
@dataclass
class BencheckQuestion:
    id: str
    question: str
    choices: List[str]
    correct: Union[int, List[int], str, List[str]]  # Flexible format
    question_type: QuestionType = QuestionType.SINGLE_CHOICE
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**QuestionType Enum:**
```python
class QuestionType(Enum):
    SINGLE_CHOICE = "single_choice"       # One correct answer
    MULTIPLE_CHOICE = "multiple_choice"   # Multiple correct answers (true MCQ)
    OPEN_ENDED = "open_ended"            # Free-text answer (future)
```

**CheckResult:**
```python
@dataclass
class CheckResult:
    check_name: str
    question_results: Dict[str, Dict[str, Any]]  # Per-question diagnostics
    metrics: Dict[str, Any]  # Aggregated metrics
```

**BenchmarkEvaluation:**
```python
@dataclass
class BenchmarkEvaluation:
    benchmark_name: str
    questions: Dict[str, Dict[str, Any]]  # Question-centric structure
    metrics: Dict[str, Dict[str, Any]]    # Metrics per check
```

### Available Components

**Adapters (Dataset Loaders):**
- `HellaSwagAdapter` - Commonsense reasoning (ActivityNet, WikiHow)
- `PIQAAdapter` - Physical commonsense reasoning
- `WinograndeAdapter` - Pronoun resolution
- `MMLUAdapter` - Multitask knowledge (57 subjects)
- `TruthfulQAAdapter` - Factual accuracy testing
- `ARC_Adapter` - Science reasoning (AI2)
- `CommonsenseQAAdapter` - Commonsense QA
- `GoldenSwagAdapter` - Enhanced HellaSwag variant
- `BigBenchAdapter` - Various BigBench tasks
- `JsonAdapter` - Custom JSON datasets
- `DummyAdapter` - Testing/development

**Models:**
- `VLLMModel` / `OpenAIModel` - Local GPU inference via vLLM server (supports log-likelihood + generation)
- `GeminiModel` - Cloud API (generation only)
- `DummyModel` - Lexical overlap (testing)

**Checks:**
- `LengthBiasCheck` - Model-free length bias analysis
- `EnumerationBiasCheck` - Positional bias detection (scoring-independent)
- `NoneOfTheAboveCheck` - Distractor quality testing (scoring-independent)
- `ContextRequirementCheck` - Context dependency analysis (scoring-dependent)
- `GrammarQualityCheck` - Grammar/spelling issue detection (scoring-independent)

---

## MCQ vs Single-Choice Handling

BenCheck distinguishes between **single-choice questions** (one correct answer) and **true multiple-choice questions** (multiple correct answers). This distinction is critical for proper evaluation and interpretation of results.

### Question Type Detection

The `BaseMCQAdapter` automatically detects question type based on the `correct` answer format:

**Single-choice examples (question_type = SINGLE_CHOICE):**
```python
# Single integer
correct = 2

# Single-element list
correct = [1]

# String letter
correct = "C"

# TruthfulQA mc1 format (binary array with one 1)
correct = [0, 1, 0, 0]
```

**Multiple-choice examples (question_type = MULTIPLE_CHOICE):**
```python
# Multiple integers
correct = [0, 2, 3]

# TruthfulQA mc2 format (binary array with multiple 1s)
correct = [1, 0, 1, 1]
```

Detection logic (in `BaseMCQAdapter._convert_record()`):
```python
def _detect_question_type(self, correct_field):
    """Automatically detect if question is single-choice or MCQ."""
    if isinstance(correct_field, list):
        # Binary label array (e.g., [1,0,1,1])
        if all(isinstance(x, int) and x in [0, 1] for x in correct_field):
            num_correct = sum(correct_field)
            return QuestionType.SINGLE_CHOICE if num_correct == 1 else QuestionType.MULTIPLE_CHOICE

        # List of indices (e.g., [0, 2, 3])
        if len(correct_field) == 1:
            return QuestionType.SINGLE_CHOICE
        elif len(correct_field) > 1:
            return QuestionType.MULTIPLE_CHOICE

    # Single value (int, str)
    return QuestionType.SINGLE_CHOICE
```

### Check Behavior with MCQ Questions

**Current Implementation:**
Most checks are designed for single-choice questions. When processing MULTIPLE_CHOICE questions:

1. Checks use **only the first correct answer** from the list
2. A warning is logged to alert users about potential inaccuracy
3. Results may not fully represent the question's true difficulty

**Checks with MCQ Warnings:**
- `LengthBiasCheck` - Compares only first correct answer's length
- `EnumerationBiasCheck` - Considers only first correct position

**Example:**
```python
# MCQ question with 3 correct answers
question = BencheckQuestion(
    id="mcq_001",
    question="Which are prime numbers?",
    choices=["2", "3", "4", "5"],
    correct=[0, 1, 3],  # Indices 0, 1, 3 are correct
    question_type=QuestionType.MULTIPLE_CHOICE
)

# LengthBiasCheck will:
# - Use only index 0 for length comparison
# - Log: "Warning: Question mcq_001 is MULTIPLE_CHOICE but check assumes single-choice"
```

### Helper Functions for MCQ

**Extract ALL correct indices (for MCQ support):**
```python
from bencheck.utils.mcq_helpers import extract_label_indices

# Example usage
correct = [1, 0, 1, 1]  # TruthfulQA mc2 format
indices = extract_label_indices(correct)
# Returns: [0, 2, 3]
```

**Extract single correct index (for single-choice):**
```python
from bencheck.checks._utils import normalize_correct_index

# Single-choice usage (no warning)
correct = 2
idx = normalize_correct_index(correct, question_id="q1")
# Returns: 2

# MCQ usage (logs warning about data loss)
correct = [0, 2, 3]
idx = normalize_correct_index(correct, question_id="q2")
# Logs: "Warning: Question q2 has multiple correct answers [0, 2, 3], using first one"
# Returns: 0
```

### Backward Compatibility

- **v0.1.x**: All questions treated as single-choice
- **v0.2.0**: Explicit `question_type` field added
- **Migration**: Existing adapters automatically set `question_type = SINGLE_CHOICE` for backward compatibility

### Example: TruthfulQA mc2 Format

TruthfulQA has two subtasks:
- **mc1**: Single correct answer (single-choice)
- **mc2**: Multiple correct answers (true MCQ)

```python
# TruthfulQA mc1 record
{
    "question": "What is the capital of France?",
    "choices": ["London", "Paris", "Berlin", "Madrid"],
    "mc1_targets": [0, 1, 0, 0]  # One correct answer
}
# Detected as: SINGLE_CHOICE

# TruthfulQA mc2 record
{
    "question": "Which are European capitals?",
    "choices": ["London", "Paris", "Berlin", "Tokyo"],
    "mc2_targets": [1, 1, 1, 0]  # Multiple correct answers
}
# Detected as: MULTIPLE_CHOICE
# correct field set to: [0, 1, 2]
```

### Implementation Details

Key files:
- `bencheck/adapters/base_mcq.py` - Question type detection logic (lines 156-206)
- `bencheck/utils/mcq_helpers.py` - MCQ field extraction utilities
- `bencheck/checks/_utils.py` - `normalize_correct_index()` helper
- `tests/test_mcq_handling.py` - 38 comprehensive tests

---

## Extension Guide

### Creating a Custom Check

```python
from bencheck.base import BencheckCheck
from bencheck.types import BencheckQuestion
from typing import Any, Dict, Optional

class MyCustomCheck(BencheckCheck):
    """Example custom diagnostic check."""

    name = "my_custom_check"

    def __init__(self, my_param: str = "default"):
        self.my_param = my_param

    def run_on_question(
        self, question: BencheckQuestion, model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Run check on a single question."""
        # Your logic here
        result = {
            "some_metric": 0.5,
            "some_flag": True,
        }
        return result

    def aggregate_metrics(
        self, question_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate per-question results into dataset-level metrics."""
        n_questions = len(question_results)

        # Example aggregation
        flags_count = sum(
            1 for r in question_results.values() if r.get("some_flag", False)
        )

        return {
            "n_questions": n_questions,
            "flag_percentage": (flags_count / n_questions) * 100 if n_questions > 0 else 0.0,
        }
```

### Creating a Custom Adapter

**For MCQ datasets (recommended):**
```python
from bencheck.adapters.base_mcq import BaseMCQAdapter

class MyDatasetAdapter(BaseMCQAdapter):
    """Adapter for My Custom Dataset."""

    # HuggingFace dataset identifiers
    dataset_name = "my-org/my-dataset"
    config_name = None  # or specific config

    # Field mappings (auto-detected if None)
    question_key = "question"
    option_key = "choices"
    label_key = "answer"
    context_key = None  # or "context" if separate field

    # Metadata fields to extract
    extra_metadata_keys = ["category", "difficulty"]
```

**For non-MCQ datasets:**
```python
from bencheck.adapters.base import DatasetAdapter
from bencheck.types import BencheckQuestion
from typing import Any, List

class CustomAdapter(DatasetAdapter):
    def load(self, source: Any | None = None) -> List[BencheckQuestion]:
        # Load your data
        data = self._load_from_source(source)

        # Convert to BencheckQuestion format
        questions = []
        for item in data:
            q = BencheckQuestion(
                id=item["id"],
                question=item["text"],
                choices=item["options"],
                correct=item["label"],
                metadata={"source": "custom"}
            )
            questions.append(q)

        return questions
```

### Creating a Custom Model

Models use Protocol (duck typing) - just implement the required methods:

**Scoring Model:**
```python
class MyCustomModel:
    """Custom model implementing ScoringModel protocol."""

    def score_continuation(self, context: str, continuation: str) -> float:
        """Score a single continuation."""
        # Your scoring logic
        score = self._compute_score(context, continuation)
        return score

    def score_continuations(
        self, context: str, continuations: List[str]
    ) -> List[float]:
        """Score multiple continuations (can be batched)."""
        return [self.score_continuation(context, c) for c in continuations]

    def _compute_score(self, context: str, continuation: str) -> float:
        # Your model's scoring implementation
        return 0.0
```

**Generative Model:**
```python
class MyGenerativeModel:
    """Implements GenerativeModel protocol."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        # Your generation logic
        output = self._generate_text(prompt, **kwargs)
        return output
```

**Combined Model:**
```python
class MyCombinedModel:
    """Implements both ScoringModel and GenerativeModel protocols."""

    def score_continuation(self, context: str, continuation: str) -> float:
        return self._score(context, continuation)

    def score_continuations(
        self, context: str, continuations: List[str]
    ) -> List[float]:
        return [self._score(context, c) for c in continuations]

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._generate(prompt, **kwargs)
```

---

## Design Decisions

### Why per-check models? (v0.2.0)

Different checks may need different models:
- Length Bias: No model needed (dataset analysis)
- Enumeration Bias: Needs log-likelihood scoring (encoder/decoder models)
- Context Requirement: Can use either scoring or generation
- Grammar Quality: Needs specialized grammar checking model

Per-check models allow:
- Model-free checks alongside model-dependent checks
- Different model types for different checks
- More flexible evaluation strategies

**Before v0.2.0 (global model):**
```python
runner = BencheckRunner(adapter=adapter, checks=checks, model=model)  #  Inflexible
```

**v0.2.0 (per-check models):**
```python
length_check = LengthBiasCheck()  # No model
enum_check = EnumerationBiasCheck(model=vllm_model)  # vLLM
context_check = ContextRequirementCheck(model=gemini_model)  # Gemini
runner = BencheckRunner(adapter=adapter, checks=[length_check, enum_check, context_check])  #  Flexible
```

### Why Protocol instead of ABC?

**Protocols (PEP 544) enable duck typing:**
- Third-party models work without modification
- No inheritance required
- Structural subtyping (if it has the methods, it works)

```python
# This works even though MyModel doesn't inherit from anything:
class MyModel:
    def score_continuation(self, context: str, continuation: str) -> float:
        return 0.0

    def score_continuations(self, context: str, continuations: List[str]) -> List[float]:
        return [0.0] * len(continuations)

model = MyModel()  #  Implements ScoringModel protocol
check = EnumerationBiasCheck(model=model)  #  Works!
```

### Why flexible ScoreOutput?

Different models return scores in different formats:
- Simple float (mean log-likelihood)
- List of floats (per-token scores)
- Dict with metadata (scores + uncertainty + tokens)

`ScoreOutput = Union[float, List[float], Dict[str, Any]]` accommodates all.

### Why flexible `correct` field?

Benchmarks use different formats:
- 0-based index: `0`
- 1-based index: `1`
- Letter: `"A"`
- Multiple correct: `[0, 2]`
- Binary array: `[1, 0, 1, 1]` (TruthfulQA mc2)

BenCheck preserves the native format. Checks normalize as needed using helpers in `bencheck/utils/mcq_helpers.py` and `bencheck/checks/_utils.py`.

### Why single `BenchmarkEvaluation` class?

Question-centric structure enables:
- Easy lookup of all diagnostics for a question
- Comparison across checks for same questions
- Simple serialization/deserialization
- Natural data organization

Alternative (check-centric) would require complex cross-referencing to answer: "What are all the issues with question X?"

---

## Additional Resources

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Installation, running experiments, testing, development workflow
- **[CHECKS.md](CHECKS.md)** - Complete diagnostic checks reference with metrics and interpretation
- **[README.md](../README.md)** - Project overview and quick start

---

*Last updated: 2026-01-03 (v0.2.0)*
