# Diagnostic Checks Reference

Complete guide to BenCheck's diagnostic checks for evaluating benchmark quality. Each check targets specific types of bias or quality issues.

---

## Check Classification

Checks are classified by when they run and what they require:

| Classification | Runs | Caching | Requires Model |
|----------------|------|---------|----------------|
| **Model-Free** | Once per dataset | Yes (by dataset) | No |
| **Scoring-Independent** | Once per (dataset, model) | Yes (by dataset+model) | Yes (always uses log_likelihood) |
| **Scoring-Dependent** | n_runs × n_modes times | No | Yes (depends on scoring_mode) |

---

## 1. Length Bias Check

**Type:** Model-Free
**File:** `bencheck/checks/length_bias.py`

### Purpose
Analyzes whether correct answers systematically differ in length from incorrect ones, which can enable trivial length-based shortcuts. This can indicate dataset construction artifacts where length becomes a spurious signal.

### How it Works
- Measures length of each option in subword tokens (default), words, or bytes
- Compares correct answer length strictly against distractor lengths — the two flags are mutually exclusive (a tie counts as neither)
- Calculates relative length differences
- Tests if correct answers are systematically longer/shorter

### MCQ Handling
This check assumes single-choice questions. When processing questions with multiple correct answers (question_type = MULTIPLE_CHOICE), it uses only the first correct answer and logs a warning. Results may not be accurate for true MCQ datasets.

### Metrics

| Metric | Type | Interpretation |
|--------|------|----------------|
| `median_relative_length_diff` | 0.0-1.0 | How varied answer lengths are within questions.<br>**0.0** = all answers same length<br>**1.0** = extreme length variation<br>**Concern if**: > 0.5 (high variability enables shortcuts) |
| `mean_correct_length` | Float | Average length of correct answers (in tokens/words/bytes) |
| `mean_incorrect_length` | Float | Average length of incorrect answers (in tokens/words/bytes) |
| `correct_is_longest_pct` | 0-100% | How often the correct answer is **strictly** longer than all distractors<br>**Expected** (4-choice): ~25% (uniform)<br>**Concern if**: > 35% or < 15% (systematic bias) |
| `correct_is_shortest_pct` | 0-100% | How often the correct answer is **strictly** shorter than all distractors<br>**Expected** (4-choice): ~25% (uniform)<br>**Concern if**: > 35% or < 15% (systematic bias)<br>**Note**: mutually exclusive with `correct_is_longest_pct` — their sum can never exceed 100% |
| `length_rank_distribution` | Array | Distribution of correct answer ranks by length<br>[shortest, 2nd, 3rd, longest]<br>**Expected** (4-choice): ~[25%, 25%, 25%, 25%] |

### Interpretation Guide

**Healthy benchmark**:
```json
{
  "median_relative_length_diff": 0.25,      // Moderate variation
  "correct_is_longest_pct": 24.5,           // ~25% (uniform)
  "correct_is_shortest_pct": 26.1,          // ~25% (uniform)
  "mean_correct_length": 8.3,               // Similar to incorrect
  "mean_incorrect_length": 8.1
}
```

**Problematic benchmark** (length bias):
```json
{
  "median_relative_length_diff": 0.68,      // HIGH - enables shortcuts
  "correct_is_longest_pct": 45.2,           // BIASED - correct answers tend to be longest
  "correct_is_shortest_pct": 8.1,           // BIASED - rarely shortest
  "mean_correct_length": 15.4,              // Much longer than incorrect
  "mean_incorrect_length": 6.2
}
```

### Red Flags
- `median_relative_length_diff` > 0.5 → High length variability enables shortcuts
- `mean_correct_length` significantly different from `mean_incorrect_length`
- `correct_is_longest_pct` > 35% or < 15% → Systematic bias
- `correct_is_shortest_pct` > 35% or < 15% → Systematic bias

### Configuration
```yaml
checks:
  - name: "LengthBiasCheck"
    model: null
    params:
      length_metric: "tokens"           # default; "words" or "bytes" also supported
      tokenizer_name_or_path: "google/gemma-3-12b-it"  # HF name or local path
```

`length_metric: "tokens"` counts subword tokens using the specified HuggingFace tokenizer (loaded once at init, cached for the process). This is the most meaningful unit for LLM benchmarks since it reflects what the model actually processes. Falls back to word count if the tokenizer cannot be loaded.

---

## 2. Enumeration Bias Check

**Type:** Scoring-Independent
**File:** `bencheck/checks/enumeration_bias.py`

### Purpose
Detects positional bias where models prefer specific positions (A/B/C/D or 0/1/2/3) regardless of content. Also checks if dataset has uniform distribution of correct answers across positions.

### How it Works
- Uses `model.score_continuations()` (log_likelihood, deterministic)
- Analyzes both **dataset distribution** (where gold answers are) and **model predictions** (what model chooses)
- Computes accuracy broken down by position of correct answer
- Performs chi-squared tests for uniformity

### Why Model is Required
While dataset distribution is model-free, enumeration bias is most meaningful when comparing:
1. **Dataset distribution** (is it uniform?)
2. **Model predictions** (does model follow dataset bias or create its own?)
3. **Combined analysis** (is poor performance due to dataset or model?)

### MCQ Handling
This check assumes single-choice questions. When processing questions with multiple correct answers (question_type = MULTIPLE_CHOICE), it uses only the first correct answer and logs a warning. Results may not be accurate for true MCQ datasets.

### Metrics

| Metric | Type | Interpretation |
|--------|------|----------------|
| `overall_accuracy` | 0-100% | Model's accuracy on this dataset<br>**Not** a benchmark quality metric - just performance baseline |
| `n_positions` | Integer | Number of answer positions (typically 4 for A/B/C/D) |
| `gold_position_dist` | Array | How correct answers are distributed across positions<br>**Expected**: ~uniform [25%, 25%, 25%, 25%]<br>**Concern if**: Highly skewed (e.g., [40%, 20%, 20%, 20%]) |
| `pred_position_dist` | Array | How model predictions are distributed across positions<br>**Expected**: ~uniform (no position preference)<br>**Concern if**: Highly skewed (model prefers certain positions) |
| `gold_chi2_p_value` | 0-1 | Statistical test for uniformity of correct answer positions<br>**> 0.05**: Uniform distribution (good)<br>**< 0.05**: Non-uniform (dataset bias) |
| `pred_chi2_p_value` | 0-1 | Statistical test for uniformity of model predictions<br>**> 0.05**: No position preference (good)<br>**< 0.05**: Model has positional bias |
| `accuracy_by_gold_position` | Dict | Accuracy broken down by where the correct answer appears<br>**Expected**: Similar across all positions<br>**Concern if**: Large variance (e.g., 80% on position 0, 40% on position 3) |

### Interpretation Guide

**Healthy benchmark + unbiased model**:
```json
{
  "overall_accuracy": 72.5,
  "gold_position_dist": [26, 24, 25, 25],       // ~Uniform (no dataset bias)
  "pred_position_dist": [24, 26, 25, 25],       // ~Uniform (no model bias)
  "gold_chi2_p_value": 0.842,                    // > 0.05 ✓ (uniform)
  "pred_chi2_p_value": 0.731,                    // > 0.05 ✓ (uniform)
  "accuracy_by_gold_position": {
    "position_0": 73.1,
    "position_1": 71.8,
    "position_2": 72.9,
    "position_3": 72.3                           // Similar across positions ✓
  }
}
```

**Problematic** (biased dataset):
```json
{
  "gold_position_dist": [45, 20, 18, 17],       // SKEWED - correct answer often in position 0
  "gold_chi2_p_value": 0.003,                    // < 0.05 ✗ (significant bias)
  "accuracy_by_gold_position": {
    "position_0": 85.2,                          // Much higher when answer is first
    "position_1": 65.3,
    "position_2": 62.1,
    "position_3": 58.9
  }
}
```

**Problematic** (model has positional preference):
```json
{
  "pred_position_dist": [52, 28, 12, 8],        // Model strongly prefers position 0
  "pred_chi2_p_value": 0.001                     // < 0.05 ✗ (model bias)
}
```

### Red Flags
- `gold_chi2_p_value` < 0.05 → Dataset has non-uniform distribution of correct answers
- `pred_chi2_p_value` < 0.05 → Model predictions are non-uniform (positional bias)
- Large variance in `accuracy_by_gold_position` → Performance varies by position

### Configuration
```yaml
checks:
  - name: "EnumerationBiasCheck"
    # Automatically uses model from experiment.models
```

---

## 3. None of the Above Check

**Type:** Scoring-Independent (Generation-Only)
**File:** `bencheck/checks/none_above.py`

### Purpose
Tests distractor quality by removing the correct answer and adding "None of the above" as the last option. If the model selects the placeholder, it indicates the incorrect options were obviously wrong.

**Important:**
- This check **ONLY works with generation-based models** (requires `.generate()` method)
- Log-likelihood scoring is NOT supported for this check
- Designed for benchmarks where "None of the above" is NOT a legitimate answer choice

### How it Works
- Uses `model.generate()` with multiple-choice prompts
- Run 1: Generate answer with original options → baseline prediction
- Run 2: Remove correct answer, add "None of the above" at the end → placeholder prediction
- If model chooses "None of the above", distractors were too easy to eliminate

### MCQ Handling
This check assumes single-choice questions. When processing questions with multiple correct answers, it uses only the first correct answer. Future versions will add MCQ-specific warnings.

### Metrics

| Metric | Type | Interpretation |
|--------|------|----------------|
| `baseline_accuracy` | 0-100% | Accuracy on original questions (normal scenario) |
| `baseline_correct_count` | Integer | Number of questions answered correctly in baseline |
| `placeholder_selected_rate` | 0-100% | How often model chose "None of the above"<br>**Expected**: Low < 10% (good distractors)<br>**Concern if**: High > 30% (distractors obviously wrong) |
| `placeholder_selected_count` | Integer | Count of times "None of the above" was selected |
| `prediction_changed_rate` | 0-100% | How often model's prediction changed when correct answer was removed<br>**Expected**: High ~70-80% (test is working)<br>**Concern if**: Low < 50% (test ineffective) |
| `prediction_changed_count` | Integer | Count of questions where prediction changed |

### Interpretation Guide

**Healthy benchmark** (strong distractors):
```json
{
  "baseline_accuracy": 75.2,
  "baseline_correct_count": 752,
  "placeholder_selected_rate": 8.5,              // Low ✓ - rarely chooses placeholder
  "placeholder_selected_count": 85,
  "prediction_changed_rate": 82.1,               // High ✓ - test is working
  "prediction_changed_count": 821
}
```
**Interpretation**: Model rarely selects "None of the above" (8.5%), indicating strong distractors. When the correct answer is removed, predictions change 82% of the time, showing the test is effective.

**Problematic benchmark** (weak distractors):
```json
{
  "baseline_accuracy": 72.5,
  "baseline_correct_count": 725,
  "placeholder_selected_rate": 35.2,             // HIGH ✗ - frequently chooses placeholder
  "placeholder_selected_count": 352,
  "prediction_changed_rate": 78.4,
  "prediction_changed_count": 784
}
```
**Interpretation**: Model frequently selects "None of the above" (35.2%) when the correct answer is removed, indicating that the remaining distractors are obviously wrong. This reveals poor distractor quality.

### Red Flags
- `placeholder_selected_rate` > 30% → Model frequently identifies distractors as obviously wrong
- `prediction_changed_rate` < 50% → Test not working properly (predictions should change)

### Configuration
```yaml
checks:
  - name: "NoneOfTheAboveCheck"
    # Automatically uses generation model from experiment.models
    params:
      placeholder_text: "None of the above"  # Optional, this is the default
```

**Note**: This check requires a generative model with `.generate()` method. It will NOT work with log-likelihood-only models.

---

## 4. Context Requirement Check

**Type:** Scoring-Dependent
**File:** `bencheck/checks/context_requirement.py`

### Purpose
Tests whether questions actually require their context (question text) or if they can be answered from choices alone. High accuracy on empty/Lorem Ipsum questions indicates bias or obvious distractors.

### How it Works
Supports two scoring modes:

#### Log-Likelihood Mode (deterministic)
- Uses `model.score_continuations()`
- Run 1: Score original question → baseline
- Run 2: Score with empty string → empty accuracy
- Run 3: Score with Lorem Ipsum → lorem accuracy

#### Generation Mode (non-deterministic if temperature > 0)
- Uses `model.generate()`
- Same 3 runs, but with text generation
- Multiple runs needed due to sampling

### Test Variants

Two transformation types (set via `transform_type` parameter):

1. **Empty**: Replace question text with empty string
2. **Lorem Ipsum**: Replace question text with Lorem Ipsum filler
3. **Both**: Test both transformations

### MCQ Handling
This check assumes single-choice questions. When processing questions with multiple correct answers, it uses only the first correct answer. Future versions will add MCQ-specific warnings.

### Metrics

| Metric | Type | Interpretation |
|--------|------|----------------|
| `baseline_accuracy` | 0-100% | Accuracy with original (full) questions |
| `empty_accuracy` | 0-100% | Accuracy with empty question text<br>**Expected**: At or below random chance (1/n_choices)<br>**Random baseline**: 25% (4-choice), 50% (2-choice), 20% (5-choice)<br>**Concern if**: Significantly above random baseline |
| `lorem_accuracy` | 0-100% | Accuracy with Lorem Ipsum placeholder text<br>**Expected**: At or below random chance (1/n_choices)<br>**Random baseline**: 25% (4-choice), 50% (2-choice), 20% (5-choice)<br>**Concern if**: Significantly above random baseline |
| `empty_vs_baseline_drop` | Float | Accuracy drop from baseline to empty<br>**Expected**: Large (≈ baseline accuracy)<br>**Concern if**: Small < 40% |
| `lorem_vs_baseline_drop` | Float | Accuracy drop from baseline to Lorem Ipsum<br>**Expected**: Large (≈ baseline accuracy)<br>**Concern if**: Small < 40% |
| `empty_prediction_agreement_pct` | 0-100% | % of questions where the model picks the **same answer option** with empty prompt as with the full prompt (regardless of correctness)<br>**Expected**: Low — high agreement means the question text is irrelevant to the model's choice<br>**Concern if**: > 70% (model mostly ignores the question) |
| `lorem_prediction_agreement_pct` | 0-100% | Same as above but for Lorem Ipsum placeholder<br>**Expected**: Similar to or slightly lower than `empty_prediction_agreement_pct` |
| `mean_context_dependency` | 0-1 | Average context dependency score across questions<br>**1.0** = Fully context-dependent<br>**0.0** = No context needed<br>**Expected**: > 0.7 |
| `questions_requiring_context_pct` | 0-100% | Percentage of questions where **both** empty and lorem predictions differ from baseline (dependency = 1.0)<br>**Expected**: > 80%<br>**Concern if**: < 60% (many questions don't need context) |

### Interpretation Guide

**Healthy benchmark** (context-dependent, 4-choice example):
```json
{
  "baseline_accuracy": 74.3,
  "empty_accuracy": 12.1,                        // Below random (25% for 4-choice) ✓
  "lorem_accuracy": 15.8,                        // Below random (25% for 4-choice) ✓
  "empty_vs_baseline_drop": 62.2,                // Large drop ✓
  "lorem_vs_baseline_drop": 58.5,                // Large drop ✓
  "mean_context_dependency": 0.84,               // High ✓
  "questions_requiring_context_pct": 89.3        // >80% ✓
}
```

**Problematic benchmark** (answerable without context, 4-choice example):
```json
{
  "baseline_accuracy": 71.5,
  "empty_accuracy": 48.2,                        // Well above random (25% for 4-choice) ✗
  "lorem_accuracy": 45.9,                        // Well above random (25% for 4-choice) ✗
  "empty_vs_baseline_drop": 23.3,                // Small drop ✗
  "lorem_vs_baseline_drop": 25.6,                // Small drop ✗
  "mean_context_dependency": 0.32,               // Low ✗
  "questions_requiring_context_pct": 42.1        // <60% ✗
}
```
**Interpretation**: Model can answer ~48% of questions correctly without seeing the question text (almost double the random chance of 25% for 4-choice). This indicates questions are answerable from choices alone (poor question design).

**PIQA/Winogrande (2-choice) example:**
```json
{
  "baseline_accuracy": 78.5,
  "empty_accuracy": 52.3,                        // Slightly above random (50% for 2-choice) - borderline
  "lorem_accuracy": 54.1,                        // Slightly above random (50% for 2-choice) - borderline
  "empty_vs_baseline_drop": 26.2,
  "lorem_vs_baseline_drop": 24.4
}
```
**Interpretation**: For 2-choice datasets (PIQA, Winogrande), random baseline is 50%. Accuracy of 52-54% is only slightly above random, which is acceptable.

### Statistical Analysis
All runs include comprehensive statistical tests:
- Binomial tests (H0: performance = random guessing)
  - Random baseline automatically calculated as 1/n_choices for each dataset
  - Tests whether transformed accuracy significantly exceeds random chance
- Wilson score confidence intervals
- Cohen's h effect sizes
- Interpretations of whether questions require context

**Example statistical output:**
```json
{
  "statistical_analysis": {
    "empty": {
      "n_questions": 1838,
      "transformed_accuracy": 0.523,
      "random_baseline": 0.5,              // Automatically calculated for 2-choice (PIQA)
      "binomial_test_pvalue": 0.042,      // p < 0.05: significantly above random
      "cohens_h_vs_random": 0.046,
      "requires_context": false,
      "interpretation": "Performance on empty questions (52.3%) is significantly above random guessing (50.0%, p=0.042), but effect size is small (h=0.046). Questions may not strongly require context."
    }
  }
}
```

### Red Flags
- `empty_accuracy` significantly above random baseline → Many questions don't need context
  - **4-choice (HellaSwag, MMLU, ARC)**: > 35% (random = 25%)
  - **2-choice (PIQA, Winogrande)**: > 60% (random = 50%)
  - **5-choice (CommonsenseQA)**: > 30% (random = 20%)
- `lorem_accuracy` significantly above random baseline → Questions answerable with gibberish
  - Use same thresholds as `empty_accuracy` above
- `questions_requiring_context_pct` < 60% → Over half don't need context
- `mean_context_dependency` < 0.5 → Low overall dependency
- Low `empty_vs_baseline_drop` → Removing context barely hurts performance

**Note**: The check now automatically calculates the correct random baseline (1/n_choices) for each dataset, so statistical analysis will use the appropriate baseline.

**Note on random baselines:** Expected accuracy for empty/lorem questions should be at or below random chance (1/n_choices). Random baseline varies by dataset:

| Dataset | Choices | Random Baseline |
|---------|---------|-----------------|
| HellaSwag | 4 | 25% |
| MMLU | 4 | 25% |
| ARC-Challenge | 4 | 25% |
| CommonsenseQA | 5 | 20% |
| **PIQA** | **2** | **50%** |
| **Winogrande** | **2** | **50%** |
| TruthfulQA MC2 | Variable | Depends on question |

Accuracy significantly above the random baseline indicates the benchmark has context-independent issues.

### Configuration
```yaml
checks:
  - name: "ContextRequirementCheck"
    params:
      transform_type: "both"  # "empty", "lorem_ipsum", or "both"
      lorem_match_by: "characters"  # or "words"
      scoring_mode: "log_likelihood"  # or "generation"
      significance_level: 0.05
      min_effect_size: 0.2
```

---

## 5. Grammar Quality Check

**Type:** Scoring-Independent (Special Case)

Two implementations are available. **`LLMGrammarCheck` is recommended** — it uses an instruction-tuned LLM as judge and produces more reliable flags than the rule-based GECToR approach.

---

### 5a. LLMGrammarCheck (Recommended)

**File:** `bencheck/checks/llm_grammar.py`

#### Purpose
Detects grammar errors in MCQ questions and answer options using an LLM as judge (via vLLM with guided JSON decoding). Returns three boolean flags per question: grammar error in the question stem, in the correct answer, and in any distractor.

**Key Quality Signal:** Uneven grammar quality between correct answers and distractors is an unintended bias signal — models can use surface errors as a shortcut.

#### How it Works
- Presents the full question + all options to a generative model (e.g., Gemma 4)
- Requests three boolean flags via guided JSON decoding (guaranteed parseable output)
- Sends the entire dataset as a single batched vLLM call
- Regex fallback handles any truncated JSON output

#### Metrics

| Metric | Type | Interpretation |
|--------|------|----------------|
| `question_has_grammar_issue_rate` | 0-1 | Proportion of question stems with a grammar error |
| `gold_has_issue_rate` | 0-1 | Proportion of correct answers with a grammar error |
| `has_distractor_issue_rate` | 0-1 | Proportion of questions where any distractor has a grammar error |
| `has_issue_rate` | 0-1 | Proportion of questions with any grammar issue (stem or options) |

Per-question fields (in `per_questions/` files):
- `question_has_grammar_issue`: boolean
- `gold_has_issue`: boolean (null if gold index unknown)
- `has_distractor_issue`: boolean (null if gold index unknown)

#### Configuration
```yaml
checks:
  - name: "LLMGrammarCheck"
    model:
      type: "vllm"
      name: "google/gemma-4-E4B-it"   # or any instruction-tuned model
      params:
        gpu_memory_utilization: 0.9
        max_model_len: 4096
```

Run (requires a GPU):
```bash
python scripts/run_experiment.py configs/examples/llm_grammar.yaml
```

---

### 5b. GrammarQualityCheck (Legacy)

**File:** `bencheck/checks/grammar_quality.py`

Uses GECToR (a rule-based grammar error correction model) instead of an LLM. Prefer `LLMGrammarCheck` for new experiments — GECToR tends to over-flag informal but grammatically valid phrasing common in benchmark options.

#### Purpose
Detects grammar, spelling, and punctuation issues in MCQ answer options using an external detector model (GECToR).

#### How it Works
- Uses GECToR model (NOT the LLM being evaluated!)
- Caches by dataset (like model-free) since GECToR results are deterministic
- Analyzes each option for grammar issues
- Counts questions with issues in correct/incorrect answers

#### Metrics

| Metric | Type | Interpretation |
|--------|------|----------------|
| `n_questions` | Integer | Total questions evaluated |
| `Q_any_issue.count` | Integer | Number of questions with at least one flagged option |
| `Q_any_issue.rate` | 0-1 | Proportion of questions with any grammar/spelling issues<br>**Expected**: < 0.05 (< 5% of questions)<br>**Concern if**: > 0.1 (> 10% of questions have issues) |
| `Q_gold_issue.count` | Integer | Number of questions where the correct answer has grammar issues |
| `Q_gold_issue.evaluated` | Integer | Number of questions with valid gold labels |
| `Q_gold_issue.rate` | 0-1 | Proportion of correct answers with issues<br>**Expected**: < 0.02 (< 2%)<br>**Concern if**: > 0.05 (correct answers have more issues) |
| `flagged_option_rate.flagged` | Integer | Total number of options with issues |
| `flagged_option_rate.total` | Integer | Total number of options evaluated |
| `flagged_option_rate.rate` | 0-1 | Overall proportion of flagged options<br>**Expected**: < 0.05<br>**Concern if**: > 0.1 (widespread quality issues) |
| `distractor_issue_rate.flagged` | Integer | Number of distractor options with issues |
| `distractor_issue_rate.total` | Integer | Total number of distractor options |
| `distractor_issue_rate.rate` | 0-1 | Proportion of distractors with issues<br>**Expected**: Similar to or slightly higher than gold rate<br>**Concern if**: Much higher than gold (distractors lower quality) |
| `option_position_flagged_rate` | Dict | Issue rate broken down by option position (0, 1, 2, 3)<br>**Expected**: Similar rates across positions<br>**Concern if**: Specific positions consistently have more issues |

#### Per-Question Details

Each question includes option-level diagnostics:
- `original`: Original option text
- `normalized`: Text after preprocessing
- `suggested`: Corrected text from detector
- `flagged`: Boolean - whether option has issues
- `edit_count`: Number of edits detected
- `token_edits`: List of token-level changes
- `transformations`: List of applied transformations

#### Interpretation Guide

**Healthy benchmark**:
```json
{
  "n_questions": 1000,
  "Q_any_issue": {
    "count": 35,
    "rate": 0.035                               // 3.5% ✓
  },
  "Q_gold_issue": {
    "count": 8,
    "evaluated": 1000,
    "rate": 0.008                               // 0.8% ✓
  },
  "flagged_option_rate": {
    "flagged": 42,
    "total": 4000,
    "rate": 0.0105                              // 1.05% ✓
  },
  "distractor_issue_rate": {
    "flagged": 34,
    "total": 3000,
    "rate": 0.0113                              // 1.13% ✓
  }
}
```

**Problematic benchmark** (grammar issues):
```json
{
  "n_questions": 1000,
  "Q_any_issue": {
    "count": 152,
    "rate": 0.152                               // 15.2% ✗
  },
  "Q_gold_issue": {
    "count": 45,
    "evaluated": 1000,
    "rate": 0.045                               // 4.5% ✗ (correct answers have issues)
  },
  "flagged_option_rate": {
    "flagged": 223,
    "total": 4000,
    "rate": 0.0558                              // 5.58% ✗
  },
  "distractor_issue_rate": {
    "flagged": 178,
    "total": 3000,
    "rate": 0.0593                              // 5.93% ✗ (much higher than gold)
  }
}
```

#### Usage

GrammarQualityCheck (GECToR) can run in two modes:

1. **Rule-based (no model)**: Basic text normalization only
```python
check = GrammarQualityCheck()  # No detector
```

2. **With GECToR detector**: Advanced grammar/spelling detection
```python
from bencheck.models import GECToRGrammarModel

model = GECToRGrammarModel(
    model_paths=["path/to/gector-model.th"],
    vocab_path="path/to/output_vocabulary",
    tokenizer_name="roberta-base"
)
check = GrammarQualityCheck(model=model)
```

See `configs/examples/grammar_quality_gector.yaml` for full configuration example.

#### Red Flags
- `Q_any_issue.rate` > 0.1 → Over 10% of questions have grammar issues
- `Q_gold_issue.rate` > 0.05 → Over 5% of correct answers have issues
- `flagged_option_rate.rate` > 0.1 → Widespread quality issues
- `distractor_issue_rate.rate` >> `Q_gold_issue.rate` → Grammar signals correctness
- High `Q_distractor_only_issue` → Incorrect answers have more errors

#### Special Notes
- Classified as scoring-independent but caches like model-free
- Uses GECToR, not the evaluation model
- Optional dependency (requires setup):
  ```bash
  # Download GECToR vocabulary files
  bash scripts/setup_gector.sh
  # Install GECToR and dependencies
  pip install -e .[grammar]
  ```

#### Configuration
```yaml
checks:
  - name: "GrammarQualityCheck"
    model:
      type: "gector_grammar"
      name: "/path/to/gector/model.th"
      params:
        vocab_path: "models/gector/data/output_vocabulary"
        verb_dict_path: "models/gector/data/verb-form-vocab.txt"
        tokenizer_name: "roberta-base"
    params: {}
```

---

## Multi-Run Statistics

When running experiments with `n_runs > 1`, each metric is aggregated with comprehensive statistics:

### Statistical Metrics (per metric)

| Statistic | Meaning |
|-----------|---------|
| `mean` | Average value across N runs |
| `std` | Standard deviation (variability between runs) |
| `median` | Median value |
| `min` | Minimum value observed |
| `max` | Maximum value observed |
| `ci_95_lower` | Lower bound of 95% confidence interval (bootstrap) |
| `ci_95_upper` | Upper bound of 95% confidence interval (bootstrap) |
| `raw_values` | Array of individual run values |
| `n_runs` | Number of runs aggregated |

### Example Aggregated Output

```json
{
  "enumeration_bias": {
    "overall_accuracy": {
      "mean": 75.23,
      "std": 2.15,
      "min": 73.33,
      "max": 77.78,
      "median": 75.00,
      "ci_95_lower": 72.89,
      "ci_95_upper": 77.57,
      "raw_values": [75.23, 73.33, 77.78],
      "n_runs": 3
    }
  }
}
```

### Interpreting Variability

- **Low std (< 2%)**: Stable metric, reliable measurement
- **Medium std (2-5%)**: Some variability, confidence intervals important
- **High std (> 5%)**: High variability, may need more runs or indicates unstable model behavior

---

## Output Structure

### Model-Free & Scoring-Independent Checks
Single file per check, no runs/modes:
```
piqa_validation_/
  ├── length_bias.json          ← Model-free
  ├── enumeration_bias.json     ← Scoring-independent
  ├── none_of_the_above.json    ← Scoring-independent
  ├── grammar_quality.json      ← Scoring-independent (special case)
  └── per_questions/
      ├── length_bias.json
      ├── enumeration_bias.json
      ├── none_of_the_above.json
      └── grammar_quality.json
```

### Scoring-Dependent Checks
Single file with runs organized by mode:
```
piqa_validation_/
  ├── context_requirement.json  ← Aggregated (all runs, all modes)
  └── per_questions/
      ├── context_requirement_log_likelihood_run_0.json
      ├── context_requirement_log_likelihood_run_1.json
      ├── context_requirement_log_likelihood_run_2.json
      ├── context_requirement_generation_run_0.json
      ├── context_requirement_generation_run_1.json
      └── context_requirement_generation_run_2.json
```

---

## Caching Behavior

### Why Caching?
With `n_runs=3` and `scoring_modes=["log_likelihood", "generation"]`, each combination runs 3 times:
- 3 datasets × 2 modes × 3 runs = **18 total runs**
- Without caching: LengthBiasCheck runs **18 times** (same result!)
- With caching: LengthBiasCheck runs **3 times** (once per dataset)

### Cache Keys
- **Model-Free**: `(dataset)`
- **Scoring-Independent**: `(dataset, model)`
- **Scoring-Dependent**: No caching (depends on run_index and scoring_mode)

### Performance Impact
For `reasoning_datasets.yaml` (4 datasets, 2 modes, 3 runs = 24 runs):
- Model-free checks: **6x speedup** (4 runs instead of 24)
- Scoring-independent checks: **6x speedup** (4 runs instead of 24)
- Scoring-dependent checks: No speedup (must run all 24)

---

## Quick Reference: Red Flags

### Length Bias
-  `median_relative_length_diff` > 0.5
-  `correct_is_longest_pct` > 35% or < 15%
-  Large difference between `mean_correct_length` and `mean_incorrect_length`

### Enumeration Bias
-  `gold_chi2_p_value` < 0.05 (dataset has positional bias)
-  `pred_chi2_p_value` < 0.05 (model has positional preference)
-  Large variance in `accuracy_by_gold_position` (e.g., 80% vs 40%)

### None of the Above
-  `placeholder_selected_rate` > 30% (distractors obviously wrong)
-  `prediction_changed_rate` < 50% (test ineffective)

### Context Requirement
-  `empty_accuracy` significantly above random baseline (dataset-specific, see table below)
-  `lorem_accuracy` significantly above random baseline (dataset-specific, see table below)
-  `questions_requiring_context_pct` < 60% (many don't need context)
-  `mean_context_dependency` < 0.5 (low overall dependency)

**Random baseline thresholds by dataset:**
- 4-choice (HellaSwag, MMLU, ARC): > 35% is concerning (random = 25%)
- 2-choice (PIQA, Winogrande): > 60% is concerning (random = 50%)
- 5-choice (CommonsenseQA): > 30% is concerning (random = 20%)

### Grammar Quality (LLMGrammarCheck)
-  `has_distractor_issue_rate` >> `gold_has_issue_rate` (distractors have more errors than correct answers)
-  `has_issue_rate` > 0.1 (> 10% of questions have any grammar issue)
-  `gold_has_issue_rate` > 0.05 (> 5% of correct answers have issues)

### Grammar Quality (GrammarQualityCheck / GECToR)
-  `Q_any_issue.rate` > 0.1 (> 10% of questions have issues)
-  `Q_gold_issue.rate` > 0.05 (> 5% of correct answers have issues)
-  `flagged_option_rate.rate` > 0.1 (widespread quality issues)
-  `distractor_issue_rate.rate` >> `Q_gold_issue.rate` (distractors much lower quality)

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and implementation
- [DEVELOPMENT.md](DEVELOPMENT.md) - Quick start and development setup
- [README.md](../README.md) - Project overview
