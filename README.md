# BenCheck

**BenCheck audits multiple-choice benchmarks, not models.** It runs a set of
diagnostic checks over a benchmark and reports issues that undermine its
validity: answers you can pick without reading the question, length and
position artifacts, broken gold answers, and grammar errors or typos.

It packages the checks from the paper *Who Benchmarks the Benchmarks? Towards
Comprehensive Evaluation of Commonsense Reasoning Benchmarks* (Chizhov et al.,
2026), which applies them to HellaSwag, PIQA, Global PIQA and Winogrande.

## Checks

| Check | Needs a model | What it tells you |
|---|---|---|
| `LengthBiasCheck` | no | Whether the correct answer tends to be the longest or shortest option |
| `EnumerationBiasCheck` | yes | Whether gold answers or model predictions cluster in one position (A/B/C/D) |
| `ContextRequirementCheck` | yes | Prompt importance: accuracy and prediction agreement when the question text is removed or replaced with Lorem ipsum |
| `NoneOfTheAboveCheck` | yes | What the model picks when the gold answer is replaced with "None of the above" |
| `LLMGrammarCheck` | yes (vLLM) | Grammar issues and typos in the prompt, the correct answer and the distractors, via an open LLM with guided decoding |
| `GrammarQualityCheck` | GECToR | Legacy grammar detector based on GECToR |

Metrics and how to read them: [docs/CHECKS.md](docs/CHECKS.md).

Built-in adapters: HellaSwag, PIQA, Global PIQA, Winogrande, MMLU, ARC,
CommonsenseQA, TruthfulQA, BIG-bench, plus a generic Hugging Face MCQ adapter
and a JSON adapter for your own data.

## Install

Python 3.10+.

```bash
git clone https://github.com/Pleias/BenCheck.git
cd BenCheck
pip install -e .              # core: enough for the quickstart
pip install -e ".[openai]"    # model checks against a vLLM / OpenAI-compatible server
pip install -e ".[vllm]"      # in-process vLLM (needed by LLMGrammarCheck)
pip install -e ".[gemini]"    # Gemini API
```

## Quickstart (CPU, about a minute)

```bash
python scripts/run_experiment.py configs/examples/quickstart.yaml
```

This runs `LengthBiasCheck` on 100 HellaSwag questions and writes
`results/quickstart/`, including a readable `REPORT.md`.

## Running the model-dependent checks

Start an OpenAI-compatible server, then run a config:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-12b-it --host 0.0.0.0 --port 8008

python scripts/run_experiment.py configs/examples/audit_all_checks.yaml
```

The model id in the config (`models[].name`) must match the served model.

| Config | Checks | Data |
|---|---|---|
| `quickstart.yaml` | length | HellaSwag, 100 questions |
| `audit_all_checks.yaml` | length, enumeration, prompt importance, NOTA | PIQA + HellaSwag, 200 questions each |
| `llm_grammar.yaml` | LLM grammar (Gemma 4 E4B, in-process vLLM) | HellaSwag, 200 questions |
| `context_requirement_reasoning_datasets.yaml` | prompt importance | PIQA, Global PIQA, HellaSwag, Winogrande (full) |
| `none_above_reasoning_datasets.yaml` | NOTA | same four, full |
| `context_requirement_full_benchmarks.yaml`, `none_above_full_benchmarks.yaml` | prompt importance / NOTA | MMLU subjects, ARC, TruthfulQA, CommonsenseQA |
| `grammar_quality_gector.yaml` | GECToR grammar (run with `scripts/run_from_config.py`) | HellaSwag, 50 questions |

To audit a slice of any split, use Hugging Face slice syntax in the adapter
params, e.g. `split: "validation[:500]"`.

## Output

```
results/<experiment>/
├── experiment_summary.json    # all datasets and checks
├── REPORT.md                  # human-readable report
└── <dataset>_/
    ├── questions_base.json
    ├── <check>.json           # aggregate metrics
    └── per_questions/         # per-question diagnostics
```

## Documentation

- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): installation, configs, running, tests
- [docs/CHECKS.md](docs/CHECKS.md): every check, its metrics and red flags
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): adapters, checks, models, runner

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
