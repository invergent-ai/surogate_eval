# Surogate Eval

Surogate Eval is the core evaluation engine for the **Surogate LLMOps framework**.  
It provides a unified interface for benchmarking, security scanning (Red Teaming),
stress testing, and custom metric evaluation for Large Language Models.

---

## 🚀 Quick Start

### Installation

Install the package using `uv` (recommended) or `pip`:
```bash
# Basic installation
uv pip install -e .

# With specific extras
uv pip install -e ".[dev,security]"

# Full installation with all extras
uv pip install -e ".[all]"

# Install from lock file (recommended for reproducibility)
uv sync
```

#### Available Extras

| Extra | Description |
|-------|-------------|
| `dev` | Development tools (pytest, ruff, black) |
| `security` | Red teaming & guardrails (deepteam, deepeval) |
| `vision` | Vision-language model evaluation (ms-vlmeval) |
| `rag` | RAG evaluation (ragas, llama-index) |
| `inference` | Local inference backends (vllm, sglang, flash-attn) |
| `all` | All of the above |

### Basic Usage

The framework is primarily driven by the `surogate-eval` CLI command.
```bash
# Run an evaluation using a config file
surogate-eval eval --config configs/my_eval.yaml

# Run specific target only
surogate-eval eval --config configs/my_eval.yaml --target openrouter-gpt4

# List previous evaluation results
surogate-eval eval --list

# View a specific result file
surogate-eval eval --view results_20240114_120000.json

# Compare two evaluation runs
surogate-eval eval --compare run1.json run2.json
```

---

## 🛠 Features

**Multi-Target Evaluation**  
Evaluate multiple models (Local, API-based, or Custom) in a single run.

**Security & Guardrails**  
Integrated Red-Teaming via `deepteam` and automated guardrail validation.

**Benchmark Integration**  
Native support for standard benchmarks like **MMLU**, **GSM8K**, **ARC**, **HellaSwag**, and more via `evalscope`.

**Custom Dataset Support**  
Use translated or custom datasets for benchmarks from local paths or HuggingFace.

**Stress Testing**  
Measure throughput, latency, and resource consumption under load.

**Distributed Execution**  
Automatic detection of multi-GPU setups using `torch.distributed`.

---

## 📋 Configuration

Evaluations are defined in YAML configuration files.

### Standard Evaluation
```yaml
project:
  name: "Llama-3-Check"
  version: "1.0.0"

targets:
  - name: "llama3-8b"
    type: llm
    provider: openai
    model: meta-llama/llama-3.1-8b-instruct
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}

    evaluations:
      - name: "General Knowledge"
        dataset: data/general_qa.jsonl
        metrics:
          - name: correctness
            type: g_eval
            criteria: "Is the response accurate?"
            judge_model:
              target: llama3-8b
          - name: latency
            type: latency
            threshold_ms: 5000

    red_teaming:
      enabled: true
      vulnerabilities:
        - toxicity
        - prompt_leakage
```

### Benchmarks with Custom Datasets
```yaml
targets:
  - name: "gpt-4"
    type: llm
    provider: openai
    model: openai/gpt-4-turbo-preview
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}

    evaluations:
      - name: "benchmarks"
        benchmarks:
          # Default dataset (ModelScope)
          - name: mmlu
            num_fewshot: 5
            limit: 100

          # HuggingFace translated dataset
          - name: gsm8k
            num_fewshot: 3
            limit: 50
            dataset_hub: huggingface
            dataset_path: OpenLLM-Ro/ro_gsm8k

          # Local dataset
          - name: arc_challenge
            num_fewshot: 3
            dataset_path: ./datasets/arc_romanian
```

---

## 📂 Project Structure
```text
.
├── pyproject.toml              # Dependencies and entry-points
├── uv.lock                     # Locked dependency versions
├── surogate_eval/              # Main package
│   ├── cli/
│   │   ├── main.py             # CLI entry point
│   │   └── eval.py             # Evaluation command
│   ├── benchmarks/
│   │   ├── backends/
│   │   │   └── evalscope_backend.py  # EvalScope integration
│   │   └── registry.py         # Benchmark registry
│   ├── metrics/                # Evaluation metrics
│   ├── targets/                # Target model interfaces
│   └── utils/                  # Logging and utilities
├── examples/
│   ├── config.yaml             # Example configuration
│   └── datasets/               # Sample datasets
└── eval_results/               # Output directory
```

---

## 📊 Results & Reporting

Results are saved as JSON files in `eval_results/`:
```json
{
  "project": {"name": "my-eval", "version": "1.0.0"},
  "timestamp": "2026-01-14T15:00:00",
  "summary": {
    "total_targets": 2,
    "total_evaluations": 5
  },
  "targets": [
    {
      "name": "gpt-4",
      "status": "success",
      "evaluations": [...],
      "benchmarks": [...]
    }
  ]
}
```

**Includes:**
- Project metadata and timestamps
- Summary statistics across all targets
- Detailed per-test-case inputs, outputs, and scores
- Benchmark results with task breakdowns
- Security findings from red-teaming and guardrails

---

## 🔧 Development
```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Format
black .
```

**Lock dependencies after updates:**
```bash
uv lock
```

---

## 🛡 License

This project is licensed under the **AGPL-3.0 License**.