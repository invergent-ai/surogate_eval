# Surogate Eval

Surogate Eval is the core evaluation engine for the **Surogate LLMOps framework**.  
It provides a unified interface for benchmarking, security scanning (Red Teaming),
stress testing, and custom metric evaluation for Large Language Models.

---

## 🚀 Quick Start

### Installation

Install the package in editable mode from the root directory:

```bash
# Basic installation
pip install -e .

# Full installation including security and inference backends
pip install -e ".[security,inference]"
```

### Basic Usage

The framework is primarily driven by the `surogate-eval` CLI command.

```bash
# Run an evaluation using a config file
surogate-eval eval --config configs/my_eval.yaml

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
Native support for standard benchmarks like **MMLU**, **GSM8K**, and more via `evalscope`.

**Stress Testing**  
Measure throughput, latency, and resource consumption under load.

**Distributed Execution**  
Automatic detection of multi-GPU setups using `torch.distributed`.

---

## 📋 Configuration

Evaluations are defined in YAML configuration files.  
Below is a standard example:

```yaml
project:
  name: "Llama-3-Check"
  version: "1.0.0"

targets:
  - name: "llama3-8b"
    type: "local"
    model: "meta-llama/Meta-Llama-3-8B-Instruct"
    evaluations:
      - name: "General Knowledge"
        dataset: "data/general_qa.jsonl"
        metrics:
          - type: "g_eval"
          - type: "latency"
    red_teaming:
      enabled: true
      vulnerabilities:
        - injection
        - bias
```

---

## 📂 Project Structure

The project follows the `src/` layout for robust packaging:

```text
.
├── pyproject.toml           # Dependency and entry-point management
├── src/
│   └── surogate_eval/       # Main package
│       ├── eval.py          # The SurogateEval Orchestrator
│       ├── cli/
│       │   ├── main.py      # Distributed CLI entry point
│       │   └── eval.py      # Argument parsing and mode selection
│       ├── backend/         # Execution backends (Local, Distributed)
│       ├── benchmarks/      # Standard benchmark integrations
│       └── utils/           # Shared logging and command utilities
```

---

## 📊 Results & Reporting

All results are consolidated into a single JSON file stored by default in
`eval_results/`.

These files include:

- **Project Metadata**  
  Versioning and timestamps

- **Summary Statistics**  
  Aggregated scores across all targets

- **Detailed Metrics**  
  Per-test-case inputs, outputs, and scores

- **Security Logs**  
  Findings from red-teaming and guardrail tests

---

## 🛡 License

This project is licensed under the **AGPL-3.0 License**.
