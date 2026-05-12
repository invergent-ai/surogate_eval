# Supported Benchmarks

## Agent

| Benchmark | Config name | Description | Samples | Metric | EvalScope |
|-----------|-------------|-------------|---------|--------|-----------|
| SWE-bench Verified | `swe_bench_verified` | Real GitHub issues requiring code understanding, planning, and multi-file edits | 500 | resolve% | Yes |
| SWE-bench Multilingual | `swe_bench_multilingual` | Cross-language software engineering tasks (Python, Java, TS, Go, Rust, C++) | 600 | resolve% | Yes |
| SWE-bench Pro | `swe_bench_pro` | Professional-difficulty multi-step debugging and refactoring tasks | 400 | resolve% | Yes |
| Terminal-Bench 2.0 | `terminal_bench` | Multi-step terminal tasks — command chaining, file manipulation, sysadmin | 89 | accuracy | Yes |
| TAU-Bench | `tau_bench` | Task-oriented agent benchmark for multi-step troubleshooting and decision-making | 500 | accuracy | Yes |
| BFCL v4 | `bfcl_v4` | Holistic agentic function calling — multi-turn tool use, web search, memory | 2000 | accuracy | Yes |
| MCPMark | `mcp_mark` | MCP tool discovery, invocation, and orchestration across servers | 300 | accuracy | No |
| MCP-Atlas | `mcp_atlas` | Large-scale MCP evaluation covering diverse server types and tool chaining | 500 | accuracy | No |
| Tool Decathlon | `tool_decathlon` | Ten categories of tool-use tasks testing breadth of agent capabilities | 400 | accuracy | No |
| DeepPlanning | `deep_planning` | Complex multi-constraint planning requiring long-horizon reasoning | 350 | accuracy | No |
| WideSearch | `wide_search` | Information retrieval and synthesis across large document collections | 400 | accuracy | No |
| VITA-Bench | `vita_bench` | Visual-interactive task automation — GUI navigation, web browsing, app interaction | 300 | accuracy | No |

## Coding

| Benchmark | Config name | Description | Samples | Metric | EvalScope |
|-----------|-------------|-------------|---------|--------|-----------|
| HumanEval | `humaneval` | Python function completion measuring code generation correctness | 164 | pass@1 | Yes |
| HumanEval+ | `humaneval_plus` | HumanEval with 80x more tests per problem to catch false positives | 164 | pass@1 | Yes |
| MBPP | `mbpp` | Mostly Basic Python Problems for code generation evaluation | 500 | pass@1 | Yes |
| LiveCodeBench v6 | `live_code_bench` | Continuously updated competitive programming — no data contamination | 880 | pass@1 | Yes |
| NL2Repo | `nl2repo` | Natural language to full repository generation from specifications | 200 | accuracy | No |
| SciCode | `scicode` | Scientific computing — multi-step research-level coding tasks | 300 | accuracy | Yes |
| MultiPL-E | `multipl_e` | HumanEval translated to 18+ languages (Rust, Go, Java, TS, etc.) | 164+ | pass@1 | Yes |

## Knowledge

| Benchmark | Config name | Description | Samples | Metric | EvalScope |
|-----------|-------------|-------------|---------|--------|-----------|
| MMLU | `mmlu` | Massive Multitask Language Understanding across 57 academic subjects | 14042 | accuracy | Yes |
| MMLU-Pro | `mmlu_pro` | 10-choice version emphasizing reasoning over recall | 12032 | accuracy | Yes |
| MMLU-Redux | `mmlu_redux` | Curated MMLU subset with corrected labels and reduced noise | 3000 | accuracy | Yes |
| SuperGPQA | `super_gpqa` | Extended graduate-level QA with multilingual expert-validated questions | 1000 | accuracy | Yes |

## Reasoning

| Benchmark | Config name | Description | Samples | Metric | EvalScope |
|-----------|-------------|-------------|---------|--------|-----------|
| GSM8K | `gsm8k` | Grade school math word problems testing multi-step arithmetic reasoning | 1319 | accuracy | Yes |
| GPQA | `gpqa` | Graduate-level expert questions in physics, chemistry, and biology | 448 | accuracy | Yes |
| ARC-AGI | `arc_agi` | Abstraction and Reasoning Corpus for fluid intelligence and pattern recognition | 400 | accuracy | Yes |
| MuSR | `musr` | Multi-step reasoning with chained deductions across complex narratives | 800 | accuracy | Yes |
| HLE | `hle` | Hard Language Evaluation — expert-level questions that challenge frontier models | 500 | accuracy | Yes |

## Instruction Following

| Benchmark | Config name | Description | Samples | Metric | EvalScope |
|-----------|-------------|-------------|---------|--------|-----------|
| IFEval | `ifeval` | Instruction Following Evaluation with 25+ verifiable constraint types | 541 | accuracy | Yes |
| IFBench | `ifbench` | Extended instruction following with complex compositional multi-constraint scenarios | 800 | accuracy | Yes |

## Safety

| Benchmark | Config name | Description | Samples | Metric | EvalScope |
|-----------|-------------|-------------|---------|--------|-----------|
| TruthfulQA | `truthfulqa` | Tests whether models generate truthful and informative answers | 817 | accuracy | Yes |
| ToxiGen | `toxigen` | Toxicity detection and generation safety across 13 demographic groups | 6541 | safety% | No |

## Language & Chat

| Benchmark | Config name | Description | Samples | Metric | EvalScope |
|-----------|-------------|-------------|---------|--------|-----------|
| HellaSwag | `hellaswag` | Sentence completion requiring commonsense reasoning | 10042 | accuracy | Yes |
| MT-Bench | `mt_bench` | Multi-turn conversation quality rated by GPT-4 judge | 80 | score/10 | No |

---

## Usage

```yaml
targets:
  - name: my-model
    type: llm
    provider: openai
    model: my-model-id
    base_url: http://localhost:8000/v1

    evaluations:
      - name: agent-suite
        benchmarks:
          - name: swe_bench_verified
            limit: 50
          - name: bfcl_v4
            limit: 100
          - name: terminal_bench
          - name: tau_bench
            limit: 50

      - name: coding-suite
        benchmarks:
          - name: humaneval
          - name: live_code_bench
            limit: 50
          - name: mbpp

      - name: knowledge-suite
        benchmarks:
          - name: mmlu_pro
            num_fewshot: 5
            limit: 100
          - name: gpqa
            num_fewshot: 0
          - name: ifeval
```
