"""Tool Decathlon (Toolathlon) — trajectory-based tool-use evaluation.

Uses the hkust-nlp/Toolathlon-Trajectories dataset which contains 5000+
multi-turn tool-use trajectories across ~108 diverse tasks with real-world
MCP servers (filesystem, web search, terminal, etc.).

Since Toolathlon requires live Docker containers with MCP servers to run
natively, this adapter uses a trajectory-informed approach:
  1. Give the model the task description + available tools
  2. Let it plan / generate a response strategy
  3. Use LLM-as-judge to compare against the reference trajectory outcome

The dataset has datetime columns that crash PyArrow during non-streaming
load_dataset(), so this is implemented as a standalone benchmark that
streams the dataset from HuggingFace.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from surogate_eval.benchmarks.base import BaseBenchmark, BenchmarkConfig, BenchmarkResult
from surogate_eval.benchmarks.registry import BenchmarkRegistry
from surogate_eval.targets import BaseTarget
from surogate_eval.targets.base import TargetRequest
from surogate_eval.utils.logger import get_logger
from surogate_eval.benchmarks.pass_rule import LEGACY_JUDGE_MIN, row_passed

logger = get_logger()

DATASET_ID = "hkust-nlp/Toolathlon-Trajectories"

_SYSTEM_PROMPT = (
    "You are a capable AI agent with access to tools. "
    "Given a task and a set of available tools, explain step-by-step "
    "how you would accomplish the task. Be specific about which tools "
    "you would call, in what order, and with what arguments. "
    "Then provide the final answer or outcome."
)

_JUDGE_PROMPT = """\
You are evaluating whether an AI agent's planned approach to a task would \
achieve the same outcome as a reference trajectory from a successful agent.

You will be given:
1. **Task**: the task description and available tools
2. **Reference trajectory**: the tool calls and outcome from a successful run
3. **Model output**: the model's planned approach

Score the model output on a scale of 0.0 to 1.0:
- 1.0 = approach would achieve the same outcome, correct tools and ordering
- 0.7 = mostly correct approach, minor differences in tool selection or order
- 0.4 = partially correct, some right ideas but significant gaps
- 0.0 = completely wrong approach or unrelated

Respond with a JSON object: {"score": <float>, "reason": "<brief explanation>"}"""


def _parse_json_field(raw: Any) -> Any:
    """Parse a field that may be a JSON string or already decoded."""
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    return raw or {}


def _extract_tool_descriptions(tool_calls: list) -> str:
    """Build a tool listing from the tool_calls field."""
    if not tool_calls:
        return "No specific tools listed."
    lines = []
    seen: set[str] = set()
    tools = tool_calls if isinstance(tool_calls, list) else _parse_json_field(tool_calls)
    if isinstance(tools, dict):
        tools = tools.get("tools", [])
    for tc in tools:
        name = ""
        desc = ""
        if isinstance(tc, dict):
            func = tc.get("function", tc)
            if isinstance(func, dict):
                name = func.get("name", "")
                desc = func.get("description", "")
            elif isinstance(func, str):
                name = func
        if name and name not in seen:
            seen.add(name)
            if desc:
                lines.append(f"- **{name}**: {desc}")
            else:
                lines.append(f"- **{name}**")
    return "\n".join(lines) if lines else "No specific tools listed."


def _summarize_reference_trajectory(messages: list) -> str:
    """Summarize the reference trajectory for the judge."""
    if not messages:
        return "No reference trajectory available."

    parts = []
    tool_calls_made = []
    final_answer = ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            tc = msg.get("tool_calls", [])
            if tc:
                for call in tc:
                    func = call.get("function", {}) if isinstance(call, dict) else {}
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "")
                    if isinstance(args, str) and len(args) > 200:
                        args = args[:200] + "..."
                    tool_calls_made.append(f"{name}({args})")
            elif content:
                final_answer = content if isinstance(content, str) else str(content)

    if tool_calls_made:
        parts.append("Tools called in order:\n" + "\n".join(
            f"  {i+1}. {tc}" for i, tc in enumerate(tool_calls_made[:20])
        ))
        if len(tool_calls_made) > 20:
            parts.append(f"  ... and {len(tool_calls_made) - 20} more calls")

    if final_answer:
        if len(final_answer) > 500:
            final_answer = final_answer[:500] + "..."
        parts.append(f"\nFinal outcome:\n{final_answer}")

    return "\n".join(parts) if parts else "No meaningful trajectory content."


def _build_prompt(record: dict) -> tuple[str, str, dict]:
    """Build prompt, reference target, and metadata from a dataset record."""
    config = _parse_json_field(record.get("config", {}))
    messages = _parse_json_field(record.get("messages", []))
    tool_calls = _parse_json_field(record.get("tool_calls", []))
    key_stats = _parse_json_field(record.get("key_stats", {}))
    task_status = _parse_json_field(record.get("task_status", {}))
    task_name = record.get("task_name", "")

    # Extract task description
    task_str = ""
    if isinstance(config, dict):
        task_str = config.get("task_str", "")
    if not task_str and isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                task_str = msg.get("content", "")
                break

    # Available tools
    mcp_servers = []
    local_tools = []
    if isinstance(config, dict):
        mcp_servers = config.get("needed_mcp_servers", [])
        local_tools = config.get("needed_local_tools", [])

    tool_listing = _extract_tool_descriptions(tool_calls)
    if mcp_servers or local_tools:
        extra = []
        if mcp_servers:
            extra.append(f"MCP servers: {', '.join(mcp_servers)}")
        if local_tools:
            extra.append(f"Local tools: {', '.join(local_tools)}")
        tool_listing += "\n" + "\n".join(extra)

    prompt = (
        f"## Task: {task_name}\n\n"
        f"{task_str}\n\n"
        f"## Available Tools\n\n"
        f"{tool_listing}\n\n"
        f"Explain your step-by-step approach to accomplish this task. "
        f"Be specific about which tools you would use, in what order, "
        f"and what information you would extract at each step. "
        f"Then provide the expected final answer or outcome."
    )

    ref_summary = _summarize_reference_trajectory(
        messages if isinstance(messages, list) else []
    )

    eval_passed = False
    if isinstance(task_status, dict):
        eval_passed = bool(task_status.get("evaluation", False))

    metadata = {
        "task_name": task_name,
        "eval_passed": eval_passed,
        "num_tool_calls": key_stats.get("tool_calls", 0)
            if isinstance(key_stats, dict) else 0,
        "mcp_servers": mcp_servers,
    }

    return prompt, ref_summary, metadata


def _load_tasks(limit: Optional[int] = None) -> list[dict]:
    """Load tasks from HuggingFace using streaming to avoid PyArrow crash."""
    from datasets import load_dataset

    logger.info(f"Loading {DATASET_ID} (streaming)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    # Deduplicate by task_name — keep first successful trajectory per task
    tasks_by_name: dict[str, dict] = {}
    for row in ds:
        name = row.get("task_name", "")
        status = _parse_json_field(row.get("task_status", {}))
        passed = bool(status.get("evaluation", False)) if isinstance(status, dict) else False

        # Prefer successful trajectories
        if name not in tasks_by_name or (passed and not tasks_by_name[name].get("_passed")):
            row["_passed"] = passed
            tasks_by_name[name] = row

        if limit and len(tasks_by_name) >= limit:
            # Check if we have enough successful ones
            break

    tasks = list(tasks_by_name.values())
    if limit and limit < len(tasks):
        tasks = tasks[:limit]

    logger.info(f"Loaded {len(tasks)} unique tasks ({sum(1 for t in tasks if t.get('_passed'))} successful)")
    return tasks


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Extract score and reason from judge response."""
    import re

    try:
        match = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            score = float(obj.get("score", 0.0))
            reason = obj.get("reason", "")
            return max(0.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    match = re.search(r'(?:score|rating)[:\s]*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        if score > 1.0:
            score = score / 10.0
        return max(0.0, min(1.0, score)), text[:200]

    return 0.0, f"Could not parse judge response: {text[:200]}"


class ToolDecathlonBenchmark(BaseBenchmark):
    """Tool Decathlon: standalone benchmark with streaming HF load + judge eval."""

    REQUIRED_TARGET_TYPES: list[str] = []

    def __init__(self, config: BenchmarkConfig):
        config.backend = "custom_eval"
        super().__init__(config)

    def evaluate(self, target: BaseTarget) -> BenchmarkResult:
        logger.info(f"Evaluating {target.name} on tool_decathlon")

        tasks = _load_tasks(limit=self.config.limit)
        if not tasks:
            raise ValueError("No Tool Decathlon tasks loaded")

        judge_target = self.config.backend_params.get("judge_target")
        detailed_results: list[dict] = []
        correct = 0

        for i, record in enumerate(tasks):
            prompt, ref_summary, meta = _build_prompt(record)
            task_name = meta["task_name"]

            # Get model output
            try:
                request = TargetRequest(
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
                response = target.send_request(request)
                model_output = response.content
            except Exception as e:
                logger.error(f"Inference error for {task_name}: {e}")
                detailed_results.append({
                    "id": f"td-{i:04d}",
                    "input": prompt[:2000],
                    "expected": ref_summary[:2000],
                    "output": "",
                    "raw_output": "",
                    "score": 0.0,
                    "success": False,
                    "reason": f"Inference error: {e}",
                    "metadata": meta,
                })
                continue

            # Judge evaluation
            score = 0.0
            reason = ""
            if judge_target:
                try:
                    judge_input = (
                        f"{_JUDGE_PROMPT}\n\n"
                        f"## Task\n{prompt[:2000]}\n\n"
                        f"## Reference Trajectory\n{ref_summary[:2000]}\n\n"
                        f"## Model Output\n{model_output[:3000]}"
                    )
                    judge_request = TargetRequest(
                        messages=[{"role": "user", "content": judge_input}]
                    )
                    judge_response = judge_target.send_request(judge_request)
                    score, reason = _parse_judge_response(judge_response.content)
                except Exception as e:
                    logger.error(f"Judge error for {task_name}: {e}")
                    reason = f"Judge error: {e}"
            else:
                # No judge — basic heuristic based on tool name overlap
                ref_tools = set()
                messages = _parse_json_field(record.get("messages", []))
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            for tc in msg.get("tool_calls", []):
                                if isinstance(tc, dict):
                                    func = tc.get("function", {})
                                    if isinstance(func, dict):
                                        ref_tools.add(func.get("name", ""))
                if ref_tools:
                    matches = sum(1 for t in ref_tools if t and t in model_output)
                    score = min(1.0, matches / len(ref_tools)) if ref_tools else 0.0
                    reason = f"Tool name overlap: {matches}/{len(ref_tools)}"

            # The shared rule, not a private copy. This adapter builds its
            # own records and never reaches _review_row_to_record, so the
            # backend-level unification missed it: ops classifies this benchmark's metric as `judge`, so it offers a
            # threshold and sends one -- which this rule silently ignored.
            success = row_passed(
                score, self.config.pass_threshold,
                legacy_minimum=LEGACY_JUDGE_MIN, legacy_inclusive=True,
            )
            if success:
                correct += 1

            detailed_results.append({
                "id": f"td-{i:04d}",
                "input": prompt[:2000],
                "expected": ref_summary[:2000],
                "output": model_output[:2000],
                "raw_output": model_output,
                "score": score,
                "success": success,
                "reason": reason,
                "metadata": meta,
            })

            if (i + 1) % 10 == 0:
                logger.info(f"Tool Decathlon progress: {i + 1}/{len(tasks)}")

        total = len(detailed_results)
        # `avg_score` is what this benchmark reports, and it is the mean of
        # the raw judge scores -- a pass threshold moves which rows are
        # labelled passing, never the score. A `correct / total` local used
        # to sit here, computed and never returned, and it read closely
        # enough like the reported score that a reviewer concluded the
        # threshold silently re-scored the benchmark and broke A/B compare.
        # It did not, but the line was worth deleting rather than explaining.
        avg_score = sum(r["score"] for r in detailed_results) / total if total else 0.0

        # Per-MCP-server breakdown
        task_results: Dict[str, Dict] = {}
        for r in detailed_results:
            servers = r.get("metadata", {}).get("mcp_servers", [])
            key = servers[0] if servers else "unknown"
            if key not in task_results:
                task_results[key] = {"n_samples": 0, "passed": 0, "score_sum": 0.0}
            task_results[key]["n_samples"] += 1
            if r["success"]:
                task_results[key]["passed"] += 1
            task_results[key]["score_sum"] += r["score"]
        for key, stats in task_results.items():
            # Threshold-relative, unlike `avg_score` beside it: this is the
            # fraction of rows that cleared `pass_threshold`, so it shifts
            # when the threshold does. Nothing in ops reads `task_results`
            # today; if that changes, compare it only against runs scored
            # under the same threshold (recorded in `metadata` below).
            stats["accuracy"] = stats["passed"] / stats["n_samples"] if stats["n_samples"] else 0.0
            stats["avg_score"] = stats["score_sum"] / stats["n_samples"] if stats["n_samples"] else 0.0

        # Print summary table
        logger.info(f"{'MCP Server':<20} {'Tasks':>6} {'Passed':>8} {'Avg Score':>10}")
        logger.info("-" * 48)
        for key, stats in task_results.items():
            logger.info(
                f"{key:<20} {stats['n_samples']:>6} "
                f"{stats['passed']:>8} {stats['avg_score']:>10.4f}"
            )
        logger.info("-" * 48)
        logger.info(
            f"{'OVERALL':<20} {total:>6} "
            f"{correct:>8} {avg_score:>10.4f}"
        )

        return BenchmarkResult(
            benchmark_name="tool_decathlon",
            overall_score=avg_score,
            num_samples=total,
            backend="tool_decathlon",
            task_results=task_results,
            detailed_results=detailed_results,
            metadata={
                "dataset": DATASET_ID,
                "has_judge": judge_target is not None,
                "status": "completed",
                # Which rule decided the per-row verdicts. Not recorded
                # anywhere else, so without it an old run's Pass/Fail column
                # cannot be read back -- 6.5 out of 10 is a pass at 5.0 and a
                # failure at 8.0, and the stored row says only "failed".
                "pass_threshold": self.config.pass_threshold,
            },
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "tool_decathlon",
            "pretty_name": "Tool Decathlon",
            "description": "Diverse multi-step tool-use evaluation across 600+ tools and MCP servers",
            "dataset_id": DATASET_ID,
            "total_tasks": 108,
            "required_target_types": [],
        }


# Register with the benchmark registry
BenchmarkRegistry._benchmarks["tool_decathlon"] = ToolDecathlonBenchmark
