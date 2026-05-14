"""VITA-Bench benchmark — visual-interactive task automation.

meituan-longcat/VitaBench evaluates multi-step agent task planning across
delivery, in-store, OTA, and cross-domain scenarios. Each task provides
environment state, user scenario, and instructions; the model must produce
a plan of actions that satisfies the expected states in evaluation_criteria.

The dataset is non-standard JSON files on ModelScope (not HF-format parquet),
so we bypass evalscope's dataset loader entirely and implement a standalone
benchmark that loads the JSON directly, prompts the model, and uses an LLM
judge to compare planned actions against expected states.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from surogate_eval.benchmarks.base import BaseBenchmark, BenchmarkConfig, BenchmarkResult
from surogate_eval.benchmarks.registry import BenchmarkRegistry
from surogate_eval.targets import BaseTarget
from surogate_eval.targets.base import TargetRequest
from surogate_eval.utils.logger import get_logger

logger = get_logger()

DATASET_ID = "meituan-longcat/VitaBench"
DOMAINS = ["delivery", "instore", "ota", "cross_domain"]

SYSTEM_PROMPT = """\
You are an intelligent task-planning assistant for a mobile app platform \
(food delivery, hotel/travel booking, in-store services). Given a user's \
environment state, profile, and instructions, produce a detailed action plan \
that fulfils the user's request.

Your plan must specify concrete actions: which services to use, what to \
order, booking details, times, addresses, and any constraints the user \
mentioned. Output your plan as a JSON object with an "actions" array, where \
each action has "type", "service", and "details" fields."""

JUDGE_PROMPT = """\
You are evaluating whether an AI assistant's action plan correctly fulfils a \
user's task. You will be given:

1. **Instructions**: what the user asked for
2. **Expected states**: the ground-truth actions/orders that should result
3. **Model output**: the assistant's planned actions

Score the model output on a scale of 0.0 to 1.0:
- 1.0 = all expected orders/actions are present with correct details
- 0.5 = partial match — some actions correct, some missing or wrong
- 0.0 = completely wrong or unrelated

Consider: correct service/store selection, order contents, timing constraints, \
address accuracy, and any special requirements mentioned in the instructions.

Respond with a JSON object: {"score": <float>, "reason": "<brief explanation>"}"""


def _download_dataset() -> Path:
    """Download VitaBench JSON files via ModelScope snapshot_download."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        old_domain = os.environ.get("MODELSCOPE_DOMAIN")
        os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.cn"
        try:
            path = snapshot_download(DATASET_ID, repo_type="dataset")
        finally:
            if old_domain is not None:
                os.environ["MODELSCOPE_DOMAIN"] = old_domain
            else:
                os.environ.pop("MODELSCOPE_DOMAIN", None)

        return Path(path)
    except ImportError:
        raise ImportError(
            "modelscope is required for VitaBench. "
            "Install with: pip install modelscope"
        )


def _load_tasks(dataset_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load tasks from all domain JSON files (English versions)."""
    tasks: list[dict] = []
    for domain in DOMAINS:
        # Prefer English version
        en_path = dataset_dir / domain / "tasks_en.json"
        zh_path = dataset_dir / domain / "tasks.json"
        path = en_path if en_path.exists() else zh_path

        if not path.exists():
            logger.warning(f"VitaBench: missing {path}, skipping domain '{domain}'")
            continue

        with open(path) as f:
            domain_tasks = json.load(f)

        for t in domain_tasks:
            t["_domain"] = domain

        tasks.extend(domain_tasks)
        logger.info(f"VitaBench: loaded {len(domain_tasks)} tasks from {domain}")

    if limit and limit < len(tasks):
        tasks = tasks[:limit]
        logger.info(f"VitaBench: limited to {limit} tasks")

    return tasks


def _format_prompt(task: Dict[str, Any]) -> str:
    """Build the user prompt from a VitaBench task."""
    parts = []

    instructions = task.get("instructions", "")
    parts.append(f"## User Instructions\n{instructions}")

    env = task.get("environment", {})
    if env:
        # Serialise environment but truncate very large nested objects
        env_str = json.dumps(env, ensure_ascii=False, default=str)
        if len(env_str) > 3000:
            env_str = env_str[:3000] + "..."
        parts.append(f"## Environment State\n```json\n{env_str}\n```")

    scenario = task.get("user_scenario", {})
    profile = scenario.get("user_profile", {})
    if profile:
        profile_str = json.dumps(profile, ensure_ascii=False, default=str)
        if len(profile_str) > 1500:
            profile_str = profile_str[:1500] + "..."
        parts.append(f"## User Profile\n```json\n{profile_str}\n```")

    history = task.get("message_history", [])
    if history:
        parts.append(f"## Message History\n{json.dumps(history, ensure_ascii=False, default=str)[:2000]}")

    return "\n\n".join(parts)


def _format_expected(task: Dict[str, Any]) -> str:
    """Serialise expected states for the judge."""
    criteria = task.get("evaluation_criteria", {})
    return json.dumps(criteria, ensure_ascii=False, default=str)[:4000]


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Extract score and reason from judge response."""
    import re

    # Try JSON parse first
    try:
        # Find JSON object in response
        match = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            score = float(obj.get("score", 0.0))
            reason = obj.get("reason", "")
            return max(0.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: look for a number
    match = re.search(r'(?:score|rating)[:\s]*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        if score > 1.0:
            score = score / 10.0  # normalise 0-10 → 0-1
        return max(0.0, min(1.0, score)), text[:200]

    return 0.0, f"Could not parse judge response: {text[:200]}"


class VitaBenchmark(BaseBenchmark):
    """VITA-Bench: standalone benchmark with direct JSON loading + judge eval."""

    REQUIRED_TARGET_TYPES: list[str] = []

    def __init__(self, config: BenchmarkConfig):
        # Override backend to avoid evalscope
        config.backend = "custom_eval"
        super().__init__(config)

    def evaluate(self, target: BaseTarget) -> BenchmarkResult:
        logger.info(f"Evaluating {target.name} on vita_bench")

        # 1. Download dataset
        dataset_dir = _download_dataset()
        logger.info(f"VitaBench dataset at: {dataset_dir}")

        # 2. Load tasks
        tasks = _load_tasks(dataset_dir, limit=self.config.limit)
        if not tasks:
            raise ValueError("No VitaBench tasks loaded")

        logger.info(f"Loaded {len(tasks)} VitaBench tasks")

        # 3. Get judge target
        judge_target = self.config.backend_params.get("judge_target")

        # 4. Evaluate each task
        detailed_results: list[dict] = []
        correct = 0

        for i, task in enumerate(tasks):
            task_id = task.get("id", f"vita-{i:04d}")
            domain = task.get("_domain", "unknown")
            prompt = _format_prompt(task)
            expected = _format_expected(task)

            # Get model output
            try:
                request = TargetRequest(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
                response = target.send_request(request)
                model_output = response.content
            except Exception as e:
                logger.error(f"Inference error for task {task_id}: {e}")
                detailed_results.append({
                    "id": task_id,
                    "domain": domain,
                    "input": prompt[:2000],
                    "expected": expected[:2000],
                    "output": "",
                    "raw_output": "",
                    "score": 0.0,
                    "success": False,
                    "reason": f"Inference error: {e}",
                })
                continue

            # Judge evaluation
            score = 0.0
            reason = ""
            if judge_target:
                try:
                    judge_prompt = (
                        f"{JUDGE_PROMPT}\n\n"
                        f"## Instructions\n{task.get('instructions', '')}\n\n"
                        f"## Expected States\n```json\n{expected}\n```\n\n"
                        f"## Model Output\n{model_output[:3000]}"
                    )
                    judge_request = TargetRequest(
                        messages=[{"role": "user", "content": judge_prompt}]
                    )
                    judge_response = judge_target.send_request(judge_request)
                    score, reason = _parse_judge_response(judge_response.content)
                except Exception as e:
                    logger.error(f"Judge error for task {task_id}: {e}")
                    reason = f"Judge error: {e}"
            else:
                # No judge — do basic keyword matching
                expected_states = task.get("evaluation_criteria", {}).get("expected_states", [])
                if expected_states:
                    matches = 0
                    for state in expected_states:
                        orders = state.get("required_orders", [])
                        for order in orders:
                            store_id = order.get("store_id", "")
                            if store_id and store_id in model_output:
                                matches += 1
                    total_orders = sum(
                        len(s.get("required_orders", []))
                        for s in expected_states
                    )
                    score = matches / total_orders if total_orders else 0.0
                    reason = f"Keyword match: {matches}/{total_orders} orders"

            success = score >= 0.5
            if success:
                correct += 1

            detailed_results.append({
                "id": task_id,
                "domain": domain,
                "input": prompt[:2000],
                "expected": expected[:2000],
                "output": model_output[:2000],
                "raw_output": model_output,
                "score": score,
                "success": success,
                "reason": reason,
            })

            if (i + 1) % 10 == 0:
                logger.info(f"VitaBench progress: {i + 1}/{len(tasks)}")

        # 5. Compute metrics
        total = len(detailed_results)
        overall_score = correct / total if total else 0.0
        avg_score = (
            sum(r["score"] for r in detailed_results) / total if total else 0.0
        )

        # Per-domain breakdown
        domain_results: dict[str, dict] = {}
        for r in detailed_results:
            d = r["domain"]
            if d not in domain_results:
                domain_results[d] = {"total": 0, "correct": 0, "score_sum": 0.0}
            domain_results[d]["total"] += 1
            domain_results[d]["correct"] += 1 if r["success"] else 0
            domain_results[d]["score_sum"] += r["score"]

        task_results = {}
        for d, stats in domain_results.items():
            task_results[d] = {
                "n_samples": stats["total"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
                "avg_score": stats["score_sum"] / stats["total"] if stats["total"] else 0.0,
            }

        # Print summary table
        logger.info(f"{'Domain':<16} {'Samples':>8} {'Accuracy':>10} {'Avg Score':>10}")
        logger.info("-" * 48)
        for d, stats in task_results.items():
            logger.info(
                f"{d:<16} {stats['n_samples']:>8} "
                f"{stats['accuracy']:>9.1%} {stats['avg_score']:>10.4f}"
            )
        logger.info("-" * 48)
        logger.info(
            f"{'OVERALL':<16} {total:>8} "
            f"{overall_score:>9.1%} {avg_score:>10.4f}"
        )

        return BenchmarkResult(
            benchmark_name="vita_bench",
            overall_score=avg_score,
            num_samples=total,
            backend="vita_bench",
            task_results=task_results,
            detailed_results=detailed_results,
            metadata={
                "dataset": DATASET_ID,
                "domains": DOMAINS,
                "has_judge": judge_target is not None,
                "status": "completed",
            },
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "vita_bench",
            "pretty_name": "VITA-Bench",
            "description": "Visual-interactive task automation benchmark",
            "dataset_id": DATASET_ID,
            "domains": DOMAINS,
            "total_tasks": 400,
            "required_target_types": [],
        }


# Register with the benchmark registry
BenchmarkRegistry._benchmarks["vita_bench"] = VitaBenchmark
