"""MCPMark benchmark — live MCP tool-use evaluation.

MCPMark (eval-sys/mcpmark) is a comprehensive MCP benchmark that tests
model capabilities across 5 real MCP services: filesystem, notion,
github, postgres, and playwright.

Each task has:
  - description.md: task instruction
  - meta.json: setup/config metadata
  - verify.py: deterministic verification script

This adapter clones the mcpmark repo, runs its pipeline against the
target model's API endpoint, and collects pass/fail results per task.
The model interacts with real MCP servers — no trajectory replay.

Results use pass@1 scoring (binary pass/fail per task).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from surogate_eval.benchmarks.base import BaseBenchmark, BenchmarkConfig, BenchmarkResult
from surogate_eval.benchmarks.registry import BenchmarkRegistry
from surogate_eval.targets import BaseTarget
from surogate_eval.utils.logger import get_logger

logger = get_logger()

MCPMARK_REPO = "https://github.com/eval-sys/mcpmark.git"
MCP_SERVICES = ["filesystem", "notion", "github", "postgres", "playwright"]

# Services that can run without external credentials
SELF_CONTAINED_SERVICES = {"filesystem"}

# Services that need external API keys / credentials
CREDENTIALED_SERVICES = {
    "notion": "NOTION_TOKEN",
    "github": "GITHUB_TOKEN",
    "postgres": None,  # uses local postgres
    "playwright": None,  # uses local browser
}


def _clone_repo(target_dir: Path) -> Path:
    """Clone mcpmark repo into target directory."""
    repo_dir = target_dir / "mcpmark"
    if repo_dir.exists():
        logger.info(f"MCPMark repo already exists at {repo_dir}")
        return repo_dir

    logger.info(f"Cloning MCPMark from {MCPMARK_REPO}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", MCPMARK_REPO, str(repo_dir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone MCPMark: {result.stderr}")

    # Install mcpmark package
    logger.info("Installing MCPMark dependencies")
    result = subprocess.run(
        ["pip", "install", "-e", str(repo_dir)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        logger.warning(f"MCPMark install warning: {result.stderr[:500]}")

    return repo_dir


def _get_available_services() -> List[str]:
    """Determine which MCP services can run based on available credentials."""
    available = list(SELF_CONTAINED_SERVICES)

    for service, env_var in CREDENTIALED_SERVICES.items():
        if env_var is None:
            # Service runs locally (postgres, playwright)
            available.append(service)
        elif os.environ.get(env_var):
            available.append(service)
        else:
            logger.info(f"Skipping MCP service '{service}' — {env_var} not set")

    return available


def _run_mcpmark_pipeline(
    repo_dir: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    services: List[str],
    task_suite: str = "standard",
    timeout: int = 600,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run MCPMark pipeline and collect results."""
    all_results: Dict[str, List[Dict]] = {}

    env = os.environ.copy()
    # MCPMark uses LiteLLM — set OpenAI-compatible endpoint as fallback
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = base_url
    env["PYTHONPATH"] = str(repo_dir)

    results_dir = repo_dir / "results" / "surogate-eval"

    for service in services:
        logger.info(f"Running MCPMark on service: {service}")

        cmd = [
            "python3", str(repo_dir / "pipeline.py"),
            "--models", model_name,
            "--mcp", service,
            "--task-suite", task_suite,
            "--tasks", "all",
            "--exp-name", "surogate-eval",
            "--timeout", str(timeout),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout * 2,  # give extra time for all tasks
                env=env, cwd=str(repo_dir),
            )

            if result.returncode != 0:
                logger.error(
                    f"MCPMark pipeline failed for {service}: "
                    f"{result.stderr[:500]}"
                )
                # Continue with other services
                continue

            logger.info(f"MCPMark {service} stdout: {result.stdout[-500:]}")

        except subprocess.TimeoutExpired:
            logger.error(f"MCPMark {service} timed out after {timeout * 2}s")
            continue

        # Parse results from output directory
        service_results = _parse_service_results(
            results_dir, model_name, service, limit,
        )
        if service_results:
            all_results[service] = service_results

    return all_results


def _parse_service_results(
    results_dir: Path,
    model_name: str,
    service: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Parse MCPMark results for a service from the results directory."""
    results = []

    # MCPMark saves results in results/<exp-name>/run-1/<model>/<service>/
    # Each task gets a directory with meta.json containing pass/fail
    search_paths = [
        results_dir,
        results_dir / "run-1",
    ]

    for base in search_paths:
        for root, dirs, files in os.walk(base):
            if "meta.json" in files:
                meta_path = Path(root) / "meta.json"
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)

                    task_name = meta.get("task", Path(root).name)
                    passed = meta.get("passed", False)
                    error = meta.get("error", "")
                    execution_log = meta.get("execution_log", "")

                    results.append({
                        "id": f"{service}/{task_name}",
                        "service": service,
                        "task": task_name,
                        "input": meta.get("instruction", task_name),
                        "expected": "Task completion verified by verify.py",
                        "output": execution_log[:2000] if execution_log else "",
                        "score": 1.0 if passed else 0.0,
                        "success": passed,
                        "reason": error if not passed else "passed",
                    })
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to parse {meta_path}: {e}")

    if limit and len(results) > limit:
        results = results[:limit]

    return results


class MCPMarkBenchmark(BaseBenchmark):
    """MCPMark: live MCP tool-use evaluation across 5 services."""

    REQUIRED_TARGET_TYPES: list[str] = []

    def __init__(self, config: BenchmarkConfig):
        config.backend = "custom_eval"
        super().__init__(config)

    def evaluate(self, target: BaseTarget) -> BenchmarkResult:
        logger.info(f"Evaluating {target.name} on mcp_mark")

        # Get target API details
        base_url = getattr(target, 'base_url', '') or os.environ.get('OPENAI_BASE_URL', '')
        api_key = getattr(target, 'api_key', '') or os.environ.get('OPENAI_API_KEY', '')
        model_name = getattr(target, 'model', '') or getattr(target, 'name', 'unknown')

        # Determine which services to run — check env var first (set by ops),
        # then backend_params, then auto-detect from credentials.
        bp = self.config.backend_params or {}
        env_services = os.environ.get("MCPMARK_SERVICES", "")
        requested_services = bp.get('services', None)
        if env_services:
            services = [s.strip() for s in env_services.split(',') if s.strip()]
        elif requested_services:
            if isinstance(requested_services, str):
                requested_services = [s.strip() for s in requested_services.split(',')]
            services = requested_services
        else:
            services = _get_available_services()

        task_suite = bp.get('task_suite', 'standard')
        task_timeout = bp.get('task_timeout', 600)

        logger.info(f"MCPMark services: {services}")
        logger.info(f"MCPMark task suite: {task_suite}")

        # Clone and set up MCPMark
        work_dir = Path(tempfile.mkdtemp(prefix="mcpmark_"))
        try:
            repo_dir = _clone_repo(work_dir)

            # Run the pipeline
            all_results = _run_mcpmark_pipeline(
                repo_dir=repo_dir,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                services=services,
                task_suite=task_suite,
                timeout=task_timeout,
                limit=self.config.limit,
            )

            # Flatten results
            detailed_results: List[Dict] = []
            for service, tasks in all_results.items():
                detailed_results.extend(tasks)

            # Compute metrics
            total = len(detailed_results)
            correct = sum(1 for r in detailed_results if r["success"])
            overall_score = correct / total if total else 0.0

            # Per-service breakdown
            task_results: Dict[str, Dict] = {}
            for service in services:
                service_tasks = [r for r in detailed_results if r["service"] == service]
                if service_tasks:
                    s_total = len(service_tasks)
                    s_correct = sum(1 for r in service_tasks if r["success"])
                    task_results[service] = {
                        "n_samples": s_total,
                        "accuracy": s_correct / s_total if s_total else 0.0,
                        "passed": s_correct,
                    }

            # Print summary table
            logger.info(f"{'Service':<16} {'Tasks':>8} {'Passed':>8} {'Pass@1':>10}")
            logger.info("-" * 46)
            for svc, stats in task_results.items():
                logger.info(
                    f"{svc:<16} {stats['n_samples']:>8} "
                    f"{stats['passed']:>8} {stats['accuracy']:>9.1%}"
                )
            logger.info("-" * 46)
            logger.info(
                f"{'OVERALL':<16} {total:>8} "
                f"{correct:>8} {overall_score:>9.1%}"
            )

            return BenchmarkResult(
                benchmark_name="mcp_mark",
                overall_score=overall_score,
                num_samples=total,
                backend="mcp_mark",
                task_results=task_results,
                detailed_results=detailed_results,
                metadata={
                    "services": services,
                    "task_suite": task_suite,
                    "status": "completed",
                },
            )

        finally:
            # Cleanup
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "mcp_mark",
            "pretty_name": "MCPMark",
            "description": "Live MCP tool-use evaluation across 5 services",
            "services": MCP_SERVICES,
            "total_tasks": 177,  # 127 standard + 50 easy
            "required_target_types": [],
        }


# Register with the benchmark registry
BenchmarkRegistry._benchmarks["mcp_mark"] = MCPMarkBenchmark
