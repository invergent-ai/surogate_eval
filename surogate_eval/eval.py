# surogate_eval/eval.py
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from surogate_eval.backend import LocalBackend
from surogate_eval.config.eval_config import TargetConfig
from surogate_eval.outcome import (
    DEFAULT_MAX_ERROR_RATE,
    REQUESTED_WORK_KEY,
    SUPPORT_TARGET_KEY,
    compute_outcome,
    exit_code_for,
)
from surogate_eval.runners import (
    _write_progress,
    run_benchmarks,
    run_evaluation,
    run_guardrails_testing_async,
    run_red_teaming_async,
    run_stress_testing,
)
from surogate_eval.targets import BaseTarget, TargetFactory
from surogate_eval.utils.command import SurogateCommand
from surogate_eval.utils.logger import get_logger

logger = get_logger()

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_FILE_SYSTEM"] = "READ_ONLY"
os.environ["EVALSCOPE_CACHE"] = os.path.join(os.path.expanduser("~"), ".cache", "evalscope")
os.environ["MODELSCOPE_TRUST_REMOTE_CODE"] = "1"

#: Work kinds dispatched by the shared benchmark loop in
#: ``_run_target_evaluations`` - one progress entry each. Metric evaluations
#: and stress testing are planned the same way but dispatched on their own.
BENCHMARK_LOOP_KINDS = ("benchmark", "red_teaming", "guardrails")


class SurogateEval(SurogateCommand):
    def __init__(self, *, config, args):
        super().__init__(config=config, args=args)

        self.consolidated_results = {
            "project": {},
            "timestamp": None,
            "summary": {
                "total_targets": 0,
                "total_evaluations": 0,
                "total_test_cases": 0,
            },
            "targets": [],
        }
        self.targets: List[BaseTarget] = []
        #: Targets another target names as its judge, simulator or
        #: evaluator. Recorded on each target entry so ``outcome.py`` can
        #: tell a support target's expected silence from a target whose
        #: config was never read.
        self.support_target_names: set = set()

    def run(self) -> int:
        """Run the evaluation pipeline. Returns a process exit code."""
        from datetime import datetime

        logger.banner("SUROGATE EVAL")

        self.consolidated_results["timestamp"] = datetime.now().isoformat()
        self.consolidated_results["project"] = {
            "name": self.config.project.name,
            "version": self.config.project.version,
            "description": self.config.project.description,
        }

        try:
            self._process_targets()
        finally:
            self._cleanup()

        configured = self.config.max_error_rate
        outcome = compute_outcome(
            self.consolidated_results,
            DEFAULT_MAX_ERROR_RATE if configured is None else float(configured),
        )
        self.consolidated_results["outcome"] = outcome

        self._save_consolidated_results()

        if outcome["status"] == "completed":
            logger.success("Surogate Eval completed")
        else:
            logger.error(f"Surogate Eval failed: {outcome['reason']}")

        return exit_code_for(outcome)

    def _process_targets(self):
        """Process all targets from config."""
        target_configs = self.config.targets

        if not target_configs:
            logger.warning("No targets specified in configuration")
            return

        logger.info(f"Processing {len(target_configs)} target(s)")
        self.consolidated_results["summary"]["total_targets"] = len(target_configs)
        self.support_target_names = self.config.support_target_names()

        # PHASE 1: Create all targets
        logger.info("Creating all targets...")
        for target_config in target_configs:
            target_name = target_config.name or "unnamed"
            try:
                logger.info(f"Creating target: {target_name}")
                target_dict = target_config.to_dict()
                target = TargetFactory.create_target(target_dict)

                if not target.health_check():
                    logger.error(f"Target '{target_name}' health check failed")
                    self.consolidated_results["targets"].append({
                        "name": target_name,
                        "status": "unhealthy",
                        "evaluations": [],
                    })
                    continue

                logger.success(f"Target '{target_name}' is healthy")
                self.targets.append(target)

            except Exception as e:
                logger.error(f"Failed to create target '{target_name}': {e}")
                self.consolidated_results["targets"].append({
                    "name": target_name,
                    "status": "failed",
                    "error": str(e),
                    "evaluations": [],
                })

        # PHASE 2: Run evaluations
        logger.info("Running evaluations on all targets...")
        for target_config in target_configs:
            target_name = target_config.name or "unnamed"
            target = self._find_target_by_name(target_name)

            if not target:
                logger.warning(f"Skipping evaluations for '{target_name}' (target not healthy)")
                continue

            logger.separator(char="═")
            logger.header(f"Target: {target_name}")
            logger.separator(char="═")

            try:
                target_results = self._run_target_evaluations(target, target_config)
                if target_results:
                    self._record_target_result(target_results)

            except Exception as e:
                logger.error(f"Failed to run evaluations for target '{target_name}': {e}")
                import traceback
                traceback.print_exc()
                # Record the crash. Without an entry the target is invisible
                # both in the results file and to the outcome computation, so
                # a run with another healthy target still reported success.
                self._record_target_result({
                    "name": target_name,
                    "status": "failed",
                    "error": str(e),
                    "evaluations": [],
                })

    def _record_target_result(self, target_results: Dict[str, Any]) -> None:
        """Insert or replace the consolidated entry for a target."""
        target_name = target_results.get("name")
        existing_idx = next(
            (i for i, t in enumerate(self.consolidated_results["targets"])
             if t.get("name") == target_name),
            None,
        )
        if existing_idx is not None:
            self.consolidated_results["targets"][existing_idx] = target_results
        else:
            self.consolidated_results["targets"].append(target_results)

    @staticmethod
    def _plan_work(target_config: TargetConfig) -> List[Tuple[str, Any]]:
        """Everything this target's config asks us to run.

        The single answer to "what was this target asked to do?". A target
        that plans no work has been asked for nothing; paired with whether
        another target names it as a judge or simulator, that is how
        ``outcome.py`` tells a support target's expected silence from a
        target whose sections were never read.

        ``_run_target_evaluations`` dispatches from this list and records it
        verbatim on the target entry, so ``outcome.py`` decides on a declared
        fact instead of guessing intent from the results that came back. Any
        new kind of work has to be planned here to run at all, which is what
        keeps the record honest without anyone remembering to update it: the
        list that runs and the list we claim to have asked for are one list.

        The order here is the order the kinds are declared, not the order
        they are dispatched in: the runner picks the entries of one kind at
        a time and runs metric evaluations, then stress testing, then the
        benchmark/red-team/guardrails loop. Order within one kind is
        preserved.
        """
        plan: List[Tuple[str, Any]] = []

        evaluations = target_config.evaluations or []
        for eval_config in evaluations:
            plan.append(("evaluation", eval_config))
        for eval_config in evaluations:
            for bench_config in eval_config.get("benchmarks", []):
                plan.append(("benchmark", bench_config))

        red_teaming = target_config.red_teaming or {}
        if red_teaming.get("enabled"):
            plan.append(("red_teaming", red_teaming))

        guardrails_cfg = target_config.guardrails or {}
        if guardrails_cfg.get("enabled"):
            plan.append(("guardrails", guardrails_cfg))

        stress_testing = target_config.stress_testing or {}
        if stress_testing.get("enabled"):
            plan.append(("stress_testing", stress_testing))

        return plan

    def _run_target_evaluations(self, target: BaseTarget, target_config: TargetConfig) -> Dict[str, Any]:
        """Run all evaluations for a single target."""
        target_name = target.name

        work = self._plan_work(target_config)

        target_result = {
            "name": target_name,
            "type": target.target_type.value,
            "model": target.config.get("model", "unknown"),
            "provider": target.config.get("provider", "unknown"),
            "status": "success",
            # What we asked this target for, taken from the plan dispatched
            # below rather than restated, and whether anyone else named this
            # target as their judge/simulator/evaluator. Together they let
            # outcome.py tell a support target's expected silence from a
            # target whose sections were never read - a misspelt
            # ``evaluations:`` plans nothing either.
            REQUESTED_WORK_KEY: [kind for kind, _ in work],
            SUPPORT_TARGET_KEY: target_name in self.support_target_names,
            "evaluations": [],
        }

        backend = self._setup_target_backend(target_config)

        # Run evaluations
        evaluations = [cfg for kind, cfg in work if kind == "evaluation"]
        if evaluations:
            logger.info(f"Running {len(evaluations)} evaluation(s) for target '{target_name}'")
            self.consolidated_results["summary"]["total_evaluations"] += len(evaluations)

            for eval_config in evaluations:
                eval_result = run_evaluation(target, eval_config, self._find_target_by_name, backend)
                if eval_result:
                    target_result["evaluations"].append(eval_result)
                    self.consolidated_results["summary"]["total_test_cases"] += eval_result.get("num_test_cases", 0)
        else:
            logger.warning(f"No evaluations specified for target '{target_name}'")

        # Standard benchmarks + security tests all go through the same loop
        # with consistent progress tracking and per-benchmark result file
        # writing.
        tasks = [(kind, cfg) for kind, cfg in work if kind in BENCHMARK_LOOP_KINDS]

        # Run stress testing (separate — not a scored benchmark)
        for stress_config in [cfg for kind, cfg in work if kind == "stress_testing"]:
            logger.info(f"Running stress testing for target '{target_name}'")
            stress_result = run_stress_testing(target, stress_config)
            if stress_result:
                target_result["stress_testing"] = stress_result

        # Execute all tasks in one loop
        from surogate_eval.runners import _run_single_benchmark, _write_bench_result

        total = len(tasks)
        if total:
            logger.separator(char="─")
            logger.header(f"Running {total} Benchmark(s)")
            logger.separator(char="─")

        for idx, (task_type, task_config) in enumerate(tasks):
            if task_type == "benchmark":
                name = task_config.get("name", "unknown")
                _write_progress(name, idx, total)
                result = _run_single_benchmark(target, task_config, self._find_target_by_name)
                if result:
                    if "benchmarks" not in target_result:
                        target_result["benchmarks"] = []
                    target_result["benchmarks"].append(result)
                    _write_bench_result(result)

            elif task_type == "red_teaming":
                _write_progress("red_teaming", idx, total)
                logger.info(f"Running red teaming for target '{target_name}'")
                rt_result = asyncio.run(
                    run_red_teaming_async(target, task_config, self._find_target_by_name)
                )
                target_result["red_teaming"] = rt_result
                _write_bench_result({"benchmark_name": "red_teaming", **rt_result})

            elif task_type == "guardrails":
                _write_progress("guardrails", idx, total)
                logger.info(f"Testing guardrails for target '{target_name}'")
                gr_result = asyncio.run(
                    run_guardrails_testing_async(target, task_config, self._find_target_by_name)
                )
                target_result["guardrails"] = gr_result
                _write_bench_result({"benchmark_name": "guardrails", **gr_result})

        # Not a benchmark switch -- the run is ending, so the row block must
        # keep the last benchmark's real final counts rather than zero them.
        _write_progress("done", total, total, clear_rows=False)

        if backend:
            backend.shutdown()

        return target_result

    def _setup_target_backend(self, target_config: TargetConfig) -> Any:
        """Setup execution backend for a target."""
        infra_config = target_config.infrastructure or {}

        if not infra_config:
            logger.debug("No infrastructure config - using default")
            return None

        backend_type = infra_config.get("backend", "local")

        if backend_type == "local":
            backend = LocalBackend(infra_config)
            logger.success(f"Local backend initialized with {infra_config.get('workers', 1)} workers")
            return backend
        else:
            raise NotImplementedError(f"Backend '{backend_type}' not implemented yet")

    def _find_target_by_name(self, name: str) -> BaseTarget:
        """Find a target by name from created targets."""
        for target in self.targets:
            if target.name == name:
                return target
        return None

    def _save_consolidated_results(self):
        """Save consolidated results to a single file."""
        try:
            import json
            from datetime import datetime
            from enum import Enum

            def convert_enum_keys(obj):
                if isinstance(obj, dict):
                    return {
                        (k.value if isinstance(k, Enum) else k): convert_enum_keys(v)
                        for k, v in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_enum_keys(item) for item in obj]
                elif isinstance(obj, Enum):
                    return obj.value
                return obj

            def custom_encoder(obj):
                if isinstance(obj, Enum):
                    return obj.value
                if isinstance(obj, dict):
                    return {(k.value if isinstance(k, Enum) else k): v for k, v in obj.items()}
                return str(obj)

            serializable_results = convert_enum_keys(self.consolidated_results)

            results_dir = Path("eval_results")
            results_dir.mkdir(exist_ok=True)

            job_id = os.environ.get("EVAL_JOB_ID") or os.environ.get("TASK_RUN_ID")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_id = job_id or timestamp

            filepath = results_dir / f"eval_{file_id}.json"

            with open(filepath, "w") as f:
                json.dump(serializable_results, f, indent=2, default=custom_encoder)

            logger.separator(char="═")
            logger.success(f"Consolidated results saved to: {filepath}")
            logger.separator(char="═")

            self._create_summary_report(serializable_results, results_dir, file_id)

        except Exception as e:
            logger.error(f"Failed to save consolidated results: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _create_summary_report(self, results: Dict[str, Any], results_dir: Path, timestamp: str):
        """Create human-readable summary reports (MD and PDF)."""
        try:
            from surogate_eval.report import ReportGenerator

            generator = ReportGenerator()

            # Generate markdown
            md_file = results_dir / f"report_{timestamp}.md"
            generator.save_markdown(results, md_file)

            # Generate PDF
            pdf_file = results_dir / f"report_{timestamp}.pdf"
            try:
                generator.save_pdf(results, pdf_file)
            except ImportError:
                logger.warning("PDF generation skipped - weasyprint not installed")
            except Exception as e:
                logger.error(f"Failed to generate PDF report: {e}")

        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _cleanup(self):
        """Cleanup all resources."""
        logger.info("Cleaning up resources")

        for target in self.targets:
            try:
                target.cleanup()
                logger.debug(f"Cleaned up target: {target.name}")
            except Exception as e:
                logger.error(f"Error cleaning up target {target.name}: {e}")

        logger.success("Cleanup complete")

    def get_results(self) -> Dict[str, Any]:
        """Get consolidated evaluation results."""
        return self.consolidated_results

    def get_targets(self) -> List[BaseTarget]:
        """Get configured targets."""
        return self.targets