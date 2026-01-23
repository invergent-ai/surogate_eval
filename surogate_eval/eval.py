# surogate_eval/eval.py
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

from surogate_eval.backend import LocalBackend
from surogate_eval.config.eval_config import TargetConfig
from surogate_eval.runners import (
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

    def run(self):
        """Run the evaluation pipeline."""
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

        self._save_consolidated_results()
        logger.success("Surogate Eval completed")

    def _process_targets(self):
        """Process all targets from config."""
        target_configs = self.config.targets

        if not target_configs:
            logger.warning("No targets specified in configuration")
            return

        logger.info(f"Processing {len(target_configs)} target(s)")
        self.consolidated_results["summary"]["total_targets"] = len(target_configs)

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
                    existing_idx = next(
                        (i for i, t in enumerate(self.consolidated_results["targets"])
                         if t.get("name") == target_name),
                        None,
                    )
                    if existing_idx is not None:
                        self.consolidated_results["targets"][existing_idx] = target_results
                    else:
                        self.consolidated_results["targets"].append(target_results)

            except Exception as e:
                logger.error(f"Failed to run evaluations for target '{target_name}': {e}")
                import traceback
                traceback.print_exc()

    def _run_target_evaluations(self, target: BaseTarget, target_config: TargetConfig) -> Dict[str, Any]:
        """Run all evaluations for a single target."""
        target_name = target.name

        target_result = {
            "name": target_name,
            "type": target.target_type.value,
            "model": target.config.get("model", "unknown"),
            "provider": target.config.get("provider", "unknown"),
            "status": "success",
            "evaluations": [],
        }

        backend = self._setup_target_backend(target_config)

        # Run evaluations
        evaluations = target_config.evaluations or []
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

        # Run benchmarks
        for eval_config in evaluations:
            benchmarks_config = eval_config.get("benchmarks", [])
            if benchmarks_config:
                logger.info(f"Running {len(benchmarks_config)} benchmark(s)")
                benchmark_results = run_benchmarks(target, benchmarks_config, self._find_target_by_name)
                if benchmark_results:
                    if "benchmarks" not in target_result:
                        target_result["benchmarks"] = []
                    target_result["benchmarks"].extend(benchmark_results)

        # Run stress testing
        stress_testing = target_config.stress_testing or {}
        if stress_testing.get("enabled"):
            logger.info(f"Running stress testing for target '{target_name}'")
            stress_result = run_stress_testing(target, stress_testing)
            if stress_result:
                target_result["stress_testing"] = stress_result

        # Run security tests
        red_teaming = target_config.red_teaming or {}
        guardrails = target_config.guardrails or {}

        if red_teaming.get("enabled") or guardrails.get("enabled"):
            async def run_security_tests():
                results = {}
                if red_teaming.get("enabled"):
                    logger.info(f"Running red teaming for target '{target_name}'")
                    results["red_teaming"] = await run_red_teaming_async(
                        target, red_teaming, self._find_target_by_name
                    )
                if guardrails.get("enabled"):
                    logger.info(f"Testing guardrails for target '{target_name}'")
                    results["guardrails"] = await run_guardrails_testing_async(
                        target, guardrails, self._find_target_by_name
                    )
                return results

            security_results = asyncio.run(run_security_tests())

            if "red_teaming" in security_results:
                target_result["red_teaming"] = security_results["red_teaming"]
            if "guardrails" in security_results:
                target_result["guardrails"] = security_results["guardrails"]

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