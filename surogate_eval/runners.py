"""Evaluation runners for different test types."""

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from surogate_eval.config.eval_config import referenced_target
from surogate_eval.errors import ConfigError
from surogate_eval.targets import BaseTarget
from surogate_eval.utils.dist import is_master
from surogate_eval.utils.logger import get_logger
from surogate_eval.utils.text import blank_as_none

logger = get_logger()

#: Combined benchmark-level and row-level state for `progress.json`, set by
#: `eval.py` before each benchmark (the benchmark-level keys) and re-used by
#: row-level updates (the rest). One dict, not two: `_write_progress` and
#: `report_rows` always read and write across both halves together --
#: `_flush_progress` used to merge two separate dicts on every write, which
#: bought nothing but an extra step, since nothing ever needed one half
#: without the other. Seeded with every key up front so `_flush_progress`
#: can serialize it directly.
#:
#: Module state rather than a parameter because the backends that report rows
#: (`custom_eval`, the evalscope watcher) have no access to the task list and
#: threading it through every backend signature would touch far more code than
#: this feature is worth. One writer owns the whole document either way, so a
#: partial file is never produced.
_PROGRESS: dict = {
    "current_benchmark": "", "completed": 0, "total": 1,
    "rows_done": 0, "rows_total": 0, "scored": 0, "errored": 0,
    "passed": 0, "score_sum": 0.0,
}

#: Guards every read-check-write of `_PROGRESS`. `report_rows` reads
#: `current_benchmark` (the `for_benchmark` check) and then writes the row
#: keys, as two separate statements; without a lock, `_write_progress` can
#: run a full benchmark switch in between, and `report_rows` then resumes
#: and stamps the old benchmark's row counts onto the new benchmark's
#: context. `eval.py` calls `_write_progress` from the main thread while a
#: background watcher (or `custom_eval`'s own scoring loop) calls
#: `report_rows`, so this is a real cross-thread window, not a hypothetical
#: one. ``ponytail: one coarse lock, not per-field -- these writes are a few
#: dict operations every couple of seconds, so contention is not a real
#: cost.``
_PROGRESS_LOCK = threading.Lock()


def run_evaluation(
    target: BaseTarget,
    eval_config: Dict[str, Any],
    find_target_fn: Callable[[str], Optional[BaseTarget]],
    backend: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Run a single evaluation on a target.

    Args:
        target: Target to evaluate
        eval_config: Evaluation configuration dict
        find_target_fn: Function to find target by name (for judge targets)
        backend: Optional execution backend

    Returns:
        Evaluation results dict or None
    """
    from surogate_eval.datasets import DatasetLoader, DatasetValidator
    from surogate_eval.datasets.test_case import TestCase, MultiTurnTestCase
    from surogate_eval.metrics import (
        BatchMetricResult,
        LLMJudgeMetric,
        MetricRegistry,
        MetricResult,
    )
    from surogate_eval.targets.base import TargetRequest

    eval_name = eval_config.get("name", "unnamed")
    dataset_path = eval_config.get("dataset")

    logger.separator(char="─")
    logger.header(f"Evaluation: {eval_name}")
    logger.info(f"Dataset: {dataset_path}")
    logger.separator(char="─")

    if not dataset_path:
        logger.warning(f"No dataset specified for evaluation '{eval_name}'")
        return None

    try:
        metric_configs = eval_config.get("metrics", [])
        if not metric_configs:
            logger.warning(f"No metrics specified for evaluation '{eval_name}'")
            return None

        # Calculate max limit for dataset loading
        max_limit = None
        all_have_limits = True
        for mc in metric_configs:
            metric_limit = mc.get("limit")
            if metric_limit is None:
                all_have_limits = False
                break
            elif max_limit is None or metric_limit > max_limit:
                max_limit = metric_limit

        dataset_limit = max_limit if all_have_limits else None

        # Load dataset
        loader = DatasetLoader()
        dataset_type = loader.detect_dataset_type(dataset_path)
        logger.info(f"Dataset type: {dataset_type}")

        all_test_cases = loader.load_test_cases(dataset_path, limit=dataset_limit)
        logger.info(f"Loaded {len(all_test_cases)} test cases")

        # Validate dataset
        validator = DatasetValidator()
        df = loader.load(dataset_path, limit=dataset_limit)
        is_valid, errors = validator.validate(df)

        if not is_valid:
            logger.error("Dataset validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return {
                "name": eval_name,
                "dataset": dataset_path,
                "dataset_type": dataset_type,
                "status": "validation_failed",
                "errors": errors,
            }

        # Filter metrics by dataset type
        filtered_metric_configs = _filter_metrics_by_dataset_type(metric_configs, dataset_type)

        if not filtered_metric_configs:
            logger.error(f"No compatible metrics for dataset type: {dataset_type}")
            return None

        logger.info(f"Using {len(filtered_metric_configs)} metric(s)")

        metrics = MetricRegistry.create_metrics(filtered_metric_configs)

        # Inference cache
        inference_cache: Dict[int, tuple] = {}

        def get_inference(idx: int):
            if idx in inference_cache:
                return inference_cache[idx]

            test_case = all_test_cases[idx]
            try:
                if isinstance(test_case, TestCase):
                    request = TargetRequest(prompt=test_case.input)
                elif isinstance(test_case, MultiTurnTestCase):
                    request = TargetRequest(messages=test_case.get_context())
                else:
                    logger.error(f"Unknown test case type: {type(test_case)}")
                    inference_cache[idx] = ("", None)
                    return inference_cache[idx]

                response = target.send_request(request)
                inference_cache[idx] = (response.content, response)

            except Exception as e:
                logger.error(f"Failed to get output for test case {idx}: {e}")
                inference_cache[idx] = ("", None)

            return inference_cache[idx]

        # Run metrics
        metric_results = {}
        detailed_results: Dict[int, Dict] = {}

        for metric, metric_config in zip(metrics, filtered_metric_configs):
            logger.info(f"Running metric: {metric.name}")

            metric_limit = metric_config.get("limit")
            num_cases = min(metric_limit, len(all_test_cases)) if metric_limit else len(all_test_cases)

            logger.info(f"  Running inference on {num_cases} test cases")

            metric_test_cases = all_test_cases[:num_cases]
            metric_outputs = []
            metric_responses = []

            for idx in range(num_cases):
                output, response = get_inference(idx)
                metric_outputs.append(output)
                metric_responses.append(response)

                if (idx + 1) % 10 == 0:
                    logger.step(idx + 1, num_cases, f"Progress: {idx + 1}/{num_cases}")

            try:
                # Set judge target if needed
                if isinstance(metric, LLMJudgeMetric):
                    judge_config = metric.config.get("judge_model", {})
                    judge_target_name = judge_config.get("target")

                    if judge_target_name:
                        judge_target = find_target_fn(judge_target_name)
                        if judge_target:
                            metric.set_judge_target(judge_target)
                            logger.debug(f"Set judge target '{judge_target_name}'")
                        else:
                            logger.warning(f"Judge target '{judge_target_name}' not found")

                batch_result = metric.evaluate_batch(metric_test_cases, metric_outputs, metric_responses)

                metric_results[metric.name] = batch_result.to_dict()

                # Store detailed results
                for i, individual_result in enumerate(batch_result.results):
                    if i not in detailed_results:
                        if isinstance(metric_test_cases[i], TestCase):
                            input_text = metric_test_cases[i].input
                        else:
                            input_text = [
                                {"role": turn.role, "content": turn.content}
                                for turn in metric_test_cases[i].turns
                            ]

                        detailed_results[i] = {
                            "test_case_index": i,
                            "input": input_text,
                            "output": metric_outputs[i] or "",
                            "metrics": {},
                        }

                    detailed_results[i]["metrics"][metric.name] = {
                        "score": individual_result.score,
                        "success": individual_result.success,
                        "reason": individual_result.reason,
                        "metadata": individual_result.metadata,
                    }

                logger.metric(f"{metric.name} - Avg Score", f"{batch_result.avg_score:.3f}")
                logger.metric(f"{metric.name} - Success Rate", f"{batch_result.success_rate:.3f}")

            except Exception as e:
                logger.error(f"Metric {metric.name} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

                # One errored unit per case that went unmeasured. A bare
                # {"error": ...} dict counted as a single error however many
                # cases the batch was going to measure, so a 200-case metric
                # crashing looked no worse than one bad answer.
                unmeasured = len(metric_test_cases) or 1
                failed_batch = BatchMetricResult(
                    metric_name=metric.name,
                    metric_type=metric.metric_type,
                    results=[
                        MetricResult.errored(
                            metric_name=metric.name,
                            metric_type=metric.metric_type,
                            reason=f"Metric batch failed: {e}",
                            metadata={'error_kind': type(e).__name__},
                        )
                        for _ in range(unmeasured)
                    ],
                )
                # ``error`` marks the metric as failed in the report; no
                # ``status`` alongside it, or the outcome walk would count the
                # crash once more on top of the per-case counts.
                metric_results[metric.name] = {**failed_batch.to_dict(), "error": str(e)}

        detailed_results_list = [detailed_results[i] for i in sorted(detailed_results.keys())]

        return {
            "name": eval_name,
            "dataset": dataset_path,
            "dataset_type": dataset_type,
            "num_test_cases": len(inference_cache),
            "num_metrics": len(metrics),
            "status": "completed",
            "metrics_summary": metric_results,
            "detailed_results": detailed_results_list,
        }

    except Exception as e:
        logger.error(f"Failed to run evaluation '{eval_name}': {e}")
        import traceback
        traceback.print_exc()

        return {
            "name": eval_name,
            "dataset": dataset_path,
            "status": "failed",
            "error": str(e),
        }


def run_benchmarks(
    target: BaseTarget,
    benchmark_configs: List[Dict[str, Any]],
    find_target_fn: Callable[[str], Optional[BaseTarget]],
) -> List[Dict[str, Any]]:
    """
    Run benchmarks on target.

    Args:
        target: Target to evaluate
        benchmark_configs: List of benchmark configurations
        find_target_fn: Function to find target by name

    Returns:
        List of benchmark results
    """
    if not benchmark_configs:
        return []

    results = []
    for bench_config in benchmark_configs:
        result = _run_single_benchmark(target, bench_config, find_target_fn)
        if result:
            results.append(result)

    return results


def _write_bench_result(result: Dict[str, Any]) -> None:
    """Write an individual benchmark result to eval_results/bench_{name}.json.

    Rank 0 only: under a distributed relaunch every process evaluates the
    same benchmark and would write this same path, so the file ops reads
    would be whichever copy landed last (E-RUN-6).
    """
    import json as _json
    from enum import Enum as _Enum
    from pathlib import Path as _Path

    if not is_master():
        return

    name = result.get("benchmark_name") or result.get("name", "unknown")
    try:
        out = _Path("eval_results")
        out.mkdir(exist_ok=True)
        path = out / f"bench_{name}.json"

        def _convert_enum_keys(obj):
            """Recursively convert enum keys/values to strings."""
            if isinstance(obj, dict):
                return {
                    (k.value if isinstance(k, _Enum) else k): _convert_enum_keys(v)
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [_convert_enum_keys(i) for i in obj]
            if isinstance(obj, _Enum):
                return obj.value
            return obj

        with open(path, "w") as f:
            _json.dump(_convert_enum_keys(result), f, indent=2, default=str)
        logger.info(f"Benchmark result saved: {path}")
    except Exception as e:
        logger.warning(f"Failed to write benchmark result for {name}: {e}")


def _progress_path() -> Path:
    return Path("eval_results") / "progress.json"


def _flush_progress() -> None:
    """Write the whole document atomically.

    Temp file plus rename, because ops polls this path every 5s and a
    truncating in-place write can be read half-finished and parsed as invalid
    JSON. Best-effort: a failure here must never fail the run.

    Rank 0 only (E-RUN-6). The atomic rename protects a reader from a partial
    document; it does nothing about N processes each publishing their own
    progress, which is a bar that jumps backwards rather than a broken parse.
    """
    if not is_master():
        return
    try:
        out = Path("eval_results")
        out.mkdir(exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(out), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(_PROGRESS, f)
            # mkstemp defaults to 0600; the previous writer used open(mode='w')
            # which produced a umask-derived mode. A reader in another account
            # must still be able to read this file.
            os.chmod(tmp, 0o644)
            os.replace(tmp, str(_progress_path()))
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception:
        pass  # Best-effort


def _write_progress(
    current_benchmark: str, completed: int, total: int, *, clear_rows: bool = True,
) -> None:
    """Record which benchmark is running, and (usually) zero out any previous
    row counts.

    Zeroed, not cleared to ``{}``: ops treats an absent row key as "this
    runner does not report rows" and leaves whatever rows_done/rows_total it
    already has in the database, so omitting the keys here left benchmark 2
    displaying benchmark 1's finished bar until its own first report -- and
    then dropping backwards onto that report's real, much smaller count.
    Zeros are a real value ops ingests (``0 is not None``), and
    ``rows_total: 0`` already means "unknown" on both sides, so this is a
    genuine reset rather than a second flavour of "no data".

    ``clear_rows=False`` for ``eval.py``'s final ``"done"`` call: that write
    marks the run ending, not a switch to a new benchmark, and the row block
    it would otherwise zero holds the run's real final counts -- exactly
    what ops needs to settle a completed run to. Zeroing there was
    indistinguishable, from ops' side, from a benchmark switch, which is why
    a completed run's rows read back as 0/0 (Finding 1: `_ingest_incremental`
    ingests that zeroed write in the same tick, immediately before finalize,
    so `_settled_progress_updates` sees `rows_total = 0` and leaves the
    settle a no-op).
    """
    with _PROGRESS_LOCK:
        updates = {
            "current_benchmark": current_benchmark,
            "completed": completed,
            "total": total,
        }
        if clear_rows:
            updates.update({
                "rows_done": 0,
                "rows_total": 0,
                "scored": 0,
                "errored": 0,
                "passed": 0,
                "score_sum": 0.0,
            })
        _PROGRESS.update(updates)
        _flush_progress()


def report_rows(
    rows_done: int,
    rows_total: int,
    scored: int,
    passed: int,
    score_sum: float,
    *,
    for_benchmark: str | None = None,
) -> None:
    """Publish row-level progress for the benchmark currently running.

    ``score_sum`` rather than an average so ops does the division: a partial
    file then cannot be internally inconsistent.

    Not a parameter: every row is exactly ``scored`` or ``errored``, so
    ``errored`` is always ``rows_done - scored`` -- every caller used to
    compute exactly that before calling in, so deriving it here leaves one
    place for that arithmetic instead of one per caller.

    ``for_benchmark``, when given, must match ``_PROGRESS``'s current
    benchmark or the write is dropped. A background watcher's ``stop()`` can
    time out while a tick is still in flight (``_evalscope_progress.
    ReviewWatcher``): the thread stays alive, ticks again, and that write
    would otherwise land after the *next* benchmark's watcher has already
    started -- stamping the old benchmark's counts over the new one's row
    block and sending ``rows_done`` backwards, exactly what Task 2 ruled
    out. Tagging each write with the benchmark it was measured for turns
    that stale write into a silent no-op instead. Default ``None`` preserves
    today's behaviour for ``custom_eval``, which reports from the same
    thread as its scoring loop and so can never go stale.

    The check and the write below share ``_PROGRESS_LOCK`` with
    ``_write_progress``'s own critical section: read-then-write is not
    atomic on its own, and a benchmark switch landing in that gap is the
    same stale-write hazard as the leaked-watcher case above, just reached
    by a race instead of a slow ``stop()``.

    ``rows_done`` and ``scored`` are not always from the same instant.
    ``custom_eval`` derives both from the same in-memory results list, so
    there they always agree (``rows_done`` is never less than ``scored``).
    The evalscope watcher does not have that luxury: its ``rows_done`` comes
    from evalscope's own ``ProgressTracker`` file, which flushes at most
    once a second, while its ``scored`` comes from an incremental read of
    the reviews JSONL (``ReviewCounter``), which evalscope appends to
    *before* it updates the tracker -- so ``scored`` is reliably ahead of
    ``rows_done`` on nearly every tick of a healthy run. Reconciling here,
    not just clamping the subtraction, keeps both published numbers
    coherent: a bar that shows fewer rows done than rows scored is wrong on
    its face, and this is the one place every caller's numbers pass through
    before publication. Do not simplify this back to ``rows_done - scored``.
    """
    with _PROGRESS_LOCK:
        if for_benchmark is not None and for_benchmark != _PROGRESS["current_benchmark"]:
            return
        # rows_done can lag scored (see above) -- raised to the coherent
        # floor before errored is derived from it. Not a redundant max: see
        # the docstring for why the two can disagree.
        rows_done = max(rows_done, scored)
        _PROGRESS.update({
            "rows_done": rows_done,
            "rows_total": rows_total,
            "scored": scored,
            "errored": rows_done - scored,
            "passed": passed,
            "score_sum": score_sum,
        })
        _flush_progress()


def _run_single_benchmark(
    target: BaseTarget,
    bench_config: Dict[str, Any],
    find_target_fn: Callable[[str], Optional[BaseTarget]],
) -> Optional[Dict[str, Any]]:
    """Run a single benchmark on target."""
    from surogate_eval.benchmarks import BenchmarkConfig, BenchmarkRegistry

    benchmark_name = bench_config.get("name")
    logger.info(f"Running benchmark: {benchmark_name}")

    try:
        config = BenchmarkConfig(
            name=benchmark_name,
            backend=bench_config.get("backend", "evalscope"),
            source=bench_config.get("source"),
            columns=bench_config.get("columns", {}),
            split=bench_config.get("split", "test"),
            prompt_template=bench_config.get("prompt_template"),
            stop_sequences=bench_config.get("stop_sequences"),
            path=bench_config.get("path"),
            num_fewshot=bench_config.get("num_fewshot"),
            limit=bench_config.get("limit"),
            pass_threshold=bench_config.get("pass_threshold"),
            tasks=bench_config.get("tasks"),
            subset=bench_config.get("subset"),
            use_cache=bench_config.get("use_cache", True),
            cache_dir=bench_config.get("cache_dir"),
            backend_params=bench_config.get("backend_params", {}),
            dataset_hub=bench_config.get("dataset_hub"),
            tokenizer=bench_config.get("tokenizer"),
            batch_size=bench_config.get("batch_size"),
            max_tokens=bench_config.get("max_tokens"),
            temperature=bench_config.get("temperature"),
            top_p=bench_config.get("top_p"),
            top_k=bench_config.get("top_k"),
            min_p=bench_config.get("min_p"),
            presence_penalty=bench_config.get("presence_penalty"),
            enable_thinking=bench_config.get("enable_thinking"),
            system_prompt=bench_config.get("system_prompt"),
            num_concurrent=bench_config.get("num_concurrent"),
            log_samples=bench_config.get("log_samples", True),
            judge_model=bench_config.get("judge_model"),
            judge_criteria=bench_config.get("judge_criteria"),
            eval_type=bench_config.get("eval_type", "exact_match"),
            matcher=bench_config.get("matcher"),
        )

        benchmark = BenchmarkRegistry.create_benchmark(config)

        # Set judge target if specified
        judge_model_config = bench_config.get("judge_model")
        if judge_model_config:
            judge_target_name = judge_model_config.get("target")
            judge_target = find_target_fn(judge_target_name)
            if judge_target:
                benchmark.config.backend_params["judge_target"] = judge_target
                logger.info(f"Using judge '{judge_target_name}' for benchmark '{benchmark_name}'")
            else:
                logger.warning(f"Judge target '{judge_target_name}' not found")

        # Validate target compatibility
        if not benchmark.validate_target(target):
            target_type = target.target_type.value
            required = benchmark.REQUIRED_TARGET_TYPES
            logger.error(
                f"Target '{target.name}' (type: {target_type}) not compatible with "
                f"benchmark '{benchmark_name}' (requires: {required})"
            )
            return {
                "benchmark": benchmark_name,
                "status": "incompatible",
                "error": f"Benchmark requires {required} target, got {target_type}",
            }

        result = benchmark.evaluate(target)
        result_dict = result.to_dict()
        result_dict["status"] = "completed"

        logger.success(f"Benchmark '{benchmark_name}' completed")
        logger.metric(f"{benchmark_name} - Overall Score", f"{result.overall_score:.4f}")

        return result_dict

    except Exception as e:
        logger.error(f"Benchmark '{benchmark_name}' failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

        return {"benchmark_name": benchmark_name, "status": "failed", "error": str(e)}


def _stress_failure(reason: str) -> Dict[str, Any]:
    """A stress run that ended before it could report its own counts.

    These are the paths that never reach ``StressTestResult.to_dict()``: a
    missing dataset, or anything raised on the way to the first request. They
    used to return a bare status, which carries no count keys, so the outcome
    walk fell through to its generic failure branch and charged the crash to
    the MEASUREMENT channel - the one thing the load/measurement split exists
    to prevent. A stress test that died is a load failure, on the load
    channel, against the load error rate.

    The zeroes are the point: the keys declare the channel, and
    ``_collect_counts`` adds one errored unit for the failure status itself.
    """
    return {
        "status": "error",
        "reason": reason,
        # See surogate_eval.outcome: LOAD_COUNT_KEYS.
        "load_scored_n": 0,
        "load_errored_n": 0,
    }


def run_stress_testing(
    target: BaseTarget,
    stress_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run stress testing on target.

    Args:
        target: Target to stress test
        stress_config: Stress testing configuration

    Returns:
        Stress test results
    """
    from surogate_eval.datasets import DatasetLoader
    from surogate_eval.metrics.stress import StressTestConfig, StressTester

    logger.info(f"Running stress test for target '{target.name}'")

    try:
        dataset_path = stress_config.get("dataset")
        if not dataset_path:
            logger.error("No dataset specified for stress testing")
            return _stress_failure("No dataset specified")

        loader = DatasetLoader()
        test_cases = loader.load_test_cases(dataset_path)

        logger.info(f"Loaded {len(test_cases)} test cases for stress testing")

        config = StressTestConfig(
            num_concurrent=stress_config.get("num_concurrent", 10),
            duration_seconds=stress_config.get("duration_seconds"),
            num_requests=stress_config.get("num_requests", 100),
            progressive=stress_config.get("progressive", False),
            start_concurrent=stress_config.get("start_concurrent", 1),
            step_concurrent=stress_config.get("step_concurrent", 2),
            step_duration_seconds=stress_config.get("step_duration_seconds", 30),
            monitor_resources=stress_config.get("monitor_resources", True),
            warmup_requests=stress_config.get("warmup_requests", 5),
        )

        tester = StressTester(target, test_cases)
        result = tester.run(config)

        return result.to_dict()

    except Exception as e:
        logger.error(f"Stress testing failed: {e}")
        import traceback
        traceback.print_exc()
        return _stress_failure(str(e))


def _judge_target(
    role: str,
    block: Any,
    target: BaseTarget,
    find_target_fn: Callable[[str], Optional[BaseTarget]],
) -> BaseTarget:
    """The target named for a judging role, or a ConfigError.

    Nothing is substituted. Every substitution this replaces ended in the
    same place -- the model under test grading its own answers (E-RUN-7) --
    and a scan that graded itself is worse than one that did not run, because
    only one of the two says so.

    Config validation rejects both shapes this raises on, so reaching here
    means the two disagree. That is a reason to be loud, not a reason to
    assume it cannot happen.
    """
    name = referenced_target(block)
    if not name:
        raise ConfigError(
            f"{role} must name a target, got {block!r}. Without one the "
            f"target under test would grade its own answers."
        )

    judge = find_target_fn(name)
    if judge is None:
        raise ConfigError(f"{role} names target '{name}', which is not configured.")

    if judge.name == target.name:
        raise ConfigError(
            f"{role} names '{name}', the target being evaluated. A target "
            f"cannot judge its own answers -- name a different target, or a "
            f"provider model."
        )

    return judge


def _judge_model(
    role: str,
    block: Any,
    target: BaseTarget,
    find_target_fn: Callable[[str], Optional[BaseTarget]],
):
    """A judge DeepTeam can call: a wrapper around a target, or a model name.

    A plain string names a provider model rather than a target, so it cannot
    be the target under test. It can still be unusable, and the case that
    made this worth checking is a config that names an OpenAI judge on a pod
    with no OpenAI key: the old code answered that by handing the grading to
    the target, which is the one substitution the user would least expect
    from naming a judge explicitly.
    """
    from surogate_eval.models import DeepEvalTargetWrapper

    if isinstance(block, str) and blank_as_none(block):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ConfigError(
                f"{role}='{block}' names a provider model but OPENAI_API_KEY "
                f"is not set. Export it, or name a target to judge with."
            )
        return block

    return DeepEvalTargetWrapper(_judge_target(role, block, target, find_target_fn))


async def run_red_teaming_async(
    target: BaseTarget,
    red_team_config: Dict[str, Any],
    find_target_fn: Callable[[str], Optional[BaseTarget]],
) -> Dict[str, Any]:
    """
    Run red teaming tests on target.

    Args:
        target: Target to test
        red_team_config: Red teaming configuration
        find_target_fn: Function to find target by name

    Returns:
        Red teaming results
    """
    from surogate_eval.models import DeepEvalTargetWrapper
    from surogate_eval.security import RedTeamConfig, RedTeamRunner

    logger.info(f"Running red-team scan for target '{target.name}'")

    try:
        # Resolve simulator_model — prefer explicit target ref, fall back
        # to the target being tested so we never hit bare OpenAI calls.
        simulator_model = red_team_config.get("simulator_model", None)
        if isinstance(simulator_model, dict) and simulator_model.get("target"):
            sim_target = find_target_fn(simulator_model["target"])
            if sim_target:
                simulator_model = DeepEvalTargetWrapper(sim_target)
                logger.info(f"Using target '{simulator_model.get_model_name()}' as simulator model")
            else:
                logger.warning(f"Simulator target '{simulator_model['target']}' not found, falling back to eval target")
                simulator_model = DeepEvalTargetWrapper(target)
        elif isinstance(simulator_model, str) and not os.environ.get("OPENAI_API_KEY"):
            logger.warning(f"No OPENAI_API_KEY for simulator_model='{simulator_model}', using eval target instead")
            simulator_model = DeepEvalTargetWrapper(target)
        elif simulator_model is None:
            # Allowed, unlike the evaluator below: writing your own attacks
            # is a weaker scan, not a fabricated verdict. Said out loud
            # because it is a real limitation of the results.
            logger.warning(
                f"No simulator_model configured; target '{target.name}' will "
                f"generate the attacks it is being tested with"
            )
            simulator_model = DeepEvalTargetWrapper(target)

        # Resolve evaluation_model. This one decides whether an attack
        # succeeded, so it may not be the target under test (E-RUN-7).
        evaluation_model = _judge_model(
            "red_teaming.evaluation_model",
            red_team_config.get("evaluation_model"),
            target,
            find_target_fn,
        )

        config = RedTeamConfig(
            vulnerabilities=red_team_config.get("vulnerabilities", []),
            vulnerability_types=red_team_config.get("vulnerability_types", {}),
            attacks=red_team_config.get("attacks", []),
            attacks_per_vulnerability=red_team_config.get("attacks_per_vulnerability", 3),
            max_concurrent=red_team_config.get("max_concurrent", 10),
            run_async=red_team_config.get("run_async", True),
            simulator_model=simulator_model,
            evaluation_model=evaluation_model,
            purpose=red_team_config.get("purpose"),
            ignore_errors=red_team_config.get("ignore_errors", True),
        )

        runner = RedTeamRunner(target, config)

        # Set translator if configured
        translator_config = red_team_config.get("translator")
        if translator_config:
            translator_target_name = translator_config.get("target")
            if translator_target_name:
                translator_target = find_target_fn(translator_target_name)
                if translator_target:
                    runner.set_translator(translator_target)
                    logger.info(f"Using translator target '{translator_target_name}'")
                else:
                    logger.warning(f"Translator target '{translator_target_name}' not found")

        risk_assessment = await runner.run()

        return risk_assessment.to_dict()

    except Exception as e:
        logger.error(f"Red-teaming failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

        return {"status": "failed", "error": str(e)}


async def run_guardrails_testing_async(
    target: BaseTarget,
    guardrails_config: Dict[str, Any],
    find_target_fn: Callable[[str], Optional[BaseTarget]],
) -> Dict[str, Any]:
    """
    Test guardrails on target.

    Args:
        target: Target to test
        guardrails_config: Guardrails configuration
        find_target_fn: Function to find target by name

    Returns:
        Guardrails test results
    """
    from surogate_eval.models import DeepEvalTargetWrapper
    from surogate_eval.security import GuardrailsConfig, GuardrailsEvaluator

    logger.info(f"Testing guardrails for target '{target.name}'")

    try:
        # Resolve simulator_model
        simulator_model = guardrails_config.get("simulator_model", "gpt-3.5-turbo")
        if isinstance(simulator_model, dict) and simulator_model.get("target"):
            sim_target = find_target_fn(simulator_model["target"])
            if sim_target:
                simulator_model = DeepEvalTargetWrapper(sim_target)
                logger.info(f"Using target '{simulator_model.get_model_name()}' as simulator model")
            else:
                logger.warning(f"Simulator target '{simulator_model['target']}' not found, using default")
                simulator_model = "gpt-3.5-turbo"

        # Both judging roles must be someone other than the target under
        # test (E-RUN-7), and both are resolved before anything is built so
        # the section fails on its config rather than part way through a scan.
        evaluation_model = _judge_model(
            "guardrails.evaluation_model",
            guardrails_config.get("evaluation_model"),
            target,
            find_target_fn,
        )
        judge_target = _judge_target(
            "guardrails.refusal_judge_model",
            guardrails_config.get("refusal_judge_model"),
            target,
            find_target_fn,
        )
        logger.info(f"Using target '{judge_target.name}' as refusal judge")

        config = GuardrailsConfig(
            vulnerabilities=guardrails_config.get("vulnerabilities", []),
            vulnerability_types=guardrails_config.get("vulnerability_types", {}),
            attacks=guardrails_config.get("attacks", []),
            attacks_per_vulnerability=guardrails_config.get("attacks_per_vulnerability", 3),
            safe_prompts_dataset=guardrails_config.get("safe_prompts_dataset"),
            refusal_judge_model_target=judge_target.name,
            max_concurrent=guardrails_config.get("max_concurrent", 10),
            simulator_model=simulator_model,
            evaluation_model=evaluation_model,
            purpose=guardrails_config.get("purpose"),
            ignore_errors=guardrails_config.get("ignore_errors", True),
        )

        evaluator = GuardrailsEvaluator(target, config, judge_target)
        result = await evaluator.evaluate()

        return result.to_dict()

    except Exception as e:
        logger.error(f"Guardrails testing failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

        return {"status": "failed", "error": str(e)}


def _filter_metrics_by_dataset_type(
    metric_configs: List[Dict[str, Any]],
    dataset_type: str,
) -> List[Dict[str, Any]]:
    """Filter metrics based on dataset type compatibility."""
    single_turn_metrics = {
        "g_eval", "dag", "multimodal_g_eval", "toxicity", "bias", "harm",
        "embedding_similarity", "classification", "latency", "throughput",
        "token_generation_speed",
    }

    multi_turn_metrics = {
        "conversational_g_eval", "conversation_coherence", "context_retention",
        "turn_analysis", "conversational_dag", "multimodal_g_eval", "toxicity",
        "bias", "harm", "latency", "throughput", "token_generation_speed",
    }

    filtered = []
    skipped = []

    for config in metric_configs:
        metric_type = config.get("type")
        metric_name = config.get("name", metric_type)

        is_compatible = (
            (dataset_type == "single_turn" and metric_type in single_turn_metrics)
            or (dataset_type == "multi_turn" and metric_type in multi_turn_metrics)
        )

        if is_compatible:
            filtered.append(config)
            logger.debug(f"Metric '{metric_name}' is compatible with {dataset_type}")
        else:
            skipped.append(metric_name)
            logger.warning(f"Skipping metric '{metric_name}' - incompatible with {dataset_type}")

    if skipped:
        logger.info(f"Skipped {len(skipped)} incompatible metrics: {', '.join(skipped)}")

    return filtered