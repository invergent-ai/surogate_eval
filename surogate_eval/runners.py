"""Evaluation runners for different test types."""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from surogate_eval.targets import BaseTarget
from surogate_eval.utils.logger import get_logger

logger = get_logger()


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
    from surogate_eval.metrics import MetricRegistry, LLMJudgeMetric
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
                metric_results[metric.name] = {"error": str(e), "status": "failed"}

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

    logger.separator(char="─")
    logger.header(f"Running {len(benchmark_configs)} Benchmark(s)")
    logger.separator(char="─")

    results = []
    for bench_config in benchmark_configs:
        result = _run_single_benchmark(target, bench_config, find_target_fn)
        if result:
            results.append(result)

    return results


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
            system_prompt=bench_config.get("system_prompt"),
            num_concurrent=bench_config.get("num_concurrent"),
            log_samples=bench_config.get("log_samples", True),
            judge_model=bench_config.get("judge_model"),
            judge_criteria=bench_config.get("judge_criteria"),
            eval_type=bench_config.get("eval_type", "exact_match"),
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

        return {"benchmark": benchmark_name, "status": "failed", "error": str(e)}


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
            return {"status": "error", "reason": "No dataset specified"}

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
        return {"status": "error", "reason": str(e)}


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
        # Resolve simulator_model
        simulator_model = red_team_config.get("simulator_model", "gpt-4o-mini")
        if isinstance(simulator_model, dict) and simulator_model.get("target"):
            sim_target = find_target_fn(simulator_model["target"])
            if sim_target:
                simulator_model = DeepEvalTargetWrapper(sim_target)
                logger.info(f"Using target '{simulator_model.get_model_name()}' as simulator model")
            else:
                logger.warning(f"Simulator target '{simulator_model['target']}' not found, using default")
                simulator_model = "gpt-4o-mini"

        # Resolve evaluation_model
        evaluation_model = red_team_config.get("evaluation_model", "gpt-4o-mini")
        if isinstance(evaluation_model, dict) and evaluation_model.get("target"):
            eval_target = find_target_fn(evaluation_model["target"])
            if eval_target:
                evaluation_model = DeepEvalTargetWrapper(eval_target)
                logger.info(f"Using target '{evaluation_model.get_model_name()}' as evaluation model")
            else:
                logger.warning(f"Evaluation target '{evaluation_model['target']}' not found, using default")
                evaluation_model = "gpt-4o-mini"

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
            ignore_errors=red_team_config.get("ignore_errors", False),
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

        # Resolve evaluation_model
        evaluation_model = guardrails_config.get("evaluation_model", "gpt-4o-mini")
        if isinstance(evaluation_model, dict) and evaluation_model.get("target"):
            eval_target = find_target_fn(evaluation_model["target"])
            if eval_target:
                evaluation_model = DeepEvalTargetWrapper(eval_target)
                logger.info(f"Using target '{evaluation_model.get_model_name()}' as evaluation model")
            else:
                logger.warning(f"Evaluation target '{evaluation_model['target']}' not found, using default")
                evaluation_model = "gpt-4o-mini"

        config = GuardrailsConfig(
            vulnerabilities=guardrails_config.get("vulnerabilities", []),
            vulnerability_types=guardrails_config.get("vulnerability_types", {}),
            attacks=guardrails_config.get("attacks", []),
            attacks_per_vulnerability=guardrails_config.get("attacks_per_vulnerability", 3),
            safe_prompts_dataset=guardrails_config.get("safe_prompts_dataset"),
            refusal_judge_model_target=guardrails_config.get("refusal_judge_model", {}).get("target"),
            max_concurrent=guardrails_config.get("max_concurrent", 10),
            simulator_model=simulator_model,
            evaluation_model=evaluation_model,
            purpose=guardrails_config.get("purpose"),
            ignore_errors=guardrails_config.get("ignore_errors", False),
        )

        # Get judge target if specified
        judge_target = None
        if config.refusal_judge_model_target:
            judge_target = find_target_fn(config.refusal_judge_model_target)
            if not judge_target:
                logger.warning(f"Judge target '{config.refusal_judge_model_target}' not found")
            else:
                logger.info(f"Using target '{judge_target.name}' as refusal judge")

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