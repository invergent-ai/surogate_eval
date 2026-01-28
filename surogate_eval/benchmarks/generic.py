# surogate/eval/benchmarks/generic.py

from typing import Dict, Any
from .base import BaseBenchmark, BenchmarkResult, BenchmarkConfig
from ..targets import BaseTarget
from ..utils.logger import get_logger

logger = get_logger()


class GenericBenchmark(BaseBenchmark):
    """Generic benchmark that delegates to backend."""

    VISION_BENCHMARKS = {
        'mmmu', 'mathvista', 'chartqa', 'docvqa', 'infovqa',
        'ocrvqa', 'ai2d', 'scienceqa', 'seedbench'
    }

    EMBEDDING_BENCHMARKS = {
        'mteb', 'cmteb', 'beir'
    }

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

        if self.name.lower() in self.VISION_BENCHMARKS:
            self.REQUIRED_TARGET_TYPES = ['multimodal']
        elif self.name.lower() in self.EMBEDDING_BENCHMARKS:
            self.REQUIRED_TARGET_TYPES = ['embedding']
        else:
            self.REQUIRED_TARGET_TYPES = []

    def evaluate(self, target: BaseTarget) -> BenchmarkResult:
        logger.info(f"Evaluating {target.name} on {self.name}")
        tokenizer = self.config.tokenizer or target.config.get('tokenizer')

        backend_config = {
            # Dataset
            'source': self.config.source,
            'columns': self.config.columns,
            'split': self.config.split,
            'prompt_template': self.config.prompt_template,
            'stop_sequences': self.config.stop_sequences,
            # Eval params
            'tasks': self.config.tasks or [self.name],
            'num_fewshot': self.config.num_fewshot,
            'limit': self.config.limit,
            'subset': self.config.subset,
            'path': self.config.path,
            'dataset_hub': self.config.dataset_hub,
            'batch_size': self.config.batch_size or self.config.backend_params.get('batch_size', 1),
            'backend_params': self.config.backend_params,
            'tokenizer': tokenizer,
            # Generation params
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'min_p': self.config.min_p,
            'presence_penalty': self.config.presence_penalty,
            'enable_thinking': self.config.enable_thinking,
            'num_concurrent': self.config.num_concurrent,
            'log_samples': self.config.log_samples,
            'system_prompt': self.config.system_prompt,
            'judge_model': self.config.judge_model,
            'judge_criteria': self.config.judge_criteria,
            'eval_type': self.config.eval_type,
        }

        if self.config.backend_params.get('judge_target'):
            backend_config['backend_params'] = {
                'judge_target': self.config.backend_params['judge_target']
            }

        backend_results = self.backend.evaluate(
            target=target,
            benchmark_name=self.name,
            config=backend_config
        )

        result = self._parse_results(backend_results)
        logger.success(f"{self.name} completed. Score: {result.overall_score:.4f}")
        return result

    def _parse_results(self, backend_results: Dict[str, Any]) -> BenchmarkResult:
        overall_score = backend_results.get('overall_score', 0.0)
        task_results = backend_results.get('task_results', {})
        detailed_results = backend_results.get('detailed_results', [])

        num_samples = sum(
            task.get('n_samples', 0)
            for task in task_results.values()
            if isinstance(task, dict)
        )

        backend_name = backend_results.get('metadata', {}).get('backend', self.config.backend)

        return BenchmarkResult(
            benchmark_name=self.name,
            overall_score=overall_score,
            num_samples=num_samples,
            backend=backend_name,
            task_results=task_results,
            detailed_results=detailed_results,
            metadata=backend_results.get('metadata', {})
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': f'Benchmark: {self.name}',
            'backend': self.config.backend,
            'required_target_types': self.REQUIRED_TARGET_TYPES,
        }