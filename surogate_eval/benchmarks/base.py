# surogate/eval/benchmarks/base.py
"""Base classes for benchmark evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, Literal

from surogate_eval.statuses import FAILED_STATUSES
from surogate_eval.targets import BaseTarget
from surogate_eval.utils.logger import get_logger

logger = get_logger()


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""
    name: str
    backend: Literal['evalscope', 'lm_eval', 'custom_eval'] = 'evalscope'

    # Dataset source (HF path, LakeFS, or local file)
    source: Optional[str] = None

    # Column mappings (instruction, answer, eval_type, judge_criteria)
    columns: Dict[str, str] = field(default_factory=dict)
    split: str = 'test'
    prompt_template: Optional[str] = None
    stop_sequences: Optional[List[str]] = None

    # Legacy/common fields
    path: Optional[str] = None  # deprecated, use source
    num_fewshot: Optional[int] = None
    limit: Optional[Union[int, float]] = None
    #: Minimum score for a row to count as a pass, as a fraction of the
    #: metric's scale (0-1). Absent, each backend keeps the rule it already
    #: had -- see benchmarks/pass_rule.py.
    pass_threshold: Optional[float] = None
    tasks: Optional[List[str]] = None
    subset: Optional[Union[str, List[str]]] = None
    dataset_hub: Optional[str] = None
    log_samples: bool = True

    # Additional parameters
    backend_params: Dict[str, Any] = field(default_factory=dict)

    # Cache configuration
    use_cache: bool = True
    cache_dir: Optional[str] = None

    # Generation/eval params
    tokenizer: Optional[str] = None
    batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = None
    num_concurrent: Optional[int] = None
    system_prompt: Optional[str] = None
    judge_model: Optional[Dict[str, Any]] = None
    judge_criteria: Optional[str] = None
    eval_type: str = 'exact_match'
    #: How string rows decide correctness. See benchmarks/matching.py.
    matcher: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    benchmark_name: str
    overall_score: float
    num_samples: int
    backend: str = "evalscope"
    task_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def result_counts(self) -> Tuple[int, int]:
        """Countable units for the run outcome, as ``(scored, errored)``.

        One unit is one benchmark task. Per-sample counting is not usable
        here: several backends report a task without a trustworthy sample
        count (lm-eval leaves ``n_samples`` at 0 whenever the harness does
        not populate it), so counting samples would report a healthy
        benchmark as having measured nothing - the exact failure this
        method exists to prevent.

        A task that reports its own ``scored_n``/``errored_n`` (the
        custom-eval toxicity path) contributes those finer counts instead,
        so the detail it went to the trouble of producing is not flattened
        to 1.
        """
        scored = errored = 0

        for task in self.task_results.values():
            if not isinstance(task, dict):
                scored += 1
                continue

            if 'scored_n' in task and 'errored_n' in task:
                scored += int(task['scored_n'])
                errored += int(task['errored_n'])
                continue

            if task.get('status') in FAILED_STATUSES:
                errored += 1
                continue

            # A task that explicitly saw zero samples measured nothing, and
            # must not be counted as evidence that it did.
            if task.get('n_samples', task.get('total')) == 0:
                continue

            scored += 1

        if scored == 0 and errored == 0:
            # A benchmark that came back with no task at all measured
            # nothing. Reported as one errored unit so it fails the run
            # loudly instead of dissolving into a zero error rate.
            errored = 1

        return scored, errored

    def to_dict(self) -> Dict[str, Any]:
        scored_n, errored_n = self.result_counts()
        return {
            'benchmark_name': self.benchmark_name,
            'backend': self.backend,
            'overall_score': self.overall_score,
            'num_samples': self.num_samples,
            'scored_n': scored_n,
            'errored_n': errored_n,
            'task_results': self.task_results,
            'detailed_results': self.detailed_results,
            'metadata': self.metadata,
        }

    def get_summary(self) -> str:
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"Overall Score: {self.overall_score:.4f}",
            f"Samples: {self.num_samples}",
        ]
        if self.task_results:
            lines.append("\nTask Results:")
            for task_name, task_data in self.task_results.items():
                score = task_data.get('score', task_data.get('accuracy', 'N/A'))
                lines.append(f"  {task_name}: {score}")
        return "\n".join(lines)


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    REQUIRED_TARGET_TYPES: List[str] = []

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.name = config.name

        if config.backend == 'lm_eval':
            from .backends.lm_eval_backend import LMEvalBackend
            self.backend = LMEvalBackend()
            logger.info(f"Initialized benchmark: {self.name} (lm-eval)")
        elif config.backend == 'custom_eval':
            from .backends.custom_eval_backend import CustomEvalBackend
            self.backend = CustomEvalBackend()
            logger.info(f"Initialized benchmark: {self.name} (custom-eval)")
        else:
            from .backends.evalscope_backend import EvalScopeBackend
            self.backend = EvalScopeBackend()
            logger.info(f"Initialized benchmark: {self.name} (EvalScope)")

    def validate_target(self, target: BaseTarget) -> bool:
        if not self.REQUIRED_TARGET_TYPES:
            return True
        target_type_str = target.target_type.value if hasattr(target.target_type, 'value') else str(target.target_type)
        return target_type_str in self.REQUIRED_TARGET_TYPES

    @abstractmethod
    def evaluate(self, target: BaseTarget) -> BenchmarkResult:
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        pass