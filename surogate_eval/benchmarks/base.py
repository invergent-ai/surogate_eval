# surogate/eval/benchmarks/base.py
"""Base classes for benchmark evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Literal

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
    num_concurrent: Optional[int] = None
    system_prompt: Optional[str] = None
    judge_model: Optional[Dict[str, Any]] = None
    judge_criteria: Optional[str] = None
    eval_type: str = 'exact_match'


@dataclass
class BenchmarkResult:
    benchmark_name: str
    overall_score: float
    num_samples: int
    backend: str = "evalscope"
    task_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'benchmark_name': self.benchmark_name,
            'backend': self.backend,
            'overall_score': self.overall_score,
            'num_samples': self.num_samples,
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