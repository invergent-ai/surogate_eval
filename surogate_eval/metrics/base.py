# surogate/eval/metrics/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from surogate_eval.datasets import TestCase, MultiTurnTestCase
from surogate_eval.targets import TargetResponse
from surogate_eval.utils.logger import get_logger

logger = get_logger()


class MetricType(Enum):
    """Types of metrics."""
    G_EVAL = "g_eval"
    CONVERSATIONAL_G_EVAL = "conversational_g_eval"
    MULTIMODAL_G_EVAL = "multimodal_g_eval"
    ARENA_G_EVAL = "arena_g_eval"
    DAG = "dag"
    CONVERSATIONAL_DAG = "conversational_dag"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TOKEN_GENERATION_SPEED = "token_generation_speed"

    # Multi-turn
    CONVERSATION_COHERENCE = "conversation_coherence"
    CONTEXT_RETENTION = "context_retention"
    TURN_ANALYSIS = "turn_analysis"

    # Safety
    TOXICITY = "toxicity"
    BIAS = "bias"
    HARM = "harm"

    # Non-LLM
    EMBEDDING_SIMILARITY = "embedding_similarity"
    CLASSIFICATION = "classification"


class MetricStatus(str, Enum):
    """Whether a result is a measurement or a failure to measure."""

    scored = "scored"
    errored = "errored"


@dataclass
class MetricResult:
    """Result from a metric evaluation."""

    metric_name: str
    metric_type: MetricType
    score: Optional[float]
    success: bool
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: MetricStatus = MetricStatus.scored

    @classmethod
    def errored(
            cls,
            *,
            metric_name: str,
            metric_type: MetricType,
            reason: str,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> "MetricResult":
        """Build a result that records a failure to measure.

        ``score`` is None rather than 0.0 so an error can never be
        averaged into a score (E-RUN-1).
        """
        return cls(
            metric_name=metric_name,
            metric_type=metric_type,
            score=None,
            success=False,
            reason=reason,
            metadata=metadata or {},
            status=MetricStatus.errored,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'metric_type': self.metric_type.value,
            'score': self.score,
            'success': self.success,
            'status': self.status.value,
            'reason': self.reason,
            'metadata': self.metadata
        }


@dataclass(frozen=True)
class _Aggregates:
    """Every figure derived from one partition of a batch's results."""

    scored_results: List[MetricResult]
    scored_n: int
    errored_n: int
    error_rate: float
    avg_score: float
    success_rate: float


@dataclass
class BatchMetricResult:
    """Results from batch evaluation."""

    metric_name: str
    metric_type: MetricType
    results: List[MetricResult]

    def _aggregate(self) -> _Aggregates:
        """Partition the results once and derive everything from that.

        Each property below used to rebuild the filtered list for itself, so
        a single ``to_dict()`` walked the results five times.
        """
        scored = [r for r in self.results if r.status is MetricStatus.scored]
        total = len(self.results)
        scored_n = len(scored)

        return _Aggregates(
            scored_results=scored,
            scored_n=scored_n,
            errored_n=total - scored_n,
            error_rate=((total - scored_n) / total) if total else 0.0,
            avg_score=(sum(r.score for r in scored) / scored_n) if scored_n else 0.0,
            success_rate=(
                sum(1 for r in scored if r.success) / scored_n
            ) if scored_n else 0.0,
        )

    @property
    def scored_results(self) -> List[MetricResult]:
        """Results that are a measurement rather than a failure."""
        return self._aggregate().scored_results

    @property
    def scored_n(self) -> int:
        return self._aggregate().scored_n

    @property
    def errored_n(self) -> int:
        return self._aggregate().errored_n

    @property
    def error_rate(self) -> float:
        return self._aggregate().error_rate

    @property
    def avg_score(self) -> float:
        """Average over what we could actually measure."""
        return self._aggregate().avg_score

    @property
    def success_rate(self) -> float:
        return self._aggregate().success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        aggregates = self._aggregate()
        return {
            'metric_name': self.metric_name,
            'metric_type': self.metric_type.value,
            'num_evaluations': len(self.results),
            'scored_n': aggregates.scored_n,
            'errored_n': aggregates.errored_n,
            'error_rate': aggregates.error_rate,
            'avg_score': aggregates.avg_score,
            'success_rate': aggregates.success_rate,
            'results': [r.to_dict() for r in self.results]
        }


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metric.

        Args:
            config: Metric configuration
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.metric_type = MetricType(config.get('type'))
        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        """Validate metric-specific configuration."""
        pass

    @abstractmethod
    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate a single test case.

        Args:
            test_case: Test case to evaluate
            actual_output: Actual output from target
            target_response: Full response from target (optional)

        Returns:
            Metric result
        """
        pass

    def evaluate_batch(
            self,
            test_cases: List[Union[TestCase, MultiTurnTestCase]],
            actual_outputs: List[str],
            target_responses: Optional[List[TargetResponse]] = None
    ) -> BatchMetricResult:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of test cases
            actual_outputs: List of actual outputs
            target_responses: List of full responses (optional)

        Returns:
            Batch metric result
        """
        if len(test_cases) != len(actual_outputs):
            raise ValueError("Number of test cases must match number of outputs")

        if target_responses and len(target_responses) != len(test_cases):
            raise ValueError("Number of target responses must match test cases")

        results = []
        for i, (test_case, actual_output) in enumerate(zip(test_cases, actual_outputs)):
            target_response = target_responses[i] if target_responses else None
            try:
                result = self.evaluate(test_case, actual_output, target_response)
            except Exception as e:
                # Safety net for E-RUN-1: a metric whose own handler misses an
                # exception must not take the rest of the batch down, and the
                # case must not disappear. It is recorded as unmeasured so the
                # run outcome still counts it.
                logger.error(f"Metric '{self.name}' raised on case {i}: {e}")
                result = MetricResult.errored(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    reason=f"Metric raised {type(e).__name__}: {e}",
                    metadata={'error_kind': type(e).__name__},
                )
            results.append(result)

        return BatchMetricResult(
            metric_name=self.name,
            metric_type=self.metric_type,
            results=results
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.metric_type.value})"


class LLMJudgeMetric(BaseMetric):
    """Base class for metrics that use LLM as judge."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge metric.

        Args:
            config: Metric configuration including judge_model
        """
        super().__init__(config)
        self.judge_model_config = config.get('judge_model', {})
        self.judge_target = None  # Will be set by runner

    def set_judge_target(self, judge_target):
        """Set the judge target for evaluation."""
        self.judge_target = judge_target

    def _no_output_result(
            self,
            target_response: Optional[TargetResponse] = None,
    ) -> MetricResult:
        """Result for an empty target output.

        A failed request is a failure to measure. An empty completion with
        no transport error is a real (bad) answer and stays a scored 0.0.

        Every judged metric needs this same distinction, so it lives here
        rather than being restated in each of them.
        """
        if target_response is not None and target_response.error:
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Target request failed: {target_response.error}",
                metadata={'error_kind': 'target'},
            )
        return MetricResult(
            metric_name=self.name,
            metric_type=self.metric_type,
            score=0.0,
            success=False,
            reason="No output to evaluate",
        )