"""Every metric family reports a failure as errored, never as a zero score.

The fail-closed rule (E-RUN-1) only holds if it holds everywhere: a metric
that crashes must not average 0.0 into the score, and must not be counted as
a real measurement. These tests drive the real metric classes - no stubbed
MetricResult, no hand-written dicts - and assert on what they return.
"""

from typing import Optional, Union

from surogate_eval.datasets.test_case import MultiTurnTestCase, TestCase
from surogate_eval.metrics.base import (
    BaseMetric,
    MetricResult,
    MetricStatus,
    MetricType,
)
from surogate_eval.metrics.embeddings import (
    ClassificationMetric,
    EmbeddingSimilarityMetric,
)
from surogate_eval.metrics.performance import (
    LatencyMetric,
    ThroughputMetric,
    TokenGenerationSpeedMetric,
)
from surogate_eval.targets.base import TargetResponse


def response(**kwargs) -> TargetResponse:
    kwargs.setdefault("content", "some answer")
    kwargs.setdefault("raw_response", {})
    return TargetResponse(**kwargs)


def assert_errored(result: MetricResult):
    """An errored result carries no score at all, so nothing can average it."""
    assert result.status is MetricStatus.errored
    assert result.score is None
    assert result.success is False
    assert result.to_dict()["status"] == "errored"


# --- performance ---------------------------------------------------------


def test_crashed_latency_metric_is_errored_not_zero():
    """``total_time`` that is not a number blows up mid-measurement."""
    metric = LatencyMetric({"name": "lat", "type": "latency"})

    result = metric.evaluate(
        TestCase(input="hi"), "hello", response(timing={"total_time": None})
    )

    assert_errored(result)
    assert result.metadata["error_kind"] == "TypeError"


def test_latency_without_timing_data_is_errored():
    """Nothing was timed, so nothing was measured."""
    metric = LatencyMetric({"name": "lat", "type": "latency"})

    assert_errored(metric.evaluate(TestCase(input="hi"), "hello", response()))


def test_latency_that_simply_exceeded_the_threshold_still_scores():
    """The failure path must not swallow a genuine slow-but-measured result."""
    metric = LatencyMetric({"name": "lat", "type": "latency", "threshold_ms": 10})

    result = metric.evaluate(
        TestCase(input="hi"), "hello", response(timing={"total_time": 5.0})
    )

    assert result.status is MetricStatus.scored
    assert result.score == 0.0
    assert result.success is False


def test_crashed_throughput_metric_is_errored_not_zero():
    metric = ThroughputMetric({"name": "tp", "type": "throughput"})

    result = metric.evaluate(
        TestCase(input="hi"), "hello", response(timing={"total_time": "quick"})
    )

    assert_errored(result)


def test_zero_latency_is_errored_for_token_speed():
    """A zero duration is flagged invalid by the metric itself."""
    metric = TokenGenerationSpeedMetric(
        {"name": "tps", "type": "token_generation_speed"}
    )

    result = metric.evaluate(
        TestCase(input="hi"), "hello", response(timing={"total_time": 0})
    )

    assert_errored(result)
    assert result.metadata["error_kind"] == "invalid_timing"


# --- embeddings ----------------------------------------------------------


def test_crashed_embedding_metric_is_errored_not_zero():
    """Mismatched dimensions make numpy raise inside the metric."""
    metric = EmbeddingSimilarityMetric(
        {"name": "emb", "type": "embedding_similarity"}
    )

    result = metric.evaluate(
        TestCase(input="hi", metadata={"expected_embedding": [1.0, 0.0]}),
        "",
        response(metadata={"embedding": [1.0, 0.0, 0.0]}),
    )

    assert_errored(result)
    assert result.metadata["error_kind"] == "ValueError"


def test_missing_embedding_is_errored():
    metric = EmbeddingSimilarityMetric(
        {"name": "emb", "type": "embedding_similarity"}
    )

    assert_errored(metric.evaluate(TestCase(input="hi"), "", response()))


def test_classification_without_an_expected_answer_is_errored():
    metric = ClassificationMetric({"name": "cls", "type": "classification"})

    assert_errored(metric.evaluate(TestCase(input="hi"), "cat", response()))


def test_a_wrong_classification_is_a_real_zero():
    """The one genuine scored zero in this family: the model answered, wrongly."""
    metric = ClassificationMetric({"name": "cls", "type": "classification"})

    result = metric.evaluate(
        TestCase(input="hi", expected_output="dog"), "cat", response()
    )

    assert result.status is MetricStatus.scored
    assert result.score == 0.0
    assert result.success is False


# --- the evaluate_batch net ---------------------------------------------


class ExplodingMetric(BaseMetric):
    """A metric with no error handling of its own, i.e. the next one written."""

    def _validate_config(self):
        pass

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None,
    ) -> MetricResult:
        raise RuntimeError("judge went away")


def test_evaluate_batch_turns_an_escaped_exception_into_an_errored_result():
    """Without the net this propagates and the whole batch is lost."""
    metric = ExplodingMetric({"name": "boom", "type": "toxicity"})

    batch = metric.evaluate_batch(
        [TestCase(input="a"), TestCase(input="b")], ["x", "y"]
    )

    assert len(batch.results) == 2
    for result in batch.results:
        assert_errored(result)
        assert result.metadata["error_kind"] == "RuntimeError"

    assert batch.scored_n == 0
    assert batch.errored_n == 2
    assert batch.error_rate == 1.0
    assert batch.avg_score == 0.0


def test_evaluate_batch_net_does_not_hide_a_partial_success():
    """One case crashing leaves the others measured."""

    class HalfExplodingMetric(ExplodingMetric):
        def evaluate(self, test_case, actual_output, target_response=None):
            if actual_output == "boom":
                raise ValueError("nope")
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=1.0,
                success=True,
            )

    metric = HalfExplodingMetric({"name": "half", "type": "toxicity"})

    batch = metric.evaluate_batch(
        [TestCase(input="a"), TestCase(input="b")], ["ok", "boom"]
    )

    assert (batch.scored_n, batch.errored_n) == (1, 1)
    assert batch.avg_score == 1.0
