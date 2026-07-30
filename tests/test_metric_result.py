from surogate_eval.metrics.base import (
    BatchMetricResult,
    MetricResult,
    MetricStatus,
    MetricType,
)


def scored(value: float, success: bool = True) -> MetricResult:
    return MetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        score=value,
        success=success,
    )


def errored() -> MetricResult:
    return MetricResult.errored(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        reason="judge exploded",
    )


def test_result_defaults_to_scored():
    assert scored(1.0).status is MetricStatus.scored


def test_errored_result_has_no_score():
    """None, not 0.0, so an error can never be averaged in by accident."""
    r = errored()
    assert r.status is MetricStatus.errored
    assert r.score is None
    assert r.success is False


def test_to_dict_carries_status():
    assert scored(1.0).to_dict()["status"] == "scored"
    assert errored().to_dict()["status"] == "errored"


def test_avg_score_excludes_errored():
    """A judge outage must not drag the score down (E-RUN-1)."""
    batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[scored(1.0), scored(0.6), errored(), errored()],
    )
    assert batch.avg_score == 0.8
    assert batch.scored_n == 2
    assert batch.errored_n == 2
    assert batch.error_rate == 0.5


def test_success_rate_excludes_errored():
    batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[scored(1.0, True), scored(0.0, False), errored()],
    )
    assert batch.success_rate == 0.5


def test_all_errored_batch_does_not_divide_by_zero():
    batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[errored(), errored()],
    )
    assert batch.avg_score == 0.0
    assert batch.success_rate == 0.0
    assert batch.error_rate == 1.0


def test_empty_batch_has_zero_error_rate():
    batch = BatchMetricResult(
        metric_name="m", metric_type=MetricType.TOXICITY, results=[],
    )
    assert batch.error_rate == 0.0


def test_batch_to_dict_reports_counts():
    batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[scored(1.0), errored()],
    )
    d = batch.to_dict()
    assert d["scored_n"] == 1
    assert d["errored_n"] == 1
    assert d["error_rate"] == 0.5
