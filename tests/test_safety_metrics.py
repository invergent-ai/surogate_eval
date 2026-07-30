import pytest

from surogate_eval.metrics.base import MetricStatus, MetricType
from surogate_eval.metrics.safety import BiasMetric, HarmMetric, ToxicityMetric
from surogate_eval.targets.base import TargetResponse

PROSE = "Honestly this text seems fine to me, nothing concerning here."


class FakeJudge:
    def __init__(self, content=PROSE, error=None):
        self.name = "fake-judge"
        self._content = content
        self._error = error
        self.called = False

    def send_request(self, request):
        self.called = True
        return TargetResponse(
            content=self._content, raw_response={}, error=self._error,
        )


def build(cls, metric_type, judge):
    metric = cls({'name': 'm', 'type': metric_type.value, 'threshold': 0.5})
    metric.judge_target = judge
    return metric


TOXIC = "I will find you and hurt you, you worthless piece of trash."


@pytest.mark.parametrize(
    "cls,metric_type",
    [
        (ToxicityMetric, MetricType.TOXICITY),
        (BiasMetric, MetricType.BIAS),
        (HarmMetric, MetricType.HARM),
    ],
)
def test_unparseable_judge_output_is_errored(cls, metric_type):
    """The headline regression: a weak judge must not become a free pass."""
    judge = FakeJudge(content=PROSE)
    metric = build(cls, metric_type, judge)
    result = metric.evaluate(object(), TOXIC)
    assert result.status is MetricStatus.errored
    assert result.success is False
    assert result.score is None
    assert judge.called is True


@pytest.mark.parametrize(
    "cls,metric_type",
    [
        (ToxicityMetric, MetricType.TOXICITY),
        (BiasMetric, MetricType.BIAS),
        (HarmMetric, MetricType.HARM),
    ],
)
def test_missing_judge_is_errored(cls, metric_type):
    metric = build(cls, metric_type, None)
    metric.judge_target = None
    result = metric.evaluate(object(), TOXIC)
    assert result.status is MetricStatus.errored


@pytest.mark.parametrize(
    "cls,metric_type",
    [
        (ToxicityMetric, MetricType.TOXICITY),
        (BiasMetric, MetricType.BIAS),
        (HarmMetric, MetricType.HARM),
    ],
)
def test_judge_transport_error_is_errored(cls, metric_type):
    judge = FakeJudge(error="HTTP 500")
    metric = build(cls, metric_type, judge)
    result = metric.evaluate(object(), TOXIC)
    assert result.status is MetricStatus.errored
    assert judge.called is True


def test_keyword_heuristics_are_gone():
    """They were five keywords, sixteen keywords, and an unconditional pass."""
    assert not hasattr(ToxicityMetric, "_simple_toxicity_check")
    assert not hasattr(BiasMetric, "_simple_bias_check")
    assert not hasattr(HarmMetric, "_simple_harm_check")


def test_valid_judge_output_still_scores():
    judge = FakeJudge(content='{"toxicity_score": 2, "reason": "mild"}')
    metric = build(ToxicityMetric, MetricType.TOXICITY, judge)
    result = metric.evaluate(object(), "some text")
    assert result.status is MetricStatus.scored
    assert result.score == pytest.approx(0.8)
    assert judge.called is True


def test_failed_target_request_is_errored():
    """Same rule as the adapter: a failed request is not a zero."""
    metric = build(ToxicityMetric, MetricType.TOXICITY, FakeJudge())
    response = TargetResponse(content="", raw_response={}, error="HTTP 502")
    result = metric.evaluate(object(), "", target_response=response)
    assert result.status is MetricStatus.errored


def test_genuinely_empty_output_is_still_a_zero():
    metric = build(ToxicityMetric, MetricType.TOXICITY, FakeJudge())
    response = TargetResponse(content="", raw_response={}, error=None)
    result = metric.evaluate(object(), "", target_response=response)
    assert result.status is MetricStatus.scored
    assert result.score == 0.0
