"""Conversation metrics must fail closed, exactly as the safety metrics do.

Mirrors tests/test_safety_metrics.py. The old behaviour scored 0.7/0.75 with
success=True from response length alone whenever the judge was missing,
unreachable or unparseable, so a judge outage produced a passing score.
"""

import pytest

from surogate_eval.datasets.test_case import MultiTurnTestCase, Turn
from surogate_eval.metrics.base import MetricStatus, MetricType
from surogate_eval.metrics.conversation import (
    ContextRetentionMetric,
    ConversationCoherenceMetric,
    TurnAnalysisMetric,
)
from surogate_eval.targets.base import TargetResponse

PROSE = "Sounds like a perfectly reasonable conversation to me."

# Parses cleanly for all three metrics, so any errored result it produces
# comes from the transport guard rather than the parse handler.
PARSEABLE = (
    '{"coherence_score": 8, "retention_score": 8, "quality_score": 8, '
    '"reason": "fine"}'
)

OUTPUT = "Yes, you told me earlier that you live in Cluj."

METRICS = [
    (ConversationCoherenceMetric, MetricType.CONVERSATION_COHERENCE),
    (ContextRetentionMetric, MetricType.CONTEXT_RETENTION),
    (TurnAnalysisMetric, MetricType.TURN_ANALYSIS),
]


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


def conversation():
    return MultiTurnTestCase(turns=[
        Turn(role="user", content="I live in Cluj."),
        Turn(role="assistant", content="Noted."),
        Turn(role="user", content="Do you remember where I live?"),
    ])


def build(cls, metric_type, judge):
    metric = cls({'name': 'm', 'type': metric_type.value})
    metric.judge_target = judge
    return metric


@pytest.mark.parametrize("cls,metric_type", METRICS)
def test_unparseable_judge_output_is_errored(cls, metric_type):
    """The headline regression: a weak judge must not become a free pass."""
    judge = FakeJudge(content=PROSE)
    metric = build(cls, metric_type, judge)
    result = metric.evaluate(conversation(), OUTPUT)
    assert result.status is MetricStatus.errored
    assert result.success is False
    assert result.score is None
    assert result.metadata['error_kind'] == 'JudgeParseError'
    assert judge.called is True


@pytest.mark.parametrize("cls,metric_type", METRICS)
def test_missing_judge_is_errored(cls, metric_type):
    metric = build(cls, metric_type, None)
    result = metric.evaluate(conversation(), OUTPUT)
    assert result.status is MetricStatus.errored
    assert result.score is None
    assert result.metadata['error_kind'] == 'no_judge'


@pytest.mark.parametrize("cls,metric_type", METRICS)
def test_judge_transport_error_is_errored(cls, metric_type):
    """The content is deliberately parseable: without the transport guard
    this would score, so the test pins the guard and not the parse handler."""
    judge = FakeJudge(content=PARSEABLE, error="HTTP 500")
    metric = build(cls, metric_type, judge)
    result = metric.evaluate(conversation(), OUTPUT)
    assert result.status is MetricStatus.errored
    assert result.metadata['error_kind'] == 'JudgeUnavailableError'
    assert judge.called is True


@pytest.mark.parametrize("cls,metric_type", METRICS)
def test_valid_judge_output_still_scores(cls, metric_type):
    judge = FakeJudge(content=PARSEABLE)
    metric = build(cls, metric_type, judge)
    result = metric.evaluate(conversation(), OUTPUT)
    assert result.status is MetricStatus.scored
    assert result.score == pytest.approx(0.8)
    assert judge.called is True


@pytest.mark.parametrize("cls,metric_type", METRICS)
def test_failed_target_request_is_errored(cls, metric_type):
    """Same rule as safety.py: a failed request is not a zero."""
    metric = build(cls, metric_type, FakeJudge(content=PARSEABLE))
    response = TargetResponse(content="", raw_response={}, error="HTTP 502")
    result = metric.evaluate(conversation(), "", target_response=response)
    assert result.status is MetricStatus.errored
    assert result.metadata['error_kind'] == 'target'


@pytest.mark.parametrize("cls,metric_type", METRICS)
def test_genuinely_empty_output_is_still_a_zero(cls, metric_type):
    metric = build(cls, metric_type, FakeJudge(content=PARSEABLE))
    response = TargetResponse(content="", raw_response={}, error=None)
    result = metric.evaluate(conversation(), "", target_response=response)
    assert result.status is MetricStatus.scored
    assert result.score == 0.0


def test_length_heuristics_are_gone():
    """They scored 0.7/0.75 with success=True from response length alone."""
    assert not hasattr(ConversationCoherenceMetric, "_simple_coherence_check")
    assert not hasattr(ContextRetentionMetric, "_simple_retention_check")
    assert not hasattr(TurnAnalysisMetric, "_simple_turn_analysis")


@pytest.mark.parametrize("cls,metric_type", METRICS)
def test_single_turn_test_case_is_errored_not_scored_zero(cls, metric_type):
    """A metric fed the wrong shape of test case never measured anything.
    Same E-RUN-1 class as the deepeval adapter's conversational-metric-on-
    single-turn-dataset guard: this is a configuration failure, not a
    measurement, so it must not average a 0.0 into the score."""
    metric = build(cls, metric_type, FakeJudge(content=PARSEABLE))
    result = metric.evaluate(object(), OUTPUT)
    assert result.status is MetricStatus.errored
    assert result.success is False
    assert result.score is None
    assert result.metadata['error_kind'] == 'config'
