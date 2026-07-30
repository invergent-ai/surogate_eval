import pytest

from surogate_eval.datasets.test_case import TestCase
from surogate_eval.errors import JudgeUnavailableError
from surogate_eval.metrics.base import MetricStatus, MetricType
from surogate_eval.metrics.adapters.deepeval_adapter import DeepEvalAdapter
from surogate_eval.targets.base import TargetResponse


class Boom:
    """A deepeval metric stand-in that fails the way a broken judge does."""

    def __init__(self, exc):
        self.exc = exc
        self.called = False

    def measure(self, test_case, _show_indicator=False):
        self.called = True
        raise self.exc


def make_adapter(deepeval_metric):
    adapter = DeepEvalAdapter.__new__(DeepEvalAdapter)
    adapter.name = "correctness"
    adapter.metric_type = MetricType.G_EVAL
    adapter.config = {"deepeval_metric_type": "g_eval"}
    adapter.deepeval_metric = deepeval_metric
    adapter._judge_target = None
    adapter.is_conversational = False
    adapter.is_multimodal = False
    return adapter


def test_judge_error_is_errored_not_zero():
    boom = Boom(JudgeUnavailableError("judge 500"))
    adapter = make_adapter(boom)
    result = adapter.evaluate(TestCase(input="What is 2+2?"), "some model output")
    assert boom.called is True
    assert result.status is MetricStatus.errored
    assert result.score is None
    # Must be labelled as a judge failure, not folded into the generic
    # internal-error branch. If `except Exception` is ever moved ahead of
    # `except JudgeError`, this is what catches it: both branches return
    # status=errored/score=None, so only the error_kind tells them apart.
    assert result.metadata.get("error_kind") == "JudgeUnavailableError"


def test_internal_error_is_errored_and_labelled():
    """A bug reaching measure() (not our own dispatch code) must still be
    labelled internal, not mistaken for a judge problem."""
    boom = Boom(AttributeError("'Steps' object has no attribute 'steps'"))
    adapter = make_adapter(boom)
    result = adapter.evaluate(TestCase(input="What is 2+2?"), "some model output")
    assert boom.called is True
    assert result.status is MetricStatus.errored
    assert result.metadata.get("error_kind") == "internal"


def test_failed_target_request_is_errored():
    adapter = make_adapter(Boom(RuntimeError("unreachable")))
    response = TargetResponse(content="", raw_response={}, error="HTTP 502")
    result = adapter.evaluate(object(), "", target_response=response)
    assert result.status is MetricStatus.errored
    assert "502" in result.reason


def test_genuinely_empty_completion_is_still_a_zero():
    """An empty answer with no transport error is a real bad answer."""
    adapter = make_adapter(Boom(RuntimeError("should not be reached")))
    response = TargetResponse(content="", raw_response={}, error=None)
    result = adapter.evaluate(object(), "", target_response=response)
    assert result.status is MetricStatus.scored
    assert result.score == 0.0
