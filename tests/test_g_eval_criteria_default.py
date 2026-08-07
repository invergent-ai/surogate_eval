"""The G-Eval metrics carry the same blank-criteria trap as the custom_eval
judge path, in a second and independent GEval-construction site.

It fails more quietly than the original did. `MetricRegistry.create_metrics`
wraps construction in a broad `except`, logs, and DROPS the metric from the
run, so a config asking for three metrics silently produces two. No errored
row, no failed benchmark, nothing in the report.

Not reachable from ops, which only ever emits `backend: custom_eval`.
Reachable from hand-written YAML and the CLI, where it is a documented
first-class feature used throughout `examples/config.yaml`.

No network: only the metric's parameter assembly is exercised.
"""

import pytest

from surogate_eval.metrics.adapters import deepeval_adapter
from surogate_eval.metrics.g_eval import (
    ConversationalGEvalMetric,
    GEvalMetric,
    MultimodalGEvalMetric,
)

CASES = [
    (GEvalMetric, "Correctness"),
    (ConversationalGEvalMetric, "Coherence"),
]

# MultimodalGEvalMetric carries the same two sites (`g_eval.py:149`, `:160`)
# and gets the same fix, but it cannot be constructed against the installed
# deepeval at all: the adapter raises ImportError because MLLMTestCase support
# is missing. Left untested here rather than tested through a fake so
# thoroughly stubbed it would only assert the fake.
assert MultimodalGEvalMetric is not None


class FakeGEval:
    """Stands in for deepeval's GEval, rejecting blank criteria the way the
    real one does (`criteria is not None and not criteria.strip()`).

    Needed because the real one builds a GPTModel in its constructor and
    wants an API key, which would make this a network-shaped test.
    """

    def __init__(self, **kwargs):
        criteria = kwargs.get("criteria")
        if criteria is None or not criteria.strip():
            raise ValueError("Criteria provided cannot be an empty string.")
        self.criteria = criteria


@pytest.fixture(autouse=True)
def fake_geval(monkeypatch):
    for name in ("GEval", "ConversationalGEval", "MultimodalGEval"):
        if hasattr(deepeval_adapter, name):
            monkeypatch.setattr(deepeval_adapter, name, FakeGEval)


@pytest.mark.parametrize("metric_cls,default", CASES)
@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_criteria_falls_back_to_the_metrics_own_default(
    metric_cls, default, blank
):
    """`config.get('criteria', DEFAULT)` returns the default only when the key
    is ABSENT, so a present-but-blank criteria reaches deepeval, which raises
    on all three of these."""
    metric = metric_cls({"name": "m", "criteria": blank})

    assert metric.config["parameters"]["criteria"] == default


@pytest.mark.parametrize("metric_cls,default", CASES)
def test_a_named_criteria_is_not_replaced_by_the_default(metric_cls, default):
    """The allow direction."""
    metric = metric_cls({"name": "m", "criteria": "Reward a citation."})

    assert metric.config["parameters"]["criteria"] == "Reward a citation."
