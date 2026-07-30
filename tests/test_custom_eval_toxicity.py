"""The custom-eval toxicity path must not hand out a free pass.

`except Exception: is_safe = True` turned a dead judge into score 1.0 and
success True for every row, which is the same fail-open the metrics just
lost. No network: the deepeval metric and the target are both fakes.
"""

import pytest

from surogate_eval.benchmarks.backends.custom_eval_backend import CustomEvalBackend
from surogate_eval.targets.base import TargetResponse

ROWS = [
    {"instruction": "Say something", "answer": "something"},
    {"instruction": "Say something else", "answer": "something else"},
]


class FakeTarget:
    name = "t1"

    def send_request(self, request):
        return TargetResponse(content="a polite answer", raw_response={}, error=None)


class FakeToxicityMetric:
    """Stands in for deepeval's ToxicityMetric."""

    instances = []

    def __init__(self, score=0.1, raises=None, **kwargs):
        self.threshold = kwargs.get("threshold", 0.5)
        self.reason = "because"
        self._score = score
        self._raises = raises
        self.score = None
        FakeToxicityMetric.instances.append(self)

    def measure(self, test_case):
        if self._raises is not None:
            raise self._raises
        self.score = self._score


@pytest.fixture
def fake_metric(monkeypatch):
    def install(**kwargs):
        FakeToxicityMetric.instances.clear()
        monkeypatch.setattr(
            "deepeval.metrics.ToxicityMetric",
            lambda **metric_kwargs: FakeToxicityMetric(**{**metric_kwargs, **kwargs}),
        )

    return install


def run_rows(fake_metric, **kwargs):
    fake_metric(**kwargs)
    backend = CustomEvalBackend()
    return backend._evaluate_toxicity_rows(ROWS, FakeTarget(), {}, {}, None)


def test_broken_judge_is_not_a_safe_verdict(fake_metric):
    results = run_rows(fake_metric, raises=RuntimeError("judge 500"))
    assert [r["status"] for r in results] == ["errored", "errored"]
    assert all(r["score"] is None for r in results)
    assert all(r["success"] is False for r in results)


def test_null_judge_score_is_not_a_safe_verdict(fake_metric):
    results = run_rows(fake_metric, score=None)
    assert [r["status"] for r in results] == ["errored", "errored"]
    assert all(r["score"] is None for r in results)


def test_safe_rows_still_score(fake_metric):
    results = run_rows(fake_metric, score=0.1)
    assert [r["status"] for r in results] == ["scored", "scored"]
    assert all(r["score"] == 1.0 and r["success"] for r in results)


def test_toxic_rows_still_score(fake_metric):
    results = run_rows(fake_metric, score=0.9)
    assert [r["status"] for r in results] == ["scored", "scored"]
    assert all(r["score"] == 0.0 and r["success"] is False for r in results)
