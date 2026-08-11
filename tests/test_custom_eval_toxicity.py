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


class UnreachableTarget:
    """A target whose request came back carrying an error."""

    name = "t1"

    def send_request(self, request):
        return TargetResponse(content=None, raw_response={}, error="connection reset")


class RaisingTarget:
    """A target whose client blew up before a response existed."""

    name = "t1"

    def send_request(self, request):
        raise RuntimeError("connection reset")


class EmptyTarget:
    """A target that answered, with nothing. Not the same thing."""

    name = "t1"

    def send_request(self, request):
        return TargetResponse(content="", raw_response={}, error=None)


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


def run_rows(fake_metric, target=None, **kwargs):
    fake_metric(**kwargs)
    backend = CustomEvalBackend()
    return backend._evaluate_toxicity_rows(
        ROWS, target or FakeTarget(), {}, {}, None
    )


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


@pytest.mark.parametrize("target", [UnreachableTarget(), RaisingTarget()])
def test_a_target_that_never_answered_is_not_judged_at_all(fake_metric, target):
    """The failed request used to be swallowed into output='', and the judge
    duly found the empty string inoffensive: score 1.0, success True, status
    scored, for a target that never said anything."""
    results = run_rows(fake_metric, target=target, score=0.1)

    assert [r["status"] for r in results] == ["errored", "errored"]
    assert all(r["score"] is None for r in results)
    assert all(r["success"] is False for r in results)
    assert all("connection reset" in r["reason"] for r in results)
    # The judge was never asked about a response that does not exist.
    assert FakeToxicityMetric.instances[0].score is None


def test_an_empty_answer_is_still_an_answer(fake_metric):
    """The other half of the shared rule (LLMJudgeMetric._no_output_result):
    an empty completion with no transport error is a real, bad answer and is
    still judged rather than errored."""
    results = run_rows(fake_metric, target=EmptyTarget(), score=0.1)

    assert [r["status"] for r in results] == ["scored", "scored"]
    assert all(r["score"] == 1.0 for r in results)


def test_toxicity_loop_reports_progress(fake_metric, monkeypatch):
    """The toxicity loop has the same row-append shape as the exact_match
    and judge loops, but used to skip their report_rows call entirely, so a
    toxicity benchmark showed no live progress. The unconditional post-loop
    write must land with the right rows_done/rows_total/scored/passed/
    score_sum."""
    report_calls = []

    def fake_report_rows(rows_done, rows_total, scored, passed, score_sum):
        report_calls.append({
            "rows_done": rows_done,
            "rows_total": rows_total,
            "scored": scored,
            "passed": passed,
            "score_sum": score_sum,
        })

    monkeypatch.setattr(
        "surogate_eval.benchmarks.backends.custom_eval_backend.runners.report_rows",
        fake_report_rows,
    )

    results = run_rows(fake_metric, score=0.1)

    assert report_calls, "report_rows was never called"
    final_call = report_calls[-1]
    assert final_call["rows_done"] == len(results) == 2
    assert final_call["rows_total"] == len(ROWS) == 2
    assert final_call["scored"] == 2
    assert final_call["passed"] == 2
    assert final_call["score_sum"] == 2.0
