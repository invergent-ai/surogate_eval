"""A target that never answered must not read as a target that answered
wrongly, in the custom-eval exact_match and judge paths.

`_evaluate_exact_match_direct` and `_evaluate_judge_rows` used to catch a
failed or raising `send_request` and record it as a scored 0.0 row - the
same confusion `_evaluate_toxicity_rows` was fixed for elsewhere on this
branch, and the same fake zero that gets averaged into the benchmark score.
No network: the targets and the G-Eval judge are both fakes.

The same rule covers the two failures that are not the target's: a row that
cannot be compared and a judge that breaks. Both are failures to measure,
so both produce one errored row and leave the rest of the benchmark alone.
"""

import pytest

from surogate_eval.benchmarks.backends import custom_eval_backend as ceb
from surogate_eval.benchmarks.backends.custom_eval_backend import CustomEvalBackend
from surogate_eval.targets.base import TargetResponse

ROWS = [
    {"instruction": "Say something", "answer": "something", "_original_idx": 0},
    {"instruction": "Say something else", "answer": "something else", "_original_idx": 1},
]


class FakeTarget:
    name = "t1"
    config = {}

    def send_request(self, request):
        return TargetResponse(content="something", raw_response={}, error=None)


class UnreachableTarget:
    """A target whose request came back carrying an error."""

    name = "t1"
    config = {}

    def send_request(self, request):
        return TargetResponse(content=None, raw_response={}, error="connection reset")


class RaisingTarget:
    """A target whose client blew up before a response existed."""

    name = "t1"
    config = {}

    def send_request(self, request):
        raise RuntimeError("connection reset")


class EmptyTarget:
    """A target that answered, with nothing. Not the same thing."""

    name = "t1"
    config = {}

    def send_request(self, request):
        return TargetResponse(content="", raw_response={}, error=None)


# --- exact_match -----------------------------------------------------------

@pytest.mark.parametrize("target", [UnreachableTarget(), RaisingTarget()])
def test_exact_match_unanswered_target_is_errored_not_scored_zero(target):
    """The failed request used to be recorded as score=0.0/success=False -
    indistinguishable from a target that answered and got it wrong."""
    backend = CustomEvalBackend()
    results = backend._evaluate_exact_match_rows(ROWS, target, {}, {})

    assert [r["status"] for r in results] == ["errored", "errored"]
    assert all(r["score"] is None for r in results)
    assert all(r["success"] is False for r in results)
    assert all("connection reset" in r["reason"] for r in results)


def test_exact_match_empty_answer_is_still_scored():
    """The other half of the rule: an empty completion with no transport
    error is a real (bad) answer and is still scored, not errored."""
    backend = CustomEvalBackend()
    results = backend._evaluate_exact_match_rows(ROWS, EmptyTarget(), {}, {})

    assert [r["status"] for r in results] == ["scored", "scored"]
    assert all(r["score"] == 0.0 and r["success"] is False for r in results)


def test_exact_match_healthy_target_still_scores():
    backend = CustomEvalBackend()
    results = backend._evaluate_exact_match_rows(ROWS, FakeTarget(), {}, {})

    assert [r["status"] for r in results] == ["scored", "scored"]
    assert results[0]["success"] is True  # "something" matches row 0's answer


# --- rows that cannot be compared ------------------------------------------

NUMERIC_ROWS = [
    {"instruction": "What is 6 * 7?", "answer": 42, "_original_idx": 0},
    {"instruction": "Say something", "answer": "something", "_original_idx": 1},
]


def test_exact_match_row_that_cannot_be_compared_errors_only_that_row():
    """A numeric ``answer`` column is inferred as int64, so ``expected``
    arrives as an int and ``_normalize_output`` calls ``.strip()`` on it.
    Left outside the protected region the AttributeError escaped the row
    loop and the whole benchmark - every row already measured with it."""
    backend = CustomEvalBackend()
    results = backend._evaluate_exact_match_rows(NUMERIC_ROWS, FakeTarget(), {}, {})

    assert [r["status"] for r in results] == ["errored", "scored"]
    assert results[0]["score"] is None
    assert results[0]["success"] is False
    # The healthy row is still measured: "something" matches row 1's answer.
    assert results[1]["score"] == 1.0


def test_numeric_answer_column_benchmark_still_reports_counts(tmp_path):
    """End to end through ``evaluate()`` with a numeric answer column, the
    shape any numeric-QA dataset has. The benchmark must come back with its
    counts, not raise and be flattened into a status-only failure node."""
    import json

    dataset_path = tmp_path / "numeric.jsonl"
    rows = [
        {"instruction": "What is 6 * 7?", "answer": 42},
        {"instruction": "What is 1 + 1?", "answer": 2},
    ]
    with open(dataset_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    backend = CustomEvalBackend()
    result = backend.evaluate(
        FakeTarget(),
        "numeric_bench",
        {"source": str(dataset_path), "eval_type": "exact_match"},
    )

    em = result["task_results"]["exact_match"]
    assert (em["scored_n"], em["errored_n"]) == (0, 2)
    assert all(r["score"] is None for r in result["detailed_results"])


# --- judge -------------------------------------------------------------


class FakeGEval:
    """Stands in for deepeval's GEval."""

    instances = []

    def __init__(self, score=0.9, raises=None, **kwargs):
        self.reason = "because"
        self.score = None
        self._score = score
        self._raises = raises
        FakeGEval.instances.append(self)

    def measure(self, test_case, _show_indicator=False):
        if self._raises is not None:
            raise self._raises
        self.score = self._score


@pytest.fixture
def fake_geval(monkeypatch):
    def install(**kwargs):
        FakeGEval.instances.clear()
        monkeypatch.setattr(
            ceb, "GEval", lambda **geval_kwargs: FakeGEval(**{**geval_kwargs, **kwargs})
        )

    return install


def run_judge_rows(fake_geval, target=None, **kwargs):
    fake_geval(**kwargs)
    backend = CustomEvalBackend()
    return backend._evaluate_judge_rows(ROWS, target or FakeTarget(), {}, {}, None)


@pytest.mark.parametrize("target", [UnreachableTarget(), RaisingTarget()])
def test_judge_unanswered_target_is_errored_not_scored_zero(fake_geval, target):
    """Same bug, judge path: the failed request used to be recorded as a
    scored 0.0 row for a target that never said anything, and the judge was
    never consulted about it."""
    results = run_judge_rows(fake_geval, target=target, score=0.9)

    assert [r["status"] for r in results] == ["errored", "errored"]
    assert all(r["score"] is None for r in results)
    assert all(r["success"] is False for r in results)
    assert all("connection reset" in r["reason"] for r in results)
    # The judge was never asked to score a response that does not exist.
    assert FakeGEval.instances == []


def test_judge_empty_answer_is_still_judged(fake_geval):
    results = run_judge_rows(fake_geval, target=EmptyTarget(), score=0.9)

    assert [r["status"] for r in results] == ["scored", "scored"]
    assert all(r["score"] == 0.9 for r in results)


def test_judge_healthy_target_still_scores(fake_geval):
    results = run_judge_rows(fake_geval, target=FakeTarget(), score=0.9)

    assert [r["status"] for r in results] == ["scored", "scored"]
    assert all(r["score"] == 0.9 and r["success"] for r in results)


# --- summary counts feeding outcome.py --------------------------------------

def test_mixed_evaluate_reports_errors_in_scored_n_errored_n(monkeypatch, tmp_path):
    """The summary counts evaluate() feeds outcome.py must reflect the
    errored rows, not flatten them into an 'accuracy'/'avg_score' that
    silently absorbed them as zeros."""
    FakeGEval.instances.clear()
    monkeypatch.setattr(ceb, "GEval", lambda **kwargs: FakeGEval(score=1.0, **kwargs))

    dataset_path = tmp_path / "mixed.jsonl"
    import json
    rows = [
        {"instruction": "q1", "answer": "a1", "eval_type": "exact_match"},
        {"instruction": "q2", "answer": "a2", "eval_type": "exact_match"},
        {"instruction": "q3", "answer": "a3", "eval_type": "judge"},
        {"instruction": "q4", "answer": "a4", "eval_type": "judge"},
    ]
    with open(dataset_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    class MixedTarget:
        name = "t1"
        config = {}
        calls = 0

        def send_request(self, request):
            MixedTarget.calls += 1
            # Every other call fails - one exact_match row and one judge row.
            if MixedTarget.calls % 2 == 0:
                return TargetResponse(content=None, raw_response={}, error="connection reset")
            return TargetResponse(content="a match", raw_response={}, error=None)

    backend = CustomEvalBackend()
    result = backend.evaluate(
        MixedTarget(),
        "mixed_bench",
        {"source": str(dataset_path), "eval_type": "hybrid"},
    )

    em = result["task_results"]["exact_match"]
    judge = result["task_results"]["judge"]

    assert (em["scored_n"], em["errored_n"]) == (1, 1)
    assert (judge["scored_n"], judge["errored_n"]) == (1, 1)

    # None of the errored rows contributed a fake zero to the rates.
    assert em["accuracy"] >= 0.0  # would ZeroDivisionError/skew if miscounted
    assert judge["avg_score"] == 1.0  # the one scored judge row, not diluted by the errored one
