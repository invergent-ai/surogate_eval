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

# A numeric ``answer`` column no longer makes a row uncomparable: the
# matcher coerces ``expected`` with ``str()`` before comparing (see
# ``surogate_eval/benchmarks/matching.py``), so an int answer is just
# compared as its string form. What can still fail to measure is the
# comparison itself: a ``regex`` pattern that catastrophically backtracks on
# one row's output raises ``MatchTimeout``, and that is still a failure to
# measure rather than a wrong answer.
CATASTROPHIC_PATTERN = r"(a+)+$"

TIMEOUT_ROWS = [
    {"instruction": "explode", "answer": "boom", "_original_idx": 0},
    {"instruction": "Say aaaa", "answer": "aaaa", "_original_idx": 1},
]


class RowAwareTarget:
    """Row 0 gets an answer built to blow up catastrophic backtracking under
    the shared pattern; row 1 gets a plain answer the same pattern matches
    immediately, so it stays scored."""

    name = "t1"
    config = {}

    def send_request(self, request):
        if request.prompt == "explode":
            return TargetResponse(content="a" * 5000 + "!", raw_response={}, error=None)
        return TargetResponse(content="aaaa", raw_response={}, error=None)


def test_exact_match_row_that_cannot_be_compared_errors_only_that_row():
    """A row whose output makes the configured pattern catastrophically
    backtrack cannot be measured in time: ``Matcher.compare`` raises
    ``MatchTimeout``, caught by the same ``except Exception`` that turns any
    other comparison failure into an errored row. Left unguarded, that
    exception would escape the row loop and the whole benchmark - every row
    already measured with it. The row next to it, matched by the same
    pattern in the ordinary way, is untouched and still scored."""
    backend = CustomEvalBackend()
    config = {"matcher": {"mode": "regex", "pattern": CATASTROPHIC_PATTERN, "timeout": 0.1}}
    results = backend._evaluate_exact_match_rows(TIMEOUT_ROWS, RowAwareTarget(), config, {})

    assert [r["status"] for r in results] == ["errored", "scored"]
    assert results[0]["score"] is None
    assert results[0]["success"] is False
    # The healthy row is still measured: the pattern matches "aaaa" outright.
    assert results[1]["score"] == 1.0


def test_numeric_answer_column_benchmark_still_reports_counts(tmp_path):
    """End to end through ``evaluate()`` with a numeric answer column, the
    shape any numeric-QA dataset has. A numeric answer is compared by
    coercing it to a string rather than erroring, so the benchmark comes
    back with its rows scored - not flattened into a status-only failure
    node, and not errored either."""
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
    assert (em["scored_n"], em["errored_n"]) == (2, 0)
    # FakeTarget always answers "something", which contains neither "42" nor
    # "2" under the default `contains` mode - both rows are measured, and
    # both are wrong.
    assert all(r["score"] == 0.0 for r in result["detailed_results"])


# --- judge -------------------------------------------------------------


class FakeGEval:
    """Stands in for deepeval's GEval."""

    instances = []

    def __init__(self, score=0.9, raises=None, **kwargs):
        # Reject blank criteria the way the real one does. deepeval's own
        # validator is `criteria is not None and not criteria.strip()`, so
        # "   " is as fatal as "" and as None. A fake that accepted anything
        # truthy is why the whitespace case looked covered: the tests below
        # passed against a stand-in more permissive than the library, which
        # is the same way the original bug hid.
        criteria = kwargs.get("criteria")
        if criteria is None or not criteria.strip():
            raise ValueError("Criteria provided cannot be an empty string.")

        self.reason = "because"
        self.score = None
        self._score = score
        self._raises = raises
        # Kept so a test can assert what GEval was actually built with, not
        # only what it returned.
        self.criteria = criteria
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


def run_judge_rows(fake_geval, target=None, config=None, **kwargs):
    fake_geval(**kwargs)
    backend = CustomEvalBackend()
    return backend._evaluate_judge_rows(
        ROWS, target or FakeTarget(), config or {}, {}, None,
    )


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


def test_judge_that_breaks_errors_the_row_instead_of_scoring_it_zero(fake_geval):
    """A judge that raises is a failure to measure, not a target that
    answered badly. This branch recorded it as a scored 0.0, so
    ``errored_n`` stayed at zero while ``avg_score`` and the benchmark's
    ``overall_score`` were dragged down by fake zeroes.
    ``_evaluate_toxicity_rows`` already treats a broken judge this way."""
    results = run_judge_rows(fake_geval, raises=RuntimeError("judge exploded"))

    assert [r["status"] for r in results] == ["errored", "errored"]
    assert all(r["score"] is None for r in results)
    assert all(r["success"] is False for r in results)
    assert all("judge exploded" in r["reason"] for r in results)
    # The target answered: what failed is the judge, and its answer is kept.
    assert all(r["raw_output"] == "something" for r in results)


def test_judge_errors_are_reported_in_the_summary_counts(monkeypatch, tmp_path):
    """The counts outcome.py reads must show the judge failure. A scored 0.0
    left ``errored_n`` at zero, so the run exited 0 on a dead judge."""
    import json

    FakeGEval.instances.clear()
    monkeypatch.setattr(
        ceb, "GEval",
        lambda **kwargs: FakeGEval(raises=RuntimeError("judge exploded"), **kwargs),
    )

    dataset_path = tmp_path / "judge.jsonl"
    with open(dataset_path, "w") as f:
        for row in [{"instruction": "q1", "answer": "a1"}, {"instruction": "q2", "answer": "a2"}]:
            f.write(json.dumps(row) + "\n")

    backend = CustomEvalBackend()
    result = backend.evaluate(
        FakeTarget(), "judge_bench", {"source": str(dataset_path), "eval_type": "judge"}
    )

    judge = result["task_results"]["judge"]
    assert (judge["scored_n"], judge["errored_n"]) == (0, 2)
    # No rate at all, rather than a rate of zero. `avg_score: 0.0` is
    # indistinguishable from a judge that scored every answer badly, and the
    # report's "not measured" branch keys on the metric being absent, so
    # filling it in was what stopped that branch ever firing.
    assert "avg_score" not in judge
    assert "success_rate" not in judge
    assert result["overall_score"] == 0.0
    # Only the task that had rows is reported: this benchmark has no
    # exact-match rows, so there is no exact-match task to speak of.
    assert set(result["task_results"]) == {"judge"}


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


# --- judge criteria default --------------------------------------------


def _criteria_seen():
    """The ``criteria`` GEval was constructed with, one per row."""
    return [g.criteria for g in FakeGEval.instances]


@pytest.mark.parametrize("config", [
    # What GenericBenchmark.evaluate used to build: every key present, and
    # `judge_criteria` None whenever the YAML omitted it. It now drops unset
    # keys, but the guard here still earns its place for the empty-string
    # case below, and this pins the shape that broke.
    {"judge_criteria": None},
    # A form that posts an empty field.
    {"judge_criteria": ""},
    # And the same form with a space in it. deepeval rejects whitespace
    # exactly as it rejects empty (`not criteria.strip()` in its own
    # validator), but a whitespace string is truthy, so a plain `or` sails
    # past it into the identical every-row failure.
    {"judge_criteria": "   "},
    {"judge_criteria": "\n\t "},
    {},
])
def test_a_benchmark_that_names_no_criteria_still_judges(fake_geval, config):
    """GEval refuses to build without criteria, so a judge benchmark naming
    none errored every row and failed the run on the error rate."""
    results = run_judge_rows(fake_geval, config=config)

    assert [r["status"] for r in results] == ["scored", "scored"]
    assert all(c for c in _criteria_seen()), _criteria_seen()


def test_a_named_criteria_is_not_overridden_by_the_default(fake_geval):
    """The allow direction: the fallback must not shadow a real criteria."""
    run_judge_rows(fake_geval, config={"judge_criteria": "Reward a professional tone."})

    assert _criteria_seen() == ["Reward a professional tone."] * len(ROWS)


def test_a_blank_criteria_cell_falls_back_to_the_benchmark_default(fake_geval):
    """The per-row path, which is the more exposed of the two: this criteria
    comes from a dataset cell, and a spreadsheet cell holding one space is
    far easier to produce by accident than a hand-typed empty string. It
    fails that row rather than falling back."""
    fake_geval()
    rows = [dict(ROWS[0], judge_criteria="  "), dict(ROWS[1], judge_criteria="")]

    results = CustomEvalBackend()._evaluate_judge_rows(
        rows, FakeTarget(), {"judge_criteria": "Reward a professional tone."}, {}, None,
    )

    assert [r["status"] for r in results] == ["scored", "scored"]
    assert _criteria_seen() == ["Reward a professional tone."] * len(rows)


def test_judge_returning_no_score_is_errored_not_scored(fake_geval):
    """A judge that produced no score measured nothing.

    Recording it as `scored` with `score: None` puts a None into the
    average this backend computes two blocks later
    (`sum(r['score'] for r in results if r['status'] != 'errored')`), which
    raises outside any try and takes down the whole benchmark after every
    row has already been paid for.

    The old rule got this right by accident: `metric.score >= 0.5` raised a
    TypeError on None *inside* the try, so the row landed as errored.
    `row_passed` deliberately accepts None -- it must, on the evalscope
    watcher path, where an unreadable review line genuinely has no score --
    so this path has to state the distinction rather than inherit it from
    an exception.

    Unreachable with the installed deepeval, whose GEval always assigns a
    float or raises. Covered anyway: the aggregate must be safe whatever a
    future version does.
    """
    results = run_judge_rows(fake_geval, score=None)

    assert results, "the rows must still be recorded"
    assert all(r["status"] == "errored" for r in results)
    assert all(r["success"] is False for r in results)
    # The aggregate the old shape would have crashed on.
    scored = [r for r in results if r["status"] != "errored"]
    assert sum(r["score"] for r in scored) == 0
