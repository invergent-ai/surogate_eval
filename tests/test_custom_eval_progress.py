import pytest
from datasets import Dataset

from surogate_eval.benchmarks.backends.custom_eval_backend import (
    _row_counts,
    CustomEvalBackend,
)
from surogate_eval.targets.base import TargetResponse


def test_counts_split_scored_from_errored():
    """An errored row is not a zero score. It must count toward `rows_done`
    without inflating `scored`, leave `passed` alone, and contribute nothing
    to `score_sum`, or a run whose target died reads as a model that
    answered everything wrong. `errored` is not a return value of
    `_row_counts` -- every row is `scored` or `errored`, so it is always
    `rows_done - scored`, same as `report_rows` derives it."""
    results = [
        {"status": "scored", "score": 1.0, "success": True},
        {"status": "scored", "score": 0.0, "success": False},
        {"status": "errored", "score": None, "success": False},
    ]

    rows_done, scored, passed, score_sum = _row_counts(results)

    assert rows_done == 3
    assert scored == 2
    assert rows_done - scored == 1, "the one errored row"
    assert passed == 1
    assert score_sum == 1.0


def test_a_judge_partial_score_sums_rather_than_rounds():
    """Judge scores are continuous. Summing them lets ops show a real running
    average; counting passes only would throw the value away."""
    results = [
        {"status": "scored", "score": 0.7, "success": True},
        {"status": "scored", "score": 0.4, "success": False},
    ]

    rows_done, scored, passed, score_sum = _row_counts(results)

    assert scored == 2
    assert passed == 1
    assert score_sum == pytest.approx(1.1)


def test_empty_results_are_all_zero():
    assert _row_counts([]) == (0, 0, 0, 0.0)


class FakeTargetForProgress:
    """A target for progress testing: rows 0 and 2 succeed, row 1 errors."""

    name = "test_target"
    config = {}

    def send_request(self, request):
        if request.prompt == "row1_instruction":
            return TargetResponse(content=None, raw_response={}, error="connection reset")
        # Rows 0 and 2 return matching answers
        return TargetResponse(content="expected", raw_response={}, error=None)


@pytest.fixture
def report_calls(monkeypatch):
    """Captures every `report_rows` call `custom_eval_backend` makes, in
    order, as dicts keyed by parameter name."""
    calls = []

    def fake_report_rows(rows_done, rows_total, scored, passed, score_sum):
        calls.append(
            {
                "rows_done": rows_done,
                "rows_total": rows_total,
                "scored": scored,
                "passed": passed,
                "score_sum": score_sum,
            }
        )

    monkeypatch.setattr(
        "surogate_eval.benchmarks.backends.custom_eval_backend.runners.report_rows",
        fake_report_rows,
    )
    return calls


def test_exact_match_loop_reports_progress(report_calls):
    """The unconditional post-loop write ensures report_rows is called at least
    once, and the final call carries the row counts correctly. This test catches:
    - Missing or broken unconditional write after loop
    - Passing wrong rows_done (should == len(results))
    - Passing wrong rows_total (should == len(rows))
    - Miscounting scored/errored/passed/score_sum in the final report
    """
    rows = [
        {"instruction": "row0_instruction", "answer": "expected", "_original_idx": 0},
        {"instruction": "row1_instruction", "answer": "expected", "_original_idx": 1},
        {"instruction": "row2_instruction", "answer": "expected", "_original_idx": 2},
    ]

    backend = CustomEvalBackend()
    results = backend._evaluate_exact_match_rows(rows, FakeTargetForProgress(), {}, {})

    # Unconditional write after loop guarantees at least one call
    assert len(report_calls) >= 1, "report_rows was never called"

    # The final call is the unconditional post-loop write
    final_call = report_calls[-1]

    # rows_done should equal the number of results
    assert final_call["rows_done"] == len(results) == 3

    # rows_total should equal the number of rows passed in
    assert final_call["rows_total"] == len(rows) == 3

    # Check counts match the results: 2 scored (rows 0,2), 1 errored (row 1)
    scored_results = [r for r in results if r["status"] == "scored"]
    errored_results = [r for r in results if r["status"] == "errored"]
    passed_results = [r for r in results if r["success"]]

    assert final_call["scored"] == len(scored_results) == 2
    assert final_call["rows_done"] - final_call["scored"] == len(errored_results) == 1
    assert final_call["passed"] == len(passed_results) == 2
    assert final_call["score_sum"] == sum(r["score"] for r in scored_results) == 2.0


class UnreachableTarget:
    """Every request comes back carrying an error -- the target-is-down
    case a user is most likely watching when live progress is what tells
    them anything is happening at all (Finding 3)."""

    name = "unreachable"
    config = {}

    def send_request(self, request):
        return TargetResponse(content=None, raw_response={}, error="connection reset")


def test_exact_match_loop_with_every_row_erroring_still_reports_mid_loop(report_calls):
    """The errored branch used to `continue` straight past the reporter, so
    a target that fails every row reported nothing until the single
    unconditional write after the loop ended. That is exactly the case a
    user is most likely watching live -- the target is down -- and it used
    to freeze the progress bar for the whole run (Finding 3)."""
    rows = [
        {"instruction": f"row{i}_instruction", "answer": "expected", "_original_idx": i}
        for i in range(3)
    ]

    backend = CustomEvalBackend()
    results = backend._evaluate_exact_match_rows(rows, UnreachableTarget(), {}, {})

    assert [r["status"] for r in results] == ["errored"] * 3
    assert len(report_calls) >= 2, "must report during the loop, not just once after it ends"
    # `_last_report` starts at 0.0, so the very first `maybe_report` call
    # (after row 0) always clears the throttle regardless of wall-clock
    # timing -- which is what lets this assert on call count/content rather
    # than on real elapsed time.
    assert report_calls[0]["rows_done"] < len(rows), (
        "the first report must land before every row is measured"
    )


def test_judge_loop_with_every_row_erroring_still_reports_mid_loop(report_calls):
    """Same bug, judge path: the request-error branch `continue`d past the
    reporter before ever reaching G-Eval (Finding 3)."""
    rows = [
        {"instruction": f"row{i}_instruction", "answer": "expected", "_original_idx": i}
        for i in range(3)
    ]

    backend = CustomEvalBackend()
    results = backend._evaluate_judge_rows(rows, UnreachableTarget(), {}, {})

    assert [r["status"] for r in results] == ["errored"] * 3
    assert len(report_calls) >= 2, "must report during the loop, not just once after it ends"
    assert report_calls[0]["rows_done"] < len(rows), (
        "the first report must land before every row is measured"
    )


class FakeGEvalForProgress:
    """A fake GEval for progress testing."""

    def __init__(self, score=0.9, **kwargs):
        self.score = score
        self.reason = "good enough"

    def measure(self, test_case, _show_indicator=False):
        pass


def test_mixed_benchmark_progress_is_cumulative(monkeypatch, report_calls):
    """Mixed benchmarks with both exact_match and judge rows must report
    cumulative progress -- not just rows_done, but every counter, since ops
    divides score_sum/scored into a running average. This test catches:
    - rows_total changing between loops (bar would reset/jump backward)
    - rows_done decreasing between consecutive reports (bar would go backward)
    - scored/passed/score_sum resetting to only what the second loop has
      measured so far, dragging the running average backward the instant
      the judge loop starts (Finding 4)

    Driven through `evaluate()` itself, with 3 exact_match rows and 2 judge
    rows, rather than the two loops called directly with hand-passed
    offsets: `evaluate()` is what actually computes each loop's offsets in
    production, and a test that bypasses it cannot catch a bug in that
    wiring (Finding 6). The uneven row counts matter too: with an equal
    split, scored/passed could tie across the reset by coincidence and the
    assertion would pass either way.
    """
    rows = [
        {"instruction": "em0", "answer": "expected", "eval_type": "exact_match"},
        {"instruction": "em1", "answer": "expected", "eval_type": "exact_match"},
        {"instruction": "em2", "answer": "expected", "eval_type": "exact_match"},
        {"instruction": "j0", "answer": "expected", "eval_type": "judge"},
        {"instruction": "j1", "answer": "expected", "eval_type": "judge"},
    ]
    dataset = Dataset.from_list(rows)

    monkeypatch.setattr(
        "surogate_eval.benchmarks.backends.custom_eval_backend.GEval",
        lambda **kwargs: FakeGEvalForProgress(**kwargs)
    )

    backend = CustomEvalBackend()
    monkeypatch.setattr(backend, "_load_dataset", lambda *a, **kw: dataset)

    backend.evaluate(
        FakeTargetForProgress(), "mixed_bench",
        {"source": "unused", "eval_type": "hybrid"},
    )

    # Verify we have at least one report per loop (from unconditional writes)
    assert len(report_calls) >= 2, "Both loops should report at least once"

    # Verify rows_total is constant throughout
    rows_totals = [call["rows_total"] for call in report_calls]
    assert all(t == 5 for t in rows_totals), f"rows_total should always be 5, got {rows_totals}"

    # Verify rows_done, and every score counter, never decreases (catches
    # a progress bar, or a running average, jumping backward)
    for key in ("rows_done", "scored", "passed", "score_sum"):
        values = [call[key] for call in report_calls]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], (
                f"{key} must not decrease: {values} at call {i} "
                f"(this would cause the progress bar or running average to go backward)"
            )

    # The final call should show all rows done, measured across both loops
    final_call = report_calls[-1]
    assert final_call["rows_done"] == 5, "Final report should show all 5 rows done"
    assert final_call["scored"] == 5, "scored must include both loops' rows, not just the judge loop's"
    assert final_call["score_sum"] == pytest.approx(3.0 + 1.8), (
        "score_sum must carry the exact_match loop's total forward into the judge loop's report"
    )
