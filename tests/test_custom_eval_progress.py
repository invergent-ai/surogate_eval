import pytest

from surogate_eval.benchmarks.backends.custom_eval_backend import (
    _row_counts,
    CustomEvalBackend,
)
from surogate_eval.targets.base import TargetResponse


def test_counts_split_scored_from_errored():
    """An errored row is not a zero score. It must raise `errored`, leave
    `passed` alone, and contribute nothing to `score_sum`, or a run whose
    target died reads as a model that answered everything wrong."""
    results = [
        {"status": "scored", "score": 1.0, "success": True},
        {"status": "scored", "score": 0.0, "success": False},
        {"status": "errored", "score": None, "success": False},
    ]

    scored, errored, passed, score_sum = _row_counts(results)

    assert scored == 2
    assert errored == 1
    assert passed == 1
    assert score_sum == 1.0


def test_a_judge_partial_score_sums_rather_than_rounds():
    """Judge scores are continuous. Summing them lets ops show a real running
    average; counting passes only would throw the value away."""
    results = [
        {"status": "scored", "score": 0.7, "success": True},
        {"status": "scored", "score": 0.4, "success": False},
    ]

    scored, errored, passed, score_sum = _row_counts(results)

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


def test_exact_match_loop_reports_progress(monkeypatch):
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

    # Record all calls to report_rows
    report_calls = []

    def fake_report_rows(rows_done, rows_total, scored, errored, passed, score_sum):
        report_calls.append(
            {
                "rows_done": rows_done,
                "rows_total": rows_total,
                "scored": scored,
                "errored": errored,
                "passed": passed,
                "score_sum": score_sum,
            }
        )

    monkeypatch.setattr("surogate_eval.benchmarks.backends.custom_eval_backend.runners.report_rows", fake_report_rows)

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
    assert final_call["errored"] == len(errored_results) == 1
    assert final_call["passed"] == len(passed_results) == 2
    assert final_call["score_sum"] == sum(r["score"] for r in scored_results) == 2.0
