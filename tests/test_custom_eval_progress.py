import pytest

from surogate_eval.benchmarks.backends.custom_eval_backend import _row_counts


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
