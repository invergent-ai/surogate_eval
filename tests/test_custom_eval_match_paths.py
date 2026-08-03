"""The chosen match mode must reach the rows, through the real evaluator.

Task 1 proved the rule in isolation. This drives
`_evaluate_exact_match_direct` with a real target and real rows, so a mode
that never reaches the comparison is caught here rather than in review.

No network: the target is a fake that returns a canned answer.
"""

import pytest

from surogate_eval.benchmarks.backends.custom_eval_backend import CustomEvalBackend
from surogate_eval.targets.base import TargetResponse


class FakeTarget:
    """Answers every request with the same text."""

    name = "t1"
    config = {}

    def __init__(self, answer):
        self.answer = answer

    def send_request(self, request):
        return TargetResponse(content=self.answer, raw_response={}, error=None)


COLUMNS = {"instruction": "instruction", "answer": "answer"}


def _score(answer, expected, matcher=None):
    """Run one row through the real direct-inference evaluator."""
    backend = CustomEvalBackend.__new__(CustomEvalBackend)
    rows = [{"instruction": "q", "answer": expected, "_original_idx": 0}]
    config = {"matcher": matcher} if matcher is not None else {}

    return backend._evaluate_exact_match_direct(
        rows, FakeTarget(answer), config, COLUMNS
    )[0]


def test_the_default_is_still_containment():
    """No matcher configured must mean exactly today's behaviour."""
    assert _score("The answer is C.", "A")["success"] is True


def test_exact_mode_reaches_the_comparison():
    row = _score("The answer is C.", "A", {"mode": "exact"})

    assert row["success"] is False
    assert row["score"] == 0.0
    assert row["status"] == "scored", "a wrong answer is measured, not errored"


def test_regex_mode_reaches_the_comparison():
    row = _score("The answer is C.", "A", {"mode": "regex", "pattern": r"\b([ABCD])\b"})

    assert row["success"] is False
    assert row["output"] == "C", "the record shows what the pattern extracted"


def test_a_bad_matcher_fails_the_benchmark_rather_than_every_row():
    from surogate_eval.errors import ConfigError

    with pytest.raises(ConfigError):
        _score("anything", "A", {"mode": "nonsense"})
