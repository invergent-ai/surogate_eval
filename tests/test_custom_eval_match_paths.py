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


def _score_lm_eval(answer, expected, matcher=None, returned_rows=1):
    """Run one row through the lm-eval path, with lm-eval itself stubbed.

    Only generation is stubbed. The scoring under test is ours.
    """
    from surogate_eval.benchmarks.backends import custom_eval_backend as ceb

    class StubLMEval:
        def evaluate(self, target, benchmark_name, config):
            details = [
                {
                    "output": "IGNORED",  # lm-eval's own extraction guess
                    "raw_output": answer,
                    "metrics": {"exact_match": 1},  # lm-eval says correct
                }
            ] * returned_rows
            return {"detailed_results": details}

    backend = ceb.CustomEvalBackend.__new__(ceb.CustomEvalBackend)
    rows = [{"instruction": "q", "answer": expected, "_original_idx": 0}]
    config = {"matcher": matcher} if matcher is not None else {}

    import surogate_eval.benchmarks.backends.lm_eval_backend as lm
    saved = lm.LMEvalBackend
    lm.LMEvalBackend = StubLMEval
    try:
        return backend._evaluate_exact_match_lm_eval(
            rows, FakeTarget(answer), config, COLUMNS, tokenizer="gpt2"
        )
    finally:
        lm.LMEvalBackend = saved


def test_the_lm_eval_path_scores_with_our_matcher_not_lm_evals_metric():
    """lm-eval said correct; the configured mode says otherwise, and wins."""
    row = _score_lm_eval("The answer is C.", "A", {"mode": "exact"})[0]

    assert row["success"] is False, "lm-eval's exact_match metric must not decide this"
    assert row["score"] == 0.0


def test_both_paths_agree_on_the_same_row():
    """A benchmark's result must not depend on whether a tokenizer is set.

    Agreement alone is not enough to assert here: pre-branch, both paths
    also agreed on this exact row - on the wrong answer, because the direct
    path matched by containment and the lm-eval path trusted its own
    `exact_match` metric. Pinning the value is what makes this test actually
    exercise the branch's headline claim rather than passing for the same
    reason it used to.
    """
    direct = _score("The answer is C.", "A", {"mode": "exact"})
    lm_eval = _score_lm_eval("The answer is C.", "A", {"mode": "exact"})[0]

    assert direct["success"] is False
    assert direct["success"] == lm_eval["success"]
    assert direct["score"] == lm_eval["score"]


def test_a_row_lm_eval_never_returned_is_unmeasured():
    """It was not scored, so it is not a wrong answer."""
    rows = [
        {"instruction": "q1", "answer": "A", "_original_idx": 0},
        {"instruction": "q2", "answer": "B", "_original_idx": 1},
    ]
    from surogate_eval.benchmarks.backends import custom_eval_backend as ceb
    import surogate_eval.benchmarks.backends.lm_eval_backend as lm

    class ShortStub:
        def evaluate(self, target, benchmark_name, config):
            return {"detailed_results": [
                {"output": "A", "raw_output": "A", "metrics": {"exact_match": 1}}
            ]}

    backend = ceb.CustomEvalBackend.__new__(ceb.CustomEvalBackend)
    saved = lm.LMEvalBackend
    lm.LMEvalBackend = ShortStub
    try:
        results = backend._evaluate_exact_match_lm_eval(
            rows, FakeTarget("A"), {}, COLUMNS, tokenizer="gpt2"
        )
    finally:
        lm.LMEvalBackend = saved

    assert results[1]["score"] is None, "a row never returned is not a zero"
    assert results[1]["success"] is False
    assert results[1]["status"] == "errored"
    assert results[1]["reason"]
