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
    """Run one row through the real direct-inference evaluator.

    Routed through `_evaluate_exact_match_rows` rather than calling the
    direct path itself, so the matcher is built by the code under test.
    With no tokenizer anywhere, that entry point picks the direct path.
    """
    backend = CustomEvalBackend.__new__(CustomEvalBackend)
    rows = [{"instruction": "q", "answer": expected, "_original_idx": 0}]
    config = {"matcher": matcher} if matcher is not None else {}

    return backend._evaluate_exact_match_rows(
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


@pytest.mark.parametrize(
    "matcher, answer, expected",
    [
        (None, "The answer is A.", "A"),
        ({"mode": "exact"}, "A", "A"),
        ({"mode": "regex", "pattern": r"\b([ABCD])\b"}, "The answer is A.", "A"),
    ],
    ids=["contains", "exact", "regex"],
)
def test_the_success_reason_names_the_mode_that_actually_ran(matcher, answer, expected):
    """`reason` used to be hardcoded `'Exact match'` on every success, which
    was only ever true under `mode: exact` - wrong under the default
    `contains` and under `regex`. `Matcher.mode` is exactly the mode that
    ran, so it is what the report should say."""
    row = _score(answer, expected, matcher)
    mode = matcher["mode"] if matcher else "contains"

    assert row["success"] is True
    assert row["reason"] == f"{mode} match"


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
    from surogate_eval.benchmarks.matching import build_matcher
    saved = lm.LMEvalBackend
    lm.LMEvalBackend = StubLMEval
    try:
        # Called directly rather than through `_evaluate_exact_match_rows`:
        # that entry point catches everything from this path and silently
        # falls back to direct inference, which would turn a real lm-eval
        # scoring failure into a passing test against the other path.
        return backend._evaluate_exact_match_lm_eval(
            rows, FakeTarget(answer), config, COLUMNS, tokenizer="gpt2",
            matcher=build_matcher(matcher),
        )
    finally:
        lm.LMEvalBackend = saved


def test_the_lm_eval_path_scores_with_our_matcher_not_lm_evals_metric():
    """lm-eval said correct; the configured mode says otherwise, and wins."""
    row = _score_lm_eval("The answer is C.", "A", {"mode": "exact"})[0]

    assert row["success"] is False, "lm-eval's exact_match metric must not decide this"
    assert row["score"] == 0.0


def test_the_lm_eval_path_success_reason_also_names_the_mode():
    """The lm-eval path used to hardcode `'Match'` on success, disagreeing
    with the direct path's (also wrong) `'Exact match'`. Both must say the
    same thing, and both must name the mode that actually ran."""
    row = _score_lm_eval("A", "A", {"mode": "regex", "pattern": r"\b([ABCD])\b"})[0]

    assert row["success"] is True
    assert row["reason"] == "regex match"


def test_both_paths_report_the_same_success_reason_text():
    """Identical record shapes must mean identical report text, not just an
    identical score - a user reading either path's output should not be able
    to tell them apart."""
    direct = _score("A", "A", {"mode": "exact"})
    lm_eval = _score_lm_eval("A", "A", {"mode": "exact"})[0]

    assert direct["reason"] == lm_eval["reason"] == "exact match"


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

    from surogate_eval.benchmarks.matching import build_matcher

    backend = ceb.CustomEvalBackend.__new__(ceb.CustomEvalBackend)
    saved = lm.LMEvalBackend
    lm.LMEvalBackend = ShortStub
    try:
        results = backend._evaluate_exact_match_lm_eval(
            rows, FakeTarget("A"), {}, COLUMNS, tokenizer="gpt2",
            matcher=build_matcher(None),
        )
    finally:
        lm.LMEvalBackend = saved

    assert results[1]["score"] is None, "a row never returned is not a zero"
    assert results[1]["success"] is False
    assert results[1]["status"] == "errored"
    assert results[1]["reason"]


def test_a_bad_matcher_is_not_reported_as_an_lm_eval_failure():
    """The matcher is built before the path choice, so a config error names
    itself.

    It used to be built inside the lm-eval path, under the caller's blanket
    `except Exception`, so `matcher: {mode: nonsense}` on a benchmark with a
    tokenizer logged `lm-eval exact_match failed, falling back to direct
    inference: unknown matcher mode 'nonsense'` and then raised the same
    error again from the direct path. The first thing a user saw pointed at
    lm-eval, which was not what was broken.
    """
    from surogate_eval.errors import ConfigError

    backend = CustomEvalBackend.__new__(CustomEvalBackend)
    rows = [{"instruction": "q", "answer": "A", "_original_idx": 0}]

    # Asserting `raises(ConfigError)` alone would pass either way, because the
    # direct path rebuilt the matcher and raised the same error one step
    # later. What changed is that the lm-eval path is never entered at all, so
    # that is what this pins - with a flag rather than a raising sentinel,
    # since the blanket `except Exception` below would swallow the sentinel
    # and fall through to the direct path exactly as it did before.
    entered = []
    backend._evaluate_exact_match_lm_eval = lambda *a, **k: entered.append(1)

    with pytest.raises(ConfigError, match="nonsense"):
        backend._evaluate_exact_match_rows(
            rows,
            FakeTarget("A"),
            {"tokenizer": "gpt2", "matcher": {"mode": "nonsense"}},
            COLUMNS,
        )

    assert not entered, "the matcher must be validated before the path choice"


def test_a_reordered_lm_eval_result_is_unmeasured_not_miscored():
    """Positional pairing became score-bearing, so it must fail closed.

    Before the matcher moved in, the score came from lm-eval's own per-sample
    metric, so a reordered sample list only mislabeled the displayed columns.
    Now `detail[i]`'s generation is compared against `rows[i]`'s answer key,
    which would silently score one row's output against another row's answer.
    """
    from surogate_eval.benchmarks.backends import custom_eval_backend as ceb
    from surogate_eval.benchmarks.matching import build_matcher
    import surogate_eval.benchmarks.backends.lm_eval_backend as lm

    rows = [
        {"instruction": "q1", "answer": "A", "_original_idx": 0},
        {"instruction": "q2", "answer": "B", "_original_idx": 1},
    ]

    class ReorderedStub:
        """Returns the two samples back to front, each with its own target."""

        def evaluate(self, target, benchmark_name, config):
            return {"detailed_results": [
                {"expected": "B", "raw_output": "B", "metrics": {}},
                {"expected": "A", "raw_output": "A", "metrics": {}},
            ]}

    backend = ceb.CustomEvalBackend.__new__(ceb.CustomEvalBackend)
    saved = lm.LMEvalBackend
    lm.LMEvalBackend = ReorderedStub
    try:
        results = backend._evaluate_exact_match_lm_eval(
            rows, FakeTarget("A"), {}, COLUMNS, tokenizer="gpt2",
            matcher=build_matcher(None),
        )
    finally:
        lm.LMEvalBackend = saved

    # Paired positionally and unguarded, both rows would have scored 1.0:
    # "B" contains "B" and "A" contains "A". They are the wrong pairs.
    assert [r["status"] for r in results] == ["errored", "errored"]
    assert all(r["score"] is None for r in results)
    assert "order" in results[0]["reason"]


def test_the_default_mode_on_the_lm_eval_path_is_containment_not_lm_evals_metric():
    """The default's blast radius on this path, pinned rather than described.

    Every other lm-eval test here passes an explicit `{"mode": "exact"}`, so
    nothing covered what a config with no `matcher` block actually does here.
    It matters more on this path than on the direct one: scoring used to come
    from lm-eval's own `exact_match` metric (full-string equality), so the
    unified default is *looser* than what this path did before, and on a
    single-letter answer key that is close to unscoreable.
    """
    row = _score_lm_eval("Hmm, I'll probably go with D.", "B")[0]

    assert row["success"] is True, (
        "expected 'B' is contained in 'probably', so containment passes a "
        "wrong answer; this is why a tokenizer-bearing MCQ config must set a "
        "mode explicitly"
    )
    assert row["reason"] == "contains match"

    strict = _score_lm_eval("Hmm, I'll probably go with D.", "B", {"mode": "exact"})[0]
    assert strict["success"] is False


def test_the_shipped_tokenizer_config_sets_a_mode():
    """`examples/custom_eval_test_gpt.yaml` sets a tokenizer and scores
    single-letter answer keys, which is exactly the combination the default
    scores near 100% on regardless of the model. It must not rely on the
    default."""
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "examples" / "custom_eval_test_gpt.yaml").read_text()
    )
    target = next(t for t in cfg["targets"] if t.get("evaluations"))
    benchmark = target["evaluations"][0]["benchmarks"][0]

    assert target.get("tokenizer"), "guard assumes this config takes the lm-eval path"
    assert benchmark.get("matcher", {}).get("mode"), "must not rely on the default"


def test_the_lm_eval_path_does_not_pretend_to_take_progress_params():
    """`_evaluate_exact_match_lm_eval` accepted `rows_total`/`rows_done_offset`
    but never reported progress with them -- every test above drives it
    without them and none of its body ever reads them. Accepting dead
    parameters reads as coverage (instrumentation) that does not actually
    exist, so they must be dropped rather than merely left unused.
    """
    import inspect

    params = inspect.signature(CustomEvalBackend._evaluate_exact_match_lm_eval).parameters
    assert "rows_total" not in params
    assert "rows_done_offset" not in params


def test_an_lm_eval_failure_carries_its_real_cause_into_the_row_reason():
    """`LMEvalBackend.evaluate` does not raise when `simple_evaluate` fails.

    It returns a payload with no `detailed_results` and the real cause in
    `metadata['error']`, so the most likely production failure (unreachable
    target, bad tokenizer) arrives here as "zero rows returned". Every row
    then takes the unmeasured branch, and without this the user sees N
    identical rows saying only "lm-eval returned no result for this row"
    while the actual diagnostic is discarded. The documented fall-back to
    direct inference never fires for this either, because nothing raised.
    """
    from surogate_eval.benchmarks.backends import custom_eval_backend as ceb
    from surogate_eval.benchmarks.matching import build_matcher
    import surogate_eval.benchmarks.backends.lm_eval_backend as lm

    class FailedStub:
        def evaluate(self, target, benchmark_name, config):
            return {
                "overall_score": 0.0,
                "num_samples": 0,
                "task_results": {},
                "metadata": {"status": "failed", "error": "Connection refused: :8000"},
            }

    backend = ceb.CustomEvalBackend.__new__(ceb.CustomEvalBackend)
    rows = [{"instruction": "q", "answer": "A", "_original_idx": 0}]
    saved = lm.LMEvalBackend
    lm.LMEvalBackend = FailedStub
    try:
        results = backend._evaluate_exact_match_lm_eval(
            rows, FakeTarget("A"), {}, COLUMNS, tokenizer="gpt2",
            matcher=build_matcher(None),
        )
    finally:
        lm.LMEvalBackend = saved

    assert results[0]["status"] == "errored"
    assert results[0]["score"] is None
    assert "Connection refused" in results[0]["reason"]
