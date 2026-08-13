"""A graded score becomes a pass/fail by one rule, in one place.

The two backends decided this differently and silently: the evalscope path
used `score > 0`, custom_eval's judge path used `score >= 0.5`. On a
judge-scored benchmark the first passes essentially every row -- a judge
rating a coherent answer gives it something, and only a literal zero fails.
Observed live on 2026-08-13: an MT-Bench run reported 10 of 10 samples
passed while its own average was 6.5 out of 10.
"""

from surogate_eval.benchmarks.pass_rule import LEGACY_JUDGE_MIN, row_passed


def test_a_configured_threshold_fails_a_row_that_scored_under_it():
    # The MT-Bench case: a 0.65 row against "at least 8.0 out of 10".
    assert row_passed(0.65, 0.8) is False
    assert row_passed(0.8, 0.8) is True, "the threshold is inclusive"
    assert row_passed(0.95, 0.8) is True


def test_without_a_threshold_evalscope_keeps_its_any_non_zero_rule():
    """Turning this on must be a choice, not a silent re-scoring of every
    existing config's history."""
    assert row_passed(0.1) is True
    assert row_passed(0.0) is False


def test_without_a_threshold_the_judge_path_keeps_its_half_mark_rule():
    assert row_passed(0.4, legacy_minimum=LEGACY_JUDGE_MIN, legacy_inclusive=True) is False
    assert row_passed(0.5, legacy_minimum=LEGACY_JUDGE_MIN, legacy_inclusive=True) is True


def test_an_unreadable_row_is_never_a_pass():
    """None is unmeasured, which is a different claim from scoring badly --
    the distinction the rest of this codebase is careful about."""
    assert row_passed(None) is False
    assert row_passed(None, 0.8) is False
    assert row_passed(None, legacy_minimum=LEGACY_JUDGE_MIN, legacy_inclusive=True) is False


def test_a_zero_threshold_is_honoured_rather_than_read_as_unset():
    """0.0 is falsy, so an `if pass_threshold:` guard would silently drop it
    back to the legacy rule. They differ: at-least-zero passes a zero row,
    the legacy evalscope rule does not."""
    assert row_passed(0.0, 0.0) is True
    assert row_passed(0.0) is False


def test_the_operator_is_stated_rather_than_inferred_from_the_value():
    """The rule used to pick > or >= by comparing the caller's number to a
    sentinel constant, so a caller passing some third value silently got
    whichever operator its number happened to match. Same number, both
    operators available, chosen explicitly.
    """
    assert row_passed(0.3, legacy_minimum=0.3, legacy_inclusive=True) is True
    assert row_passed(0.3, legacy_minimum=0.3, legacy_inclusive=False) is False


def test_the_adapters_record_which_rule_scored_the_run():
    """Provenance, because nothing else stores it.

    A stored row says only "failed". Whether that was a failure depends on
    the threshold it was judged against -- 6.5 out of 10 passes at 5.0 and
    fails at 8.0 -- and without the rule in the payload an old run's
    Pass/Fail column cannot be read back, nor two runs compared knowing
    whether the rule moved between them.

    Asserted on the source rather than by running the adapters, which need
    a dataset and a live judge. Both build one metadata dict at one return.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "surogate_eval"
    for name in ("tool_decathlon", "vita_bench"):
        src = (root / "benchmarks" / "adapters" / f"{name}.py").read_text()
        assert '"pass_threshold": self.config.pass_threshold,' in src, (
            f"{name} must record the threshold its verdicts were decided under"
        )


def test_the_adapters_report_a_score_the_threshold_cannot_move():
    """`overall_score` is the mean of the raw scores, not the pass fraction.

    This is what makes the threshold safe to change: A/B compare reads
    `overall_score`, so two runs of one model at different thresholds stay
    comparable. A `correct / total` local used to sit beside it in both
    files, computed and (in tool_decathlon) never used, close enough to the
    reported field to read as though the threshold re-scored the benchmark.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "surogate_eval"
    for name in ("tool_decathlon", "vita_bench"):
        src = (root / "benchmarks" / "adapters" / f"{name}.py").read_text()
        assert "overall_score=avg_score," in src, f"{name} must report the mean score"
        assert "overall_score = correct / total" not in src, (
            f"{name} has a pass-fraction local shadowing the reported score"
        )
