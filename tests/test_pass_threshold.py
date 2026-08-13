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
