"""A custom benchmark's string rows must be scored by a rule the user chose.

`exact_match` meant `expected in output` after a normalisation pass that also
guessed the answer out of the output with hardcoded heuristics. So expected
"A" matched "The answer is C.", expected "no" matched "Nobody knows.", and
every row of an MCQ benchmark passed.

No network: these are plain strings.
"""

import pytest

from surogate_eval.benchmarks.matching import (
    MatchTimeout,
    build_matcher,
    clean_formatting,
)
from surogate_eval.errors import ConfigError

# The false positives that motivated this, as (expected, output).
FALSE_POSITIVES = [
    ("A", "The answer is C."),
    ("no", "Nobody knows."),
    ("4", "It took 14 minutes."),
    ("Paris", "Paris is not the capital; Berlin is."),
]


@pytest.mark.parametrize("expected, output", FALSE_POSITIVES)
def test_contains_still_accepts_them_so_the_default_does_not_regress(expected, output):
    """`contains` is the default and must behave exactly as today."""
    success, _cleaned = build_matcher(None).compare(output, expected)

    assert success is True


@pytest.mark.parametrize("expected, output", FALSE_POSITIVES)
def test_exact_rejects_them(expected, output):
    success, _cleaned = build_matcher({"mode": "exact"}).compare(output, expected)

    assert success is False


def test_exact_accepts_a_real_match_through_markdown():
    """Formatting cleanup stays: markdown is presentation, not the answer."""
    success, cleaned = build_matcher({"mode": "exact"}).compare("**42**", "42")

    assert success is True
    assert cleaned == "42"


def test_regex_extracts_the_group_and_compares_it():
    matcher = build_matcher({"mode": "regex", "pattern": r"\b([ABCD])\b"})

    wrong, cleaned = matcher.compare("The answer is C.", "A")
    right, _ = matcher.compare("The answer is A.", "A")

    assert wrong is False, "extracted C must not match expected A"
    assert cleaned == "C", "the record should show what we extracted"
    assert right is True


def test_regex_without_a_capture_group_uses_the_whole_match():
    matcher = build_matcher({"mode": "regex", "pattern": r"\d+"})

    success, cleaned = matcher.compare("It took 14 minutes.", "14")

    assert success is True
    assert cleaned == "14"


def test_regex_that_does_not_match_is_a_wrong_answer_not_an_error():
    """The pattern is the answer format the benchmark asked for."""
    matcher = build_matcher({"mode": "regex", "pattern": r"\b([ABCD])\b"})

    success, cleaned = matcher.compare("I am not sure.", "A")

    assert success is False
    assert cleaned == ""


def test_regex_flags_are_honoured():
    matcher = build_matcher({"mode": "regex", "pattern": r"answer: (\w+)", "flags": "i"})

    success, _cleaned = matcher.compare("ANSWER: yes", "yes")

    assert success is True


def test_an_explicit_group_index_is_honoured():
    matcher = build_matcher(
        {"mode": "regex", "pattern": r"(\w+)=(\w+)", "group": 2}
    )

    success, cleaned = matcher.compare("key=value", "value")

    assert success is True
    assert cleaned == "value"


def test_an_unknown_mode_is_rejected_rather_than_silently_treated_as_contains():
    with pytest.raises(ConfigError) as excinfo:
        build_matcher({"mode": "fuzzy"})

    assert "fuzzy" in str(excinfo.value)


def test_an_invalid_pattern_is_rejected_at_build_time():
    """Every row would hit it, so it is a config error, not a row error."""
    with pytest.raises(ConfigError):
        build_matcher({"mode": "regex", "pattern": "([unclosed"})


def test_regex_mode_requires_a_pattern():
    with pytest.raises(ConfigError):
        build_matcher({"mode": "regex"})


def test_a_group_the_pattern_does_not_have_is_rejected_at_build_time():
    with pytest.raises(ConfigError):
        build_matcher({"mode": "regex", "pattern": r"(\w+)", "group": 3})


def test_a_catastrophic_pattern_is_bounded_rather_than_hanging_the_run():
    """The pattern is the tenant's own, so this is a foot-gun not an attack.

    It still must cost a row rather than the pod, which is why the match runs
    under a timeout instead of stdlib `re`, which has none.
    """
    matcher = build_matcher(
        {"mode": "regex", "pattern": r"(a+)+$", "timeout": 0.1}
    )

    with pytest.raises(MatchTimeout):
        matcher.compare("a" * 5000 + "!", "a")


def test_clean_formatting_leaves_a_plain_answer_alone():
    assert clean_formatting("  42  ") == "42"
    assert clean_formatting("**42**") == "42"


def test_the_retired_heuristics_no_longer_rewrite_the_output():
    """The behaviour change, stated as a test.

    `_normalize_output` used to pull an email out of a sentence and compare
    that. Under `exact` the sentence is now simply not the answer; a user who
    wants the old behaviour writes a pattern for it, and gets to see the rule.
    """
    by_hand = build_matcher({"mode": "exact"})
    with_pattern = build_matcher(
        {"mode": "regex", "pattern": r"[\w\.-]+@[\w\.-]+\.\w+"}
    )

    assert by_hand.compare("Contact: a@b.com", "a@b.com")[0] is False
    assert with_pattern.compare("Contact: a@b.com", "a@b.com")[0] is True
