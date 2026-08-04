"""A custom benchmark's string rows must be scored by a rule the user chose.

`exact_match` meant `expected in output` after a normalisation pass that also
guessed the answer out of the output with hardcoded heuristics. So expected
"A" matched "The answer is C.", expected "no" matched "Nobody knows.", and
every row of an MCQ benchmark passed.

No network: these are plain strings.
"""

import re
import tomllib
from pathlib import Path

import pytest

from surogate_eval.benchmarks.matching import (
    DEFAULT_MODE,
    MatchTimeout,
    build_matcher,
    clean_formatting,
)
from surogate_eval.errors import ConfigError, UnscorableRow

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


@pytest.mark.parametrize(
    "cfg",
    [
        {"mode": "regex", "pattern": "([unclosed"},
        {"mode": "regex"},
        {"mode": "regex", "pattern": r"(\w+)", "group": 3},
    ],
    ids=["invalid-pattern", "missing-pattern", "group-out-of-range"],
)
def test_bad_regex_config_is_rejected_at_build_time(cfg):
    """Every row would hit these, so each is a config error, not a row error."""
    with pytest.raises(ConfigError):
        build_matcher(cfg)


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


@pytest.mark.parametrize(
    "markup, expected",
    [
        ("**A**.", "A."),
        ("**50**%", "50%"),
        ("*yes*!", "yes!"),
        ("**A**,", "A,"),
    ],
)
def test_clean_formatting_does_not_space_pad_inline_markup_boundaries(markup, expected):
    """``get_text(separator=' ')`` inserted a space at every tag boundary, so
    ``**A**.`` became ``A .``. Under `contains` that never mattered
    (``"a" in "a ."``), but `exact` and `regex` compare the cleaned text
    verbatim, so a correct ``**A**.`` was scored wrong against expected
    ``A``. The one pre-existing markdown test (``**42**``) happens to be the
    single shape where nothing sits next to the markup, so it never caught
    this."""
    assert clean_formatting(markup) == expected


def test_exact_accepts_inline_markup_directly_adjacent_to_punctuation():
    """The regression stated the way a benchmark would actually hit it: an
    `exact` comparison of a model's ``**A**.`` against an expected ``A``."""
    success, cleaned = build_matcher({"mode": "exact"}).compare("**A**.", "A.")

    assert success is True
    assert cleaned == "A."


def test_clean_formatting_does_not_run_words_together_across_paragraphs():
    """The fix for the above (dropping the inline separator) must not merge
    text across BLOCK boundaries instead. Two paragraphs must not collapse
    into one word."""
    cleaned = clean_formatting("a\n\nb")

    assert cleaned != "ab"
    assert cleaned == "a b"


def test_clean_formatting_does_not_run_list_items_together():
    cleaned = clean_formatting("- item one\n- item two\n- item three")

    assert cleaned == "item one item two item three"
    assert "oneitem" not in cleaned


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


# --- fix round 1 -------------------------------------------------------


def test_a_negative_group_is_rejected_at_build_time():
    """The bounds check only guarded the upper end. A negative group passed
    build-time validation and then crashed with a raw IndexError on the
    first row compared - exactly the every-row misconfiguration this module
    exists to catch before evaluation starts."""
    with pytest.raises(ConfigError):
        build_matcher({"mode": "regex", "pattern": r"(\w+)", "group": -1})


def test_a_non_string_expected_does_not_crash_compare():
    """A numeric answer key authored as ``42`` rather than ``"42"`` is a
    realistic config shape. It must not crash inside ``clean_formatting``;
    values are coerced to str at the boundary instead."""
    success, cleaned = build_matcher(None).compare(42, "4")

    assert success is True
    assert cleaned == "42"


def test_a_non_string_raw_output_does_not_crash_compare():
    success, _cleaned = build_matcher(None).compare("True story", True)

    assert success is True


@pytest.mark.parametrize(
    "cfg",
    [
        {"mode": "regex", "pattern": r"(\w+)", "group": "abc"},
        {"mode": "regex", "pattern": r"(\w+)", "timeout": "soon"},
        {"mode": "regex", "pattern": 123},
    ],
    ids=["bad-group", "bad-timeout", "non-string-pattern"],
)
def test_malformed_regex_config_values_raise_config_error_not_a_raw_exception(cfg):
    """``group``, ``timeout`` and ``pattern`` all funnel through ``int()``,
    ``float()`` or ``regex.compile()`` and previously let a ``ValueError`` or
    ``TypeError`` escape uncaught. Tasks 2 and 3 catch ``ConfigError`` as the
    single documented build-time contract, so these must route through it
    too."""
    with pytest.raises(ConfigError):
        build_matcher(cfg)


# --- fix round 2 -------------------------------------------------------


@pytest.mark.parametrize("bad_timeout", [0, -1])
def test_a_non_positive_timeout_is_rejected_at_build_time(bad_timeout):
    """A zero or negative timeout disables the backtracking guard entirely:
    ``regex.search(..., timeout=-1)`` never raises, so a catastrophic pattern
    hangs the row (and the run) instead of costing it a `MatchTimeout`. This
    is the one property the module claims to give a tenant's own pattern, so
    it must be validated at build time like every other matcher config."""
    with pytest.raises(ConfigError) as excinfo:
        build_matcher(
            {"mode": "regex", "pattern": r"(\w+)", "timeout": bad_timeout}
        )

    assert "timeout" in str(excinfo.value)
    assert str(bad_timeout) in str(excinfo.value)


def test_an_explicit_zero_timeout_is_not_silently_replaced_by_the_default():
    """``cfg.get('timeout') or DEFAULT`` cannot tell an explicit ``0`` apart
    from unset, unlike ``group`` three lines above which does. An explicit
    ``0`` must be rejected, not quietly coerced to the 2s default."""
    with pytest.raises(ConfigError):
        build_matcher({"mode": "regex", "pattern": r"(\w+)", "timeout": 0})


def test_regex_is_a_declared_dependency():
    """``matching.py`` imports ``regex`` unguarded for the ``timeout=`` kwarg
    the catastrophic-backtracking safety net depends on. It was present in
    this environment only transitively (via sacrebleu/tiktoken/nltk/...), so
    an upstream dependency bump dropping it would break this module at
    import time with an undiagnosable ``ModuleNotFoundError``."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())
    deps = pyproject["project"]["dependencies"]
    names = [re.split(r"[<>=!\[; ]", dep, maxsplit=1)[0] for dep in deps]

    assert "regex" in names


# --- review round ------------------------------------------------------


@pytest.mark.parametrize(
    "expected, output_matching_only_the_cleaned_form",
    [
        ("1. Paris", "Paris"),
        ("# 42", "42"),
        ("The tag is <br> here", "The tag is here"),
    ],
    ids=["ordered-list", "heading", "inline-tag"],
)
def test_the_expected_value_is_not_run_through_the_markdown_cleanup(
    expected, output_matching_only_the_cleaned_form
):
    """``clean_formatting`` is lossy by design: it renders markdown and drops
    anything HTML-shaped. That is right for model prose and wrong for an
    answer key. Applied to ``expected`` it rewrote the value the row is scored
    against - ``1. Paris`` became ``Paris`` - so the benchmark measured
    something the dataset never asked for. Pre-branch, ``expected`` only ever
    got ``.strip().lower()``.

    Each output here matches the *cleaned* expected and not the literal one,
    so it succeeds exactly when the cleanup is wrongly applied to the key.
    """
    success, _cleaned = build_matcher({"mode": "exact"}).compare(
        output_matching_only_the_cleaned_form, expected
    )

    assert success is False, "the answer key must be compared as written"


def test_an_html_shaped_expected_no_longer_matches_everything():
    """The worst instance of the above, because it compounds with the
    empty-``wanted`` fail-open below: ``<answer>`` cleaned to ``''``, and
    ``'' in anything`` is True, so a benchmark keyed on a literal tag scored
    every row correct under the default mode."""
    with pytest.raises(UnscorableRow):
        build_matcher(None).compare("the model said something else", "")

    success, _cleaned = build_matcher(None).compare(
        "the model said something else", "<answer>"
    )
    assert success is False


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_row_with_no_expected_answer_is_unscorable(blank):
    """A blank answer column reaches here as ``''`` (``_get_column_value``
    maps a null cell and the literal string 'null' onto its default). Every
    output contains the empty string, so ``contains`` scored the entire
    benchmark 1.0 off one blank column. Neither verdict is honest: raising
    lets both call sites record the row as unmeasured, which is what it is."""
    with pytest.raises(UnscorableRow):
        build_matcher(None).compare("any output at all", blank)


@pytest.mark.parametrize("falsy", [[], "", 0])
def test_a_falsy_non_mapping_matcher_is_rejected_not_ignored(falsy):
    """``cfg = cfg or {}`` ran before the ``isinstance`` check, so a falsy
    non-mapping (``matcher: []``, a plausible typo for ``matcher: {...}``)
    was swallowed as "no matcher" and ran the default, while the truthy
    ``matcher: exact`` was correctly rejected. A validator whose job is
    catching every-row misconfiguration accepted half of them."""
    with pytest.raises(ConfigError, match="mapping"):
        build_matcher(falsy)


def test_an_absent_matcher_block_is_still_the_default():
    """The other half of the check above: only ``None`` counts as unset."""
    assert build_matcher(None).mode == DEFAULT_MODE


@pytest.mark.parametrize("flag", ["m", "s"])
def test_newline_only_regex_flags_are_rejected_rather_than_silently_inert(flag):
    """``clean_formatting`` collapses every newline into a space before the
    pattern runs, so MULTILINE and DOTALL cannot change any outcome. Accepted,
    they validated happily and did nothing: an anchored ``^Answer: (\\w+)$``
    with ``flags: m`` scored every row 0.0 with reason 'No match' and no
    signal that the flag was inert."""
    with pytest.raises(ConfigError, match="no effect"):
        build_matcher({"mode": "regex", "pattern": r"^(\w+)$", "flags": flag})


def test_the_ignorecase_flag_still_works():
    """The rejection above must not take the one flag that does something."""
    matcher = build_matcher(
        {"mode": "regex", "pattern": r"answer: (\w+)", "flags": "i"}
    )

    success, extracted = matcher.compare("ANSWER: B", "B")
    assert success is True and extracted == "B"


# --- external review 2026-08-04 ----------------------------------------


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_timeout_is_rejected(bad):
    """`<= 0` alone does not cover these. Every NaN comparison is False, so
    NaN slipped straight through, and `inf` is positive so it passed too.
    Measured against `regex` 2025.11.3: `inf` raises `TimeoutError`
    immediately, the same as `0`, so every row would error and nothing would
    ever be measured."""
    with pytest.raises(ConfigError, match="finite"):
        build_matcher({"mode": "regex", "pattern": r"(\w+)", "timeout": bad})


@pytest.mark.parametrize("literal", [".inf", ".nan", "-.inf"])
def test_the_yaml_spellings_of_non_finite_reach_that_check(literal):
    """`timeout: .inf` in a config is not the string '.inf': YAML resolves it
    to a float before `build_matcher` sees it. Pinned so the check above is
    known to guard the shape a user can actually author, rather than one that
    only exists in a test."""
    import yaml

    value = yaml.safe_load(f"t: {literal}")["t"]

    assert isinstance(value, float)
    with pytest.raises(ConfigError, match="finite"):
        build_matcher({"mode": "regex", "pattern": r"(\w+)", "timeout": value})


def test_regex_mode_is_case_sensitive_without_the_i_flag():
    """`contains` and `exact` lowercase both sides unconditionally; a pattern
    owns its own case instead, because case is part of what a pattern
    expresses. Worth pinning: it is the one place the three modes disagree
    about case, and nothing else documents it."""
    sensitive = build_matcher({"mode": "regex", "pattern": r"(YES)"})
    insensitive = build_matcher({"mode": "regex", "pattern": r"(YES)", "flags": "i"})

    assert sensitive.compare("yes", "yes")[0] is False, "pattern case is the pattern's own"
    assert insensitive.compare("yes", "yes")[0] is True


def test_the_expected_value_is_still_compared_case_insensitively_under_regex():
    """The pattern owning its case must not leak into the comparison: an
    extracted `D` still matches an answer key authored as `d`."""
    matcher = build_matcher({"mode": "regex", "pattern": r"\b([A-D])\b"})

    assert matcher.compare("The answer is D", "d")[0] is True


# --- external review, follow-up pass ------------------------------------


@pytest.mark.parametrize("key", ["", "   ", ".", "...", " . ", "!?", ",", None])
def test_a_key_with_nothing_to_compare_is_unscorable(key):
    """Blank is the obvious case. A key of `'.'` is the same failure wearing
    a thinner disguise: a period occurs in almost every prose generation, so
    under `contains` it scores near everything correct."""
    with pytest.raises(UnscorableRow):
        build_matcher(None).compare("The answer is Paris.", key)


@pytest.mark.parametrize("key", ["+", "=", ">", "%", "@", "-", "0"])
def test_a_symbolic_answer_key_is_still_a_real_key(key):
    """The guard above must not swallow these. An operator or comparison
    benchmark legitimately keys on a single symbol, and unlike a period they
    are not near-universal substrings of prose.

    Asserted as "the key is still compared" rather than as a round trip:
    ``clean_formatting`` runs on the OUTPUT side and renders markdown, so a
    generation of exactly ``>`` or ``-`` is read as a blockquote or a list
    bullet and cleans to nothing. That is a separate, pre-existing property
    of the cleanup and not what this guard decides.
    """
    matcher = build_matcher({"mode": "exact"})

    # Reaches the comparison at all, rather than being rejected as unscorable.
    assert matcher.compare("something else", key)[0] is False
    assert matcher.compare(f"the answer is {key} today", key)[0] is False
    assert build_matcher(None).compare(f"a {key} b", key)[0] is True


MCQ_CORPUS = [
    # Real gpt-4o-mini generations from the live run.
    ("C. Paris", "C"),
    ("D. Pacific", "D"),
    ("The chemical symbol for gold is B. Au.", "B"),
    ("A. 300,000 km/s", "A"),
    ("Answer: D", "D"),
    ("B. Mars is known as the Red Planet.", "B"),
    # Reasoning BEFORE the answer. `flags: i` broke all of these by matching
    # the article "a" or the lowercase option marker.
    ("I think a good answer is B", "B"),
    ("Looking at a few options, C is correct", "C"),
    ("a) no  d) yes, so D", "D"),
    ("The answer is B", "B"),
    # Reasoning AFTER the answer, naming the rejected options. Taking the
    # LAST match instead of the first breaks all of these, which is how the
    # first attempt at fixing the "a" problem went wrong: its corpus only
    # contained the shapes above, so it pinned the fix without probing the
    # direction the fix opened.
    ("So the answer is B. (Not A, C, or D.)", "B"),
    ("B. The other options A, C, D are incorrect.", "B"),
    ("I choose B. The options were A, B, C, D.", "B"),
    ("B is my answer. Ruling out A, C and D.", "B"),
    ("The answer is C, not A or B.", "C"),
]

MCQ_PATTERN = r"\b([ABCD])\b"


@pytest.mark.parametrize("output, key", MCQ_CORPUS)
def test_the_documented_mcq_pattern_survives_real_model_prose(output, key):
    """The pattern in `examples/` is the one users copy, so it is worth a
    test rather than an eyeball.

    Extracting one letter from free-form prose is a heuristic and no
    position-based rule survives every shape. This corpus deliberately holds
    BOTH failure directions, so a change that fixes one by trading it for the
    other fails here rather than looking like an improvement.
    """
    matcher = build_matcher({"mode": "regex", "pattern": MCQ_PATTERN})

    assert matcher.compare(output, key)[0] is True


@pytest.mark.parametrize(
    "output, key, extracted",
    [("A good answer is B", "B", "A")],
)
def test_the_mcq_pattern_has_a_known_and_accepted_blind_spot(output, key, extracted):
    """Pinned so it is a documented cost rather than a latent surprise.

    A sentence-opening "A" is a capitalised article, indistinguishable from
    an answer by position alone. The obvious repair is to take the last match
    instead, which costs five of the elimination-phrasing cases above. This
    test exists to make that trade visible to whoever tries it.
    """
    matcher = build_matcher({"mode": "regex", "pattern": MCQ_PATTERN})
    success, got = matcher.compare(output, key)

    assert success is False and got == extracted


def test_the_mcq_pattern_still_rejects_a_wrong_answer():
    """Tolerating prose must not mean matching anything."""
    matcher = build_matcher({"mode": "regex", "pattern": MCQ_PATTERN})

    assert matcher.compare("The answer is C", "B")[0] is False
    assert matcher.compare("I am not sure", "B")[0] is False


def test_the_shipped_example_configs_use_that_pattern():
    """The tests above are worth nothing if the file users copy differs."""
    import yaml
    from pathlib import Path

    examples = Path(__file__).resolve().parents[1] / "examples"
    for name in ("custom_eval_test.yaml", "custom_eval_test_gpt.yaml"):
        cfg = yaml.safe_load((examples / name).read_text())
        target = next(t for t in cfg["targets"] if t.get("evaluations"))
        matcher = target["evaluations"][0]["benchmarks"][0]["matcher"]

        assert matcher["pattern"] == MCQ_PATTERN, name
        assert "flags" not in matcher, f"{name}: `i` makes prose extract the article 'a'"
