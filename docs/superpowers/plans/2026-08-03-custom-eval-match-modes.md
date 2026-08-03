# Custom-eval match modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a custom benchmark's string scoring three named modes (`contains`, `exact`, `regex`) instead of one unnamed substring rule, and apply the same one on both scoring paths.

**Architecture:** The comparison moves into a new `benchmarks/matching.py` as a small validated object built once per benchmark and called per row. `custom_eval_backend.py` is already 934 lines and holds dataset loading, three row evaluators and result assembly; the comparison rule is a separable concern with a clean boundary, and putting it in its own module is what makes it testable without a dataset. Task 1 builds it standalone, Task 2 wires the direct path, Task 3 wires the lm-eval path.

**Tech Stack:** Python 3.12, pytest, the `regex` module (already installed, 2025.11.3) for its `timeout` parameter.

## Global Constraints

- **The default is `contains`**, which is today's behaviour minus the retired heuristics. This plan
  must not change the result of any existing config that sets no `matcher`.
- **A failure to measure is never a score.** An unmeasured row is `score=None`, `success=False`,
  `status='errored'` with a reason, matching `_evaluate_exact_match_direct`'s existing rows.
- **A non-matching pattern is a wrong answer** (`score=0.0`), not an unmeasured row. An invalid
  pattern or unknown mode fails the benchmark at start. A match timeout errors that row.
- **No test may make a network call.** `tests/conftest.py` enforces the offline environment; every
  test here passes plain strings.
- Run tests with `./.venv/bin/python -m pytest`. Baseline on this branch: 226 passing.
- Conventional Commits, plain backticks in messages. Never mention AI or tooling in a commit message;
  no `Co-Authored-By` or `Generated-with` trailers.

## File structure

| File | Responsibility |
|---|---|
| `surogate_eval/benchmarks/matching.py` (new) | The comparison rule: formatting cleanup, the three modes, validation, timeout. No knowledge of datasets, rows or backends. |
| `surogate_eval/benchmarks/base.py` | Carries `matcher` on `BenchmarkConfig`, beside `eval_type`. |
| `surogate_eval/benchmarks/generic.py` | Passes `matcher` to the backend, beside `eval_type`. |
| `surogate_eval/runners.py` | Reads `matcher` off the benchmark config dict. |
| `surogate_eval/benchmarks/backends/custom_eval_backend.py` | Builds the matcher once, uses it on both exact-match paths. Loses its heuristic extraction. |
| `tests/test_custom_eval_matching.py` (new) | The modes, in isolation. |
| `tests/test_custom_eval_match_paths.py` (new) | Both scoring paths agreeing, through the real evaluators. |

---

### Task 1: The matcher

**Files:**
- Create: `surogate_eval/benchmarks/matching.py`
- Modify: `surogate_eval/errors.py`
- Test: `tests/test_custom_eval_matching.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `build_matcher(cfg: Optional[Dict[str, Any]]) -> Matcher`, raising `ConfigError` on an
  unknown mode or invalid pattern. `Matcher.compare(raw_output: str, expected: str) -> Tuple[bool, str]`
  returning `(success, cleaned_output)` and raising `MatchTimeout` when a pattern exceeds the budget.
  `Matcher.mode: str`. Also `clean_formatting(text: str) -> str`. Tasks 2 and 3 use all of these.

- [x] **Step 1: Write the failing test**

Create `tests/test_custom_eval_matching.py`:

```python
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_custom_eval_matching.py -q`
Expected: FAIL at import, `ModuleNotFoundError: No module named 'surogate_eval.benchmarks.matching'`

- [x] **Step 3: Add the two error types**

Append to `surogate_eval/errors.py`:

```python
class MatchTimeout(EvalError):
    """A user-supplied pattern took longer than its budget on one row.

    Raised rather than returning "no match", because a pattern we abandoned
    tells us nothing about whether the answer was right. The caller records
    the row as unmeasured.
    """
```

`ConfigError` already exists in that file and is used for the build-time failures.

- [x] **Step 4: Write the matcher**

Create `surogate_eval/benchmarks/matching.py`:

```python
"""How a custom benchmark's string rows decide whether an answer is correct.

Kept out of the backend deliberately. The rule is a separable concern with a
clean boundary - text in, verdict out - and isolating it is what lets it be
tested without a dataset, a target or a temp file.

Three modes, because "exact match" previously meant `expected in output`
after a normalisation pass that also guessed the answer out of the output.
Formatting cleanup survives that split; guessing does not, since that is what
``regex`` mode now does explicitly and visibly.
"""

import re as stdlib_re
from typing import Any, Dict, Optional, Tuple

import regex

from surogate_eval.errors import ConfigError, MatchTimeout
from surogate_eval.utils.logger import get_logger

logger = get_logger()

#: The mode used when a benchmark names none. Today's behaviour, so an
#: existing config keeps its results until it opts into something stricter.
DEFAULT_MODE = 'contains'

VALID_MODES = ('contains', 'exact', 'regex')

#: Seconds one pattern may spend on one row. The pattern is the tenant's own
#: and runs on their own pod, so a runaway is a foot-gun rather than an
#: attack - but it must still cost a row instead of the run.
DEFAULT_TIMEOUT_SECONDS = 2.0

_FLAG_CHARS = {
    'i': regex.IGNORECASE,
    'm': regex.MULTILINE,
    's': regex.DOTALL,
}


def clean_formatting(text: str) -> str:
    """Strip markdown and collapse whitespace.

    Formatting only. This deliberately does not try to find the answer inside
    the text: that is the matcher's job, and having two of them is how you get
    a result nobody can explain.
    """
    try:
        from bs4 import BeautifulSoup
        import markdown

        html = markdown.markdown(text)
        cleaned = BeautifulSoup(html, 'html.parser').get_text(separator=' ')
    except ImportError:
        cleaned = text
        cleaned = stdlib_re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        cleaned = stdlib_re.sub(r'\*([^*]+)\*', r'\1', cleaned)
        cleaned = stdlib_re.sub(r'`([^`]+)`', r'\1', cleaned)
        cleaned = stdlib_re.sub(r'#{1,6}\s*', '', cleaned)
        cleaned = stdlib_re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)

    return stdlib_re.sub(r'\s+', ' ', cleaned).strip()


class Matcher:
    """Compares one model output against one expected answer."""

    def __init__(self, mode: str, compiled=None, group: int = 0,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.mode = mode
        self._compiled = compiled
        self._group = group
        self._timeout = timeout

    def compare(self, raw_output: str, expected: str) -> Tuple[bool, str]:
        """Return ``(success, cleaned_output)``.

        ``cleaned_output`` is what was actually compared, so the row's record
        can show it rather than the raw generation. Raises ``MatchTimeout``
        when a pattern exceeds its budget; the caller treats that as a failure
        to measure, since an abandoned match says nothing either way.
        """
        cleaned = clean_formatting(raw_output or '')
        wanted = clean_formatting(expected or '').strip().lower()

        if self.mode == 'regex':
            try:
                found = self._compiled.search(cleaned, timeout=self._timeout)
            except TimeoutError as exc:
                raise MatchTimeout(
                    f"pattern exceeded {self._timeout}s on one row"
                ) from exc
            if not found:
                # The pattern is the answer format the benchmark asked for, so
                # producing nothing that matches it is a wrong answer.
                return False, ''
            extracted = found.group(self._group) or ''
            return extracted.strip().lower() == wanted, extracted.strip()

        got = cleaned.strip().lower()
        if self.mode == 'exact':
            return got == wanted, cleaned
        return wanted in got, cleaned


def build_matcher(cfg: Optional[Dict[str, Any]]) -> Matcher:
    """Validate a benchmark's ``matcher`` block once, at benchmark start.

    Every row would hit a bad pattern or an unknown mode, so both are config
    errors rather than per-row failures.
    """
    cfg = cfg or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"matcher must be a mapping, got {type(cfg).__name__}")

    mode = cfg.get('mode', DEFAULT_MODE)
    if mode not in VALID_MODES:
        raise ConfigError(
            f"unknown matcher mode {mode!r}; expected one of {', '.join(VALID_MODES)}"
        )

    if mode != 'regex':
        return Matcher(mode)

    pattern = cfg.get('pattern')
    if not pattern:
        raise ConfigError("matcher mode 'regex' requires a 'pattern'")

    flags = 0
    for char in str(cfg.get('flags') or ''):
        if char not in _FLAG_CHARS:
            raise ConfigError(
                f"unknown regex flag {char!r}; expected any of {', '.join(_FLAG_CHARS)}"
            )
        flags |= _FLAG_CHARS[char]

    try:
        compiled = regex.compile(pattern, flags)
    except regex.error as exc:
        raise ConfigError(f"invalid matcher pattern {pattern!r}: {exc}") from exc

    # Default to the first capture group when the pattern has one, and to the
    # whole match when it does not, so a simple pattern needs no `group`.
    group = cfg.get('group')
    if group is None:
        group = 1 if compiled.groups else 0
    group = int(group)
    if group > compiled.groups:
        raise ConfigError(
            f"matcher group {group} but pattern {pattern!r} has "
            f"{compiled.groups} capture group(s)"
        )

    timeout = float(cfg.get('timeout') or DEFAULT_TIMEOUT_SECONDS)
    logger.debug(f"Matcher: mode={mode} group={group} timeout={timeout}s")
    return Matcher(mode, compiled=compiled, group=group, timeout=timeout)
```

- [x] **Step 5: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_custom_eval_matching.py -q`
Expected: PASS, 17 tests

If `test_a_catastrophic_pattern_is_bounded_rather_than_hanging_the_run` does not raise, the `regex`
module optimised that particular pattern away. Do not weaken the assertion: replace the pattern with
`r"(a|a)+$"` and re-run, and if it still completes, raise the input length rather than deleting the
test.

- [x] **Step 6: Run the full suite**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS, 226 existing plus the new file, no regressions

- [x] **Step 7: Commit**

```bash
git add surogate_eval/benchmarks/matching.py surogate_eval/errors.py tests/test_custom_eval_matching.py
git commit -m "feat(benchmarks): add named match modes for custom-eval string scoring

A custom benchmark's string rows were scored by `expected in output` after a
normalisation pass that also guessed the answer out of the output, so expected
'A' matched 'The answer is C.' and every row of an MCQ benchmark passed.

Adds contains, exact and regex as named modes, validated once at benchmark
start: an unknown mode or an invalid pattern is a config error rather than a
silent fall back to containment. regex extracts a capture group and compares
it, so a pattern that does not match is a wrong answer, while a pattern that
exceeds its timeout is a failure to measure.

Not wired into the backend yet."
```

---

### Task 2: The direct path uses it, and the heuristics retire

**Files:**
- Modify: `surogate_eval/benchmarks/base.py:60` (add the field), `surogate_eval/benchmarks/generic.py:72` (pass it), `surogate_eval/runners.py:381` (read it), `surogate_eval/benchmarks/backends/custom_eval_backend.py:177-232` (delete `_normalize_output`) and `:449-455` (the comparison)
- Test: `tests/test_custom_eval_match_paths.py` (create)

**Interfaces:**
- Consumes: `build_matcher`, `Matcher.compare`, `clean_formatting` from Task 1.
- Produces: `BenchmarkConfig.matcher: Optional[Dict[str, Any]] = None`, reaching the backend as
  `config['matcher']`. Task 3 reads the same built matcher.

- [x] **Step 1: Write the failing test**

Create `tests/test_custom_eval_match_paths.py`:

```python
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_custom_eval_match_paths.py -q`
Expected: FAIL — `test_exact_mode_reaches_the_comparison` reports `success is True`, because the
matcher is ignored and containment still applies.

- [x] **Step 3: Carry `matcher` on the config**

In `surogate_eval/benchmarks/base.py`, after line 60 (`eval_type: str = 'exact_match'`), add:

```python
    #: How string rows decide correctness. See benchmarks/matching.py.
    matcher: Optional[Dict[str, Any]] = None
```

In `surogate_eval/benchmarks/generic.py`, after line 72 (`'eval_type': self.config.eval_type,`), add:

```python
            'matcher': self.config.matcher,
```

In `surogate_eval/runners.py`, after line 381 (`eval_type=bench_config.get("eval_type", "exact_match"),`), add:

```python
            matcher=bench_config.get("matcher"),
```

- [x] **Step 4: Use the matcher in the direct path**

In `custom_eval_backend.py`, add to the imports at the top of the file:

```python
from surogate_eval.benchmarks.matching import build_matcher
```

In `_evaluate_exact_match_direct`, immediately after the `system_prompt` line near the top of the
method, add:

```python
        # Built once: an unknown mode or a bad pattern would fail every row,
        # so it is a config error and belongs here rather than in the loop.
        matcher = build_matcher(config.get('matcher'))
```

Then replace the comparison block (currently `:448-455`, inside the `try`):

```python
                    normalized_output = self._normalize_output(raw_output, expected)
                    expected_clean = expected.strip().lower()
                    output_clean = normalized_output.strip().lower()
                    success = (
                        expected_clean == output_clean
                        or expected_clean in output_clean
                        or output_clean.startswith(expected_clean)
                    )
```

with:

```python
                    success, normalized_output = matcher.compare(raw_output, expected)
```

`MatchTimeout` is an `EvalError` and this block is already wrapped by
`except Exception as e: row_error = f'Comparison error: {e}'`, so a timed-out row becomes an errored
row with no further change.

- [x] **Step 5: Delete the heuristics**

Delete `_normalize_output` entirely from `custom_eval_backend.py` (currently lines 177-232). Its
formatting half now lives in `matching.clean_formatting`; its email, percentage, date and yes/no
extraction is what `regex` mode replaces.

Confirm nothing else calls it:

Run: `grep -rn "_normalize_output" surogate_eval/ tests/`
Expected: no output.

- [x] **Step 6: Run the tests**

Run: `./.venv/bin/python -m pytest tests/test_custom_eval_match_paths.py tests/test_custom_eval_matching.py -q`
Expected: PASS

- [x] **Step 7: Run the full suite**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS, no regressions.

**One existing test names `_normalize_output` and must keep passing unchanged.**
`tests/test_custom_eval_inference_errors.py:106-118` asserts that a numeric `answer` column (pandas
infers int64, so `expected` arrives as an `int`) errors only its own row and leaves the healthy row
scored. That still holds: `Matcher.compare` calls `clean_formatting(expected)`, which raises on an
`int` exactly as `.strip()` did, and the caller's `except Exception` turns it into the same errored
row. Its **docstring** references `_normalize_output` by name and goes stale — update the prose to
name `clean_formatting`, and do not touch the assertions. If any assertion actually fails, stop and
report it rather than editing it.

- [x] **Step 8: Commit**

```bash
git add surogate_eval/benchmarks/base.py surogate_eval/benchmarks/generic.py surogate_eval/runners.py surogate_eval/benchmarks/backends/custom_eval_backend.py tests/test_custom_eval_match_paths.py
git commit -m "fix(benchmarks): score custom-eval rows with the configured match mode

Carries a matcher block from the benchmark config through to the direct
inference path, so a benchmark can ask for exact or regex scoring instead of
getting substring containment under the name exact match.

Retires _normalize_output's answer-guessing. It stripped markdown, which is
formatting and now lives in clean_formatting, but it also pulled emails,
percentages, dates and yes/no out of the output and compared that instead -
a second, invisible extractor competing with the one the user configures."
```

---

### Task 3: The lm-eval path uses the same matcher

**Files:**
- Modify: `surogate_eval/benchmarks/backends/custom_eval_backend.py:294-384` (`_evaluate_exact_match_lm_eval`)
- Test: `tests/test_custom_eval_match_paths.py`

**Interfaces:**
- Consumes: `build_matcher` and `Matcher.compare` from Task 1; the `config['matcher']` plumbing from
  Task 2.
- Produces: nothing new.

- [x] **Step 1: Write the failing test**

Append to `tests/test_custom_eval_match_paths.py`:

```python
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

    original = ceb.LMEvalBackend if hasattr(ceb, "LMEvalBackend") else None
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
    """A benchmark's result must not depend on whether a tokenizer is set."""
    direct = _score("The answer is C.", "A", {"mode": "exact"})
    lm_eval = _score_lm_eval("The answer is C.", "A", {"mode": "exact"})[0]

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
```

- [x] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_custom_eval_match_paths.py -q -k "lm_eval or both_paths"`
Expected: FAIL — the first reports `success is True` (lm-eval's metric decided), and the last
reports `score == 0.0` rather than `None`.

- [x] **Step 3: Score with the matcher and record an unreturned row as unmeasured**

In `_evaluate_exact_match_lm_eval`, add near the top of the method, beside the existing
`logger.info`:

```python
        matcher = build_matcher(config.get('matcher'))
```

Then replace the result-mapping block (currently `:347-362`):

```python
            for i, row in enumerate(lm_eval_rows):
                if i < len(detailed_results):
                    detail = detailed_results[i]
                    score = 1.0 if detail.get('metrics', {}).get('exact_match', 0) else 0.0
                    success = bool(detail.get('metrics', {}).get('exact_match', 0))
                    output = detail.get('output', '')
                    raw_output = detail.get('raw_output', '')
                    reason = 'Exact match' if success else 'No match'
                else:
                    score = 0.0
                    success = False
                    output = ''
                    raw_output = ''
                    reason = 'No result'
```

with:

```python
            for i, row in enumerate(lm_eval_rows):
                if i >= len(detailed_results):
                    # lm-eval returned fewer rows than we sent. This row was
                    # never scored, which is not the same as scoring it wrong.
                    results.append({
                        'original_idx': row['_original_idx'],
                        'eval_type': 'exact_match',
                        'instruction': row['instruction'],
                        'expected': row['answer'],
                        'output': '',
                        'raw_output': '',
                        'status': 'errored',
                        'score': None,
                        'success': False,
                        'reason': 'lm-eval returned no result for this row',
                    })
                    continue

                detail = detailed_results[i]
                # lm-eval generated; the comparison is ours. Its own
                # exact_match metric is deliberately not read, and neither is
                # its `output`, which is its own extraction heuristic - a
                # third extractor beside the matcher.
                raw_output = detail.get('raw_output', '') or ''
                status = 'scored'
                reason = None
                try:
                    success, output = matcher.compare(raw_output, row['answer'])
                    score = 1.0 if success else 0.0
                    reason = 'Match' if success else 'No match'
                except Exception as e:
                    status = 'errored'
                    score = None
                    success = False
                    output = ''
                    reason = f'Comparison error: {e}'
```

And update the record built just below it to carry the new fields:

```python
                result = {
                    'original_idx': row['_original_idx'],
                    'eval_type': 'exact_match',
                    'instruction': row['instruction'],
                    'expected': row['answer'],
                    'output': output,
                    'raw_output': raw_output,
                    'status': status,
                    'score': score,
                    'success': success,
                    'reason': reason,
                }
                results.append(result)
```

The trailing `logger.info` summing `r['success']` still works, since `success` is always a bool.

- [x] **Step 4: Run the tests**

Run: `./.venv/bin/python -m pytest tests/test_custom_eval_match_paths.py -q`
Expected: PASS

- [x] **Step 5: Run the full suite**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS, no regressions

- [x] **Step 6: Commit**

```bash
git add surogate_eval/benchmarks/backends/custom_eval_backend.py tests/test_custom_eval_match_paths.py
git commit -m "fix(benchmarks): score the lm-eval path with the same matcher

The tokenizer path handed correctness to lm-eval's own exact_match metric, so
the same benchmark was scored by different rules depending on whether a
tokenizer was set. lm-eval keeps generation; the comparison comes back here,
against raw_output rather than lm-eval's own extraction heuristic, which was a
third extractor beside the matcher.

A row lm-eval never returned was recorded as score 0.0 with reason 'No result'
- an answer the model got wrong, for a row nobody scored. It is unmeasured
now, matching the direct path's convention, and carries a status like every
other row this backend emits."
```

---

## Self-Review

**Spec coverage.** The `matcher` block and its plumbing are Task 2 Step 3. The three modes,
extraction semantics, validation and timeout are Task 1. The heuristics retiring is Task 2 Step 5.
Both-paths-one-matcher and the unreturned-row fix are Task 3. The spec's testing list maps onto
Task 1's cases (false positives under each mode, markdown survival, group defaulting, no-match,
invalid pattern, unknown mode, timeout, the retired-heuristic behaviour change) and Task 3's
(both paths agreeing, unmeasured rows). `greeting-check`'s shape is covered by
`test_the_default_is_still_containment` plus the `contains` parametrisation, since its expected value
is long and its output contains it.

**Placeholders.** None. Every code step carries its code, including the two replaced blocks quoted in
full so they can be found.

**Type consistency.** `build_matcher(cfg) -> Matcher` and `Matcher.compare(raw_output, expected) ->
(bool, str)` are defined in Task 1 Step 4 and called with those exact shapes in Task 2 Step 4 and
Task 3 Step 3. `MatchTimeout` is defined in Task 1 Step 3 and relied on in Task 1's timeout test and
Task 2 Step 4's note. `BenchmarkConfig.matcher` is added in Task 2 Step 3 and read as
`config.get('matcher')` in Tasks 2 and 3.

**Out of scope, unchanged:** sending a mode from the platform, flipping the default to `exact`, and
the create form's fields.
