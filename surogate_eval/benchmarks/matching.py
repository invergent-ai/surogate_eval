"""How a custom benchmark's string rows decide whether an answer is correct.

Kept out of the backend deliberately. The rule is a separable concern with a
clean boundary - text in, verdict out - and isolating it is what lets it be
tested without a dataset, a target or a temp file.

Three modes, because "exact match" previously meant `expected in output`
after a normalisation pass that also guessed the answer out of the output.
Formatting cleanup survives that split; guessing does not, since that is what
``regex`` mode now does explicitly and visibly.
"""

import math
import re as stdlib_re
from typing import Any, Dict, Optional, Tuple

import regex

from surogate_eval.errors import ConfigError, MatchTimeout, UnscorableRow
from surogate_eval.utils.logger import get_logger

logger = get_logger()

#: The mode used when a benchmark names none.
#:
#: On the direct-inference path this is today's behaviour, so an existing
#: config keeps its results until it opts into something stricter.
#:
#: On the lm-eval path it is LOOSER than what that path did before. Scoring
#: there used to come from lm-eval's own ``exact_match`` metric, which is
#: full-string equality; unifying the two paths onto one matcher means a
#: config that names no mode now gets containment there. On a single-letter
#: answer key that is close to unscoreable - expected ``B`` is contained in
#: "probably", "best" and "because" - so an MCQ benchmark scores near 100%
#: whatever the model says. Any config with a tokenizer and short answer keys
#: should set a mode explicitly; see ``examples/custom_eval_test_gpt.yaml``.
DEFAULT_MODE = 'contains'

VALID_MODES = ('contains', 'exact', 'regex')

#: Seconds one pattern may spend on one row. The pattern is the tenant's own
#: and runs on their own pod, so a runaway is a foot-gun rather than an
#: attack - but it must still cost a row instead of the run.
DEFAULT_TIMEOUT_SECONDS = 2.0

_FLAG_CHARS = {
    'i': regex.IGNORECASE,
}

#: `m` and `s` are the two stdlib flags that only change behaviour around
#: newlines, and `clean_formatting` collapses every newline into a space
#: before the pattern ever runs. Accepting them would validate happily and
#: then do nothing, so an anchored `^Answer: (\w+)$` would score every row 0.0
#: with no signal that the flag was inert. Rejected at build time instead.
_INERT_FLAG_CHARS = ('m', 's')


def clean_formatting(text: str) -> str:
    """Strip markdown and collapse whitespace.

    Formatting only. This deliberately does not try to find the answer inside
    the text: that is the matcher's job, and having two of them is how you get
    a result nobody can explain.

    Model output only. It is lossy by design - it renders markdown and drops
    anything HTML-shaped - so running it on an *expected* value would rewrite
    the answer key the row is scored against. See ``Matcher.compare``.
    """
    from bs4 import BeautifulSoup
    import markdown

    html = markdown.markdown(text)
    # Not `separator=' '`: that padded every inline-markup boundary with a
    # space (`**A**.` -> `A .`), which `contains` never noticed
    # (`"a" in "a ."`) but `exact` and `regex` compare verbatim, so a correct
    # answer was scored wrong. `separator=''` does not merge words across
    # BLOCK boundaries instead: `markdown.markdown` always joins top-level
    # block elements (paragraphs, list items, headings) with a literal `\n`
    # in the HTML it emits, and that text node survives regardless of
    # `separator`; the whitespace collapse below turns it back into a space.
    cleaned = BeautifulSoup(html, 'html.parser').get_text(separator='')

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
        when a pattern exceeds its budget, and ``UnscorableRow`` when the row
        carries no answer key; the caller treats both as a failure to measure.
        """
        # `or ''` only guards falsy values; a truthy non-string (a numeric
        # answer key authored as `42` rather than `"42"`, or a bool) must
        # still not crash inside `clean_formatting`, so coerce explicitly.
        cleaned = clean_formatting('' if raw_output is None else str(raw_output))
        # Only `.strip().lower()`, deliberately not `clean_formatting`. The
        # expected value is an answer key, not model prose, and the cleanup
        # is lossy: it renders markdown and drops HTML-shaped tokens, so a
        # literal `<answer>` cleaned to `''` and a `1. Paris` to `Paris`.
        # An erased key then matched everything under `contains`, turning a
        # benchmark that should score 0% into one reporting 100%.
        wanted = ('' if expected is None else str(expected)).strip().lower()

        if not wanted:
            # Every output contains the empty string, so `contains` would
            # score the whole benchmark 1.0 off one blank answer column.
            # A blank key is a dataset defect, and neither verdict is honest.
            raise UnscorableRow('row has no expected answer to compare against')

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
            extracted = (found.group(self._group) or '').strip()
            return extracted.lower() == wanted, extracted

        got = cleaned.strip().lower()
        if self.mode == 'exact':
            return got == wanted, cleaned
        return wanted in got, cleaned


def build_matcher(cfg: Optional[Dict[str, Any]]) -> Matcher:
    """Validate a benchmark's ``matcher`` block once, before its rows are scored.

    Every row would hit a bad pattern or an unknown mode, so both are config
    errors rather than per-row failures.

    Not a load-time check: this runs when a benchmark first scores a string
    row, so a matcher block on a benchmark that scores no string rows is never
    validated. Harmless today, since such a block has no effect either way. If
    that stops being true, ``EvalConfig._validate_evaluations`` already walks
    every benchmark dict at load and is where the eager check belongs.
    """
    # Order matters: `cfg or {}` first would swallow every *falsy* non-mapping
    # (`matcher: []`, `matcher: ""`) as "no matcher" and quietly run the
    # default, while rejecting only the truthy ones (`matcher: exact`). Only
    # an absent block counts as unset.
    if cfg is None:
        cfg = {}
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
        if char in _INERT_FLAG_CHARS:
            raise ConfigError(
                f"regex flag {char!r} has no effect here: the output's newlines "
                f"are collapsed to spaces before the pattern runs, so write the "
                f"pattern against a single line"
            )
        if char not in _FLAG_CHARS:
            raise ConfigError(
                f"unknown regex flag {char!r}; expected any of {', '.join(_FLAG_CHARS)}"
            )
        flags |= _FLAG_CHARS[char]

    try:
        compiled = regex.compile(pattern, flags)
    except (regex.error, TypeError) as exc:
        raise ConfigError(f"invalid matcher pattern {pattern!r}: {exc}") from exc

    # Default to the first capture group when the pattern has one, and to the
    # whole match when it does not, so a simple pattern needs no `group`.
    group = cfg.get('group')
    if group is None:
        group = 1 if compiled.groups else 0
    else:
        try:
            group = int(group)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"matcher group must be an integer, got {group!r}"
            ) from exc
    # Both bounds matter: a pattern has groups 0 (the whole match) through
    # `compiled.groups`, and a negative index passes straight through to
    # `re.Match.group()`, which raises `IndexError` on the first row rather
    # than failing here, at build time, where every row would hit it.
    if group < 0 or group > compiled.groups:
        raise ConfigError(
            f"matcher group {group} but pattern {pattern!r} has "
            f"{compiled.groups} capture group(s)"
        )

    # `cfg.get('timeout')` rather than `... or DEFAULT_TIMEOUT_SECONDS`: the
    # latter cannot tell an explicit `0` apart from unset, unlike `group`
    # above, and would silently replace it with the default instead of
    # rejecting it below.
    timeout_cfg = cfg.get('timeout')
    if timeout_cfg is None:
        timeout = DEFAULT_TIMEOUT_SECONDS
    else:
        try:
            timeout = float(timeout_cfg)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"matcher timeout must be a number, got {timeout_cfg!r}"
            ) from exc
        # Both ends are rejected, for opposite reasons, both measured against
        # `regex` 2025.11.3 rather than assumed:
        #
        #   timeout=-1   no limit at all, so a catastrophic pattern hangs the
        #                row and the safety property this module claims is
        #                silently off.
        #   timeout=0    the reverse - `TimeoutError` fires immediately, even
        #                on a trivial instant match, so every row errors and
        #                nothing is ever measured.
        #   timeout=inf  also raises immediately, same as 0.
        #   timeout=nan  slips past a `<= 0` test, since every NaN comparison
        #                is False.
        #
        # So the guard is "finite and positive", not just "positive".
        if not math.isfinite(timeout) or timeout <= 0:
            raise ConfigError(
                f"matcher timeout must be a finite positive number, "
                f"got {timeout_cfg!r}"
            )
    logger.debug(f"Matcher: mode={mode} group={group} timeout={timeout}s")
    return Matcher(mode, compiled=compiled, group=group, timeout=timeout)
