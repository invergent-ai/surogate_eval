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
