"""Small text helpers shared across config, benchmark and metric code."""

from typing import Any, Optional


def blank_as_none(value: Any) -> Optional[Any]:
    """``None`` for a string that is empty or only whitespace, else *value*.

    Written for the ``blank_as_none(x) or DEFAULT`` chain, which is the only
    form that actually reaches a default for user-supplied text.

    ``config.get(key, DEFAULT)`` reaches it when the key is absent and no
    other time, so a key present with ``None`` skips it. A plain
    ``x or DEFAULT`` fixes that and ``""`` as well, but not ``"   "``, and a
    space is what a form field and a spreadsheet cell actually produce. The
    libraries downstream do not distinguish: deepeval rejects blank criteria
    with ``criteria is not None and not criteria.strip()``, so all three
    spellings of "the user set nothing" fail identically.

    Non-strings pass through untouched, so a deliberate falsy setting
    (``0``, ``False``, ``[]``) is not mistaken for unset.
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value
