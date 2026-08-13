"""What counts as a passing row, in one place.

A graded score has to be turned into a pass/fail somewhere, and the two
backends did it differently and silently: the evalscope path used
``score > 0`` while custom_eval's judge path used ``score >= 0.5``. On a
judge-scored benchmark the first rule passes essentially every row, because
a judge rating a coherent answer gives it *something* -- 0.4, 0.6, 0.7 --
and only a literal zero fails. An MT-Bench run observed on 2026-08-13
reported 10 of 10 samples passed while its own average was 6.5 out of 10.

The threshold is expressed as a fraction of the metric's scale (0-1),
matching the scale the runner's per-row scores already use, so one number
is comparable across a ``score/10`` benchmark and a ``score/5`` one.
"""

from typing import Optional

#: Applied when no threshold is configured, so an existing config keeps
#: behaving exactly as it did. Not a good rule -- it is what the judge path
#: already used, kept only so that turning the threshold on is an explicit
#: choice rather than a silent re-scoring of everyone's history.
LEGACY_JUDGE_MIN = 0.5


def row_passed(
    score: Optional[float],
    pass_threshold: Optional[float] = None,
    *,
    legacy_minimum: float = 0.0,
    legacy_inclusive: bool = False,
) -> bool:
    """Whether one row's *score* counts as a pass.

    ``None`` is never a pass: a row we could not read is unmeasured, which
    is a different claim from a row that scored badly.

    With a *pass_threshold*, the row passes when it reaches it -- always
    inclusive, because a threshold reads as "at least this good".

    Without one, the caller's own historical rule applies, stated as a value
    plus whether it is inclusive: the evalscope path passed anything
    strictly above 0, the judge path anything at or above 0.5. Both are
    spelled out at the call site rather than inferred, so a caller passing
    some third value gets the operator it asked for instead of whichever one
    its number happened to match.
    """
    if score is None:
        return False
    if pass_threshold is not None:
        return score >= pass_threshold
    return score >= legacy_minimum if legacy_inclusive else score > legacy_minimum
