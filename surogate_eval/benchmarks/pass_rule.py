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

#: Applied when no threshold is configured, per backend, so an existing
#: config keeps behaving exactly as it did. Neither is a good rule -- they
#: are what each path already used, kept only so that turning this on is an
#: explicit choice rather than a silent re-scoring of everyone's history.
LEGACY_EVALSCOPE_MIN = 0.0    # strictly greater than: any non-zero score passed
LEGACY_JUDGE_MIN = 0.5        # at least: custom_eval's judge path


def row_passed(
    score: Optional[float],
    pass_threshold: Optional[float] = None,
    *,
    legacy_minimum: float = LEGACY_EVALSCOPE_MIN,
) -> bool:
    """Whether one row's *score* counts as a pass.

    ``None`` is never a pass: a row we could not read is unmeasured, which
    is a different claim from a row that scored badly.

    With a *pass_threshold*, the row passes when it reaches it. Without one,
    the caller's *legacy_minimum* applies, which is strictly-greater-than
    for the evalscope path and at-least for the judge path -- the two rules
    those paths already had.
    """
    if score is None:
        return False
    if pass_threshold is not None:
        return score >= pass_threshold
    if legacy_minimum == LEGACY_EVALSCOPE_MIN:
        return score > legacy_minimum
    return score >= legacy_minimum
