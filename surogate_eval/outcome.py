"""Run-level outcome: did this evaluation produce trustworthy results?

The runner used to end every run with "completed" and exit 0, even when no
target was reachable and every metric had errored. Ops reads the exit code
to decide whether a run failed, so the outcome computed here is what makes
a broken run visible.

Two channels, and why
---------------------
The walk below answers two different questions, and an emitter has to say
which one its counts answer:

* **Measurement** (``scored_n``/``errored_n``) - attempts to measure the
  quality of a target's output. A metric case, a benchmark task, a red-team
  attack, a guardrails prompt. These are what ``error_rate`` is computed
  over: of the things we tried to measure, what fraction failed?
* **Load** (``load_scored_n``/``load_errored_n``) - requests put to a target
  to observe how it behaves under pressure. A stress test is the only
  emitter today. These are evidence that the target did work, so a
  load-only target does not trip the "measured nothing" rule, but they are
  not quality measurements and must stay out of ``error_rate``: a stress
  test's default 100 requests would otherwise outvote a ten-case metric
  batch, and a run with half its metrics erroring would report "completed".

A load emitter is not privileged. Its counts get their own error rate,
compared against the same ``max_error_rate``, so a stress test whose
requests failed still fails the run - on its own denominator instead of
everyone else's.

An emitter picks its channel(s) by which pair(s) of keys it emits. Nothing
here knows what a "stress test" is; it only knows what an emitter declared.
A node that declares both pairs feeds both channels - see ``_collect_counts``
for how a failure status is charged when that happens.
"""

from typing import Any, Dict, List, NamedTuple

from surogate_eval.statuses import FAILED_STATUSES

DEFAULT_MAX_ERROR_RATE = 0.2

#: Keys an emitter uses to declare quality-measurement counts.
MEASUREMENT_COUNT_KEYS = ('scored_n', 'errored_n')

#: Keys an emitter uses to declare load-generation counts.
LOAD_COUNT_KEYS = ('load_scored_n', 'load_errored_n')


class Counts(NamedTuple):
    """Countable units found in a results tree, split by channel."""

    scored: int = 0
    errored: int = 0
    load_scored: int = 0
    load_errored: int = 0

    def __add__(self, other: 'Counts') -> 'Counts':
        return Counts(*(a + b for a, b in zip(self, other)))

    @property
    def measured(self) -> int:
        """Units the error rate is computed over."""
        return self.scored + self.errored

    @property
    def load(self) -> int:
        """Units the load error rate is computed over."""
        return self.load_scored + self.load_errored

    @property
    def evidence(self) -> int:
        """Units of any kind: did anything at all happen here?"""
        return self.measured + self.load


def _rate(errored: int, total: int) -> float:
    return (errored / total) if total else 0.0


def _has_keys(node: Dict[str, Any], keys) -> bool:
    return all(key in node for key in keys)


def _collect_counts(node: Any) -> Counts:
    """Sum the countable units across an arbitrarily nested results tree.

    A summary dict - a metric batch, a benchmark, a red-team assessment, a
    guardrails result, a stress test - carries both its counts and the
    individual results that produced them, so when the summary keys are
    present we take those and do NOT descend, or every unit is counted
    twice.

    Any other dict carrying a failure status counts as one errored unit and
    is still descended into, because a failed node may contain partial
    results that also have to be counted.
    """
    if isinstance(node, dict):
        # A failure status is counted wherever it appears, including on a
        # node that also carries summary counts. Stated explicitly because
        # the reverse precedence - counts winning and the failure being
        # dropped - is silent, and silence is what this module exists to
        # prevent.
        failed = 1 if node.get('status') in FAILED_STATUSES else 0

        has_load = _has_keys(node, LOAD_COUNT_KEYS)
        has_measurement = _has_keys(node, MEASUREMENT_COUNT_KEYS)

        if has_load or has_measurement:
            # A summary node carries both its counts and the results that
            # produced them, so take the counts and do NOT descend, or every
            # unit is counted twice. Checked before the MetricResult rule
            # below because a metric batch carries a ``metric_name`` too.
            #
            # No emitter declares both channels today, but the two checks
            # used to be an if/elif, so a node that ever did would be read
            # as load-only and its measurement counts would vanish - the
            # exact silent drop this module exists to prevent. They are
            # independent now: a dual-channel node feeds both. Its failure
            # marker, if any, is charged to every channel the node declared
            # - a status doesn't say which channel failed, so the fail-
            # closed choice is to charge both rather than guess one.
            counts = Counts()
            if has_load:
                counts += Counts(
                    load_scored=int(node[LOAD_COUNT_KEYS[0]]),
                    load_errored=int(node[LOAD_COUNT_KEYS[1]]) + failed,
                )
            if has_measurement:
                counts += Counts(
                    scored=int(node[MEASUREMENT_COUNT_KEYS[0]]),
                    errored=int(node[MEASUREMENT_COUNT_KEYS[1]]) + failed,
                )
            return counts
        if 'metric_name' in node and 'status' in node:
            # No production path emits a bare MetricResult dict today: every
            # result is wrapped in a batch. Kept deliberately as a
            # fail-closed net so that if one ever does, it is counted rather
            # than silently ignored. Its own status is the whole signal, so
            # ``failed`` must not be added on top of it.
            if node['status'] == 'scored':
                return Counts(scored=1)
            return Counts(errored=1)

        counts = Counts(errored=failed)
        for value in node.values():
            counts += _collect_counts(value)
        return counts

    if isinstance(node, list):
        counts = Counts()
        for item in node:
            counts += _collect_counts(item)
        return counts

    return Counts()


def compute_outcome(
        consolidated: Dict[str, Any],
        max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> Dict[str, Any]:
    """Decide whether a finished run should be reported as failed."""
    targets = consolidated.get('targets') or []
    healthy = [t for t in targets if t.get('status') == 'success']

    totals = Counts()
    empty_targets: List[str] = []
    broken_targets: List[str] = []

    for entry in targets:
        target_counts = _collect_counts(entry)
        totals += target_counts
        name = entry.get('name') or 'unnamed'

        # A target that never ran is a coarse failure, not a rate. Diluted
        # into the run-wide error rate it disappears behind a busier target.
        if entry.get('status') in FAILED_STATUSES:
            broken_targets.append(name)
        # A target that passed its health check and then produced nothing at
        # all is not a success. "We measured nothing" used to divide to an
        # error rate of 0.0 and exit 0. Load counts answer this question -
        # a stress-only target did do work - even though they are excluded
        # from the error rate below.
        elif entry.get('status') == 'success' and target_counts.evidence == 0:
            empty_targets.append(name)

    error_rate = _rate(totals.errored, totals.measured)
    load_error_rate = _rate(totals.load_errored, totals.load)

    status = 'completed'
    reason = None

    if not healthy:
        status = 'failed'
        reason = 'No target completed its evaluations.'
    elif broken_targets:
        status = 'failed'
        reason = (
            f'Target(s) {", ".join(broken_targets)} did not complete their '
            'evaluations.'
        )
    elif empty_targets:
        status = 'failed'
        reason = (
            'No results were produced for target(s) '
            f'{", ".join(empty_targets)}; nothing was measured.'
        )
    elif error_rate > max_error_rate:
        status = 'failed'
        reason = (
            f'Error rate {error_rate:.1%} exceeds the maximum '
            f'{max_error_rate:.1%} ({totals.errored} of {totals.measured} '
            'evaluations errored).'
        )
    elif load_error_rate > max_error_rate:
        status = 'failed'
        reason = (
            f'Load error rate {load_error_rate:.1%} exceeds the maximum '
            f'{max_error_rate:.1%} ({totals.load_errored} of {totals.load} '
            'load requests failed).'
        )

    return {
        'status': status,
        'reason': reason,
        'scored': totals.scored,
        'errored': totals.errored,
        'error_rate': round(error_rate, 4),
        'load_scored': totals.load_scored,
        'load_errored': totals.load_errored,
        'load_error_rate': round(load_error_rate, 4),
        'max_error_rate': max_error_rate,
    }


def exit_code_for(outcome: Dict[str, Any]) -> int:
    """0 when the run is trustworthy, 1 otherwise."""
    return 0 if outcome.get('status') == 'completed' else 1
