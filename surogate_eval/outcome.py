"""Run-level outcome: did this evaluation produce trustworthy results?

The runner used to end every run with "completed" and exit 0, even when no
target was reachable and every metric had errored. Ops reads the exit code
to decide whether a run failed, so the outcome computed here is what makes
a broken run visible.
"""

from typing import Any, Dict, List, Tuple

DEFAULT_MAX_ERROR_RATE = 0.2

#: Statuses a node can carry when it represents a failure to measure rather
#: than a measurement. Failures above the metric level (a whole evaluation
#: crashing, a benchmark blowing up, a target that never ran) produce no
#: ``scored_n``/``errored_n`` counts of their own, so without this set they
#: were invisible: ``total`` stayed 0, ``error_rate`` stayed 0.0, and the run
#: reported "completed".
FAILED_STATUSES = frozenset({
    'failed',
    'error',
    'validation_failed',
    'incompatible',
    'unhealthy',
})


def _collect_counts(node: Any) -> Tuple[int, int]:
    """Sum (scored, errored) across an arbitrarily nested results tree.

    A ``BatchMetricResult`` dict carries both summary counts and the
    individual results that produced them, so when the summary keys are
    present we take those and do NOT descend, or every case is counted
    twice.

    Any other dict carrying a failure status counts as one errored unit and
    is still descended into, because a failed node may contain partial
    results that also have to be counted.
    """
    if isinstance(node, dict):
        if 'scored_n' in node and 'errored_n' in node:
            return int(node['scored_n']), int(node['errored_n'])
        if 'metric_name' in node and 'status' in node:
            # No production path emits a bare MetricResult dict today: every
            # result is wrapped in a batch. Kept deliberately as a
            # fail-closed net so that if one ever does, it is counted rather
            # than silently ignored - which is the exact failure mode this
            # module exists to prevent.
            return (1, 0) if node['status'] == 'scored' else (0, 1)

        scored = 0
        errored = 1 if node.get('status') in FAILED_STATUSES else 0
        for value in node.values():
            s, e = _collect_counts(value)
            scored += s
            errored += e
        return scored, errored

    if isinstance(node, list):
        scored = errored = 0
        for item in node:
            s, e = _collect_counts(item)
            scored += s
            errored += e
        return scored, errored

    return 0, 0


def compute_outcome(
        consolidated: Dict[str, Any],
        max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> Dict[str, Any]:
    """Decide whether a finished run should be reported as failed."""
    targets = consolidated.get('targets') or []
    healthy = [t for t in targets if t.get('status') == 'success']

    scored = errored = 0
    empty_targets: List[str] = []
    broken_targets: List[str] = []

    for entry in targets:
        target_scored, target_errored = _collect_counts(entry)
        scored += target_scored
        errored += target_errored
        name = entry.get('name') or 'unnamed'

        # A target that never ran is a coarse failure, not a rate. Diluted
        # into the run-wide error rate it disappears behind a busier target.
        if entry.get('status') in FAILED_STATUSES:
            broken_targets.append(name)
        # A target that passed its health check and then measured nothing at
        # all is not a success. "We measured nothing" used to divide to an
        # error rate of 0.0 and exit 0.
        elif entry.get('status') == 'success' and target_scored + target_errored == 0:
            empty_targets.append(name)

    total = scored + errored
    error_rate = (errored / total) if total else 0.0

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
            f'{max_error_rate:.1%} ({errored} of {total} evaluations errored).'
        )

    return {
        'status': status,
        'reason': reason,
        'scored': scored,
        'errored': errored,
        'error_rate': round(error_rate, 4),
        'max_error_rate': max_error_rate,
    }


def exit_code_for(outcome: Dict[str, Any]) -> int:
    """0 when the run is trustworthy, 1 otherwise."""
    return 0 if outcome.get('status') == 'completed' else 1
