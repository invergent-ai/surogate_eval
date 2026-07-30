"""Run-level outcome: did this evaluation produce trustworthy results?

The runner used to end every run with "completed" and exit 0, even when no
target was reachable and every metric had errored. Ops reads the exit code
to decide whether a run failed, so the outcome computed here is what makes
a broken run visible.
"""

from typing import Any, Dict, Tuple

DEFAULT_MAX_ERROR_RATE = 0.2


def _collect_counts(node: Any) -> Tuple[int, int]:
    """Sum (scored, errored) across an arbitrarily nested results tree.

    A ``BatchMetricResult`` dict carries both summary counts and the
    individual results that produced them, so when the summary keys are
    present we take those and do NOT descend, or every case is counted
    twice.
    """
    if isinstance(node, dict):
        if 'scored_n' in node and 'errored_n' in node:
            return int(node['scored_n']), int(node['errored_n'])
        if 'metric_name' in node and 'status' in node:
            return (1, 0) if node['status'] == 'scored' else (0, 1)
        scored = errored = 0
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

    scored, errored = _collect_counts(targets)
    total = scored + errored
    error_rate = (errored / total) if total else 0.0

    status = 'completed'
    reason = None

    if not healthy:
        status = 'failed'
        reason = 'No target completed its evaluations.'
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
