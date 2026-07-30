from surogate_eval.outcome import (
    DEFAULT_MAX_ERROR_RATE,
    compute_outcome,
    exit_code_for,
)


def batch(scored_n, errored_n):
    """Shape emitted by BatchMetricResult.to_dict()."""
    return {"scored_n": scored_n, "errored_n": errored_n, "results": []}


def target(status="success", **extra):
    return {"name": "t1", "status": status, **extra}


def test_default_threshold_is_twenty_percent():
    assert DEFAULT_MAX_ERROR_RATE == 0.2


def test_clean_run_completes():
    consolidated = {"targets": [target(evaluations=[batch(10, 0)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["status"] == "completed"
    assert exit_code_for(outcome) == 0


def test_no_healthy_target_fails_regardless_of_threshold():
    consolidated = {"targets": [target(status="unhealthy")]}
    outcome = compute_outcome(consolidated)
    assert outcome["status"] == "failed"
    assert exit_code_for(outcome) == 1


def test_empty_targets_fails():
    outcome = compute_outcome({"targets": []})
    assert outcome["status"] == "failed"


def test_error_rate_over_threshold_fails():
    consolidated = {"targets": [target(evaluations=[batch(5, 5)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["error_rate"] == 0.5
    assert outcome["status"] == "failed"
    assert exit_code_for(outcome) == 1


def test_error_rate_under_threshold_completes():
    consolidated = {"targets": [target(evaluations=[batch(90, 10)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["error_rate"] == 0.1
    assert outcome["status"] == "completed"


def test_threshold_is_configurable():
    consolidated = {"targets": [target(evaluations=[batch(90, 10)])]}
    outcome = compute_outcome(consolidated, max_error_rate=0.05)
    assert outcome["status"] == "failed"


def test_counts_are_not_double_counted():
    """A batch dict carries both summary counts and a results list."""
    nested = {
        "scored_n": 1,
        "errored_n": 1,
        "results": [
            {"metric_name": "m", "status": "scored"},
            {"metric_name": "m", "status": "errored"},
        ],
    }
    outcome = compute_outcome({"targets": [target(evaluations=[nested])]})
    assert outcome["scored"] == 1
    assert outcome["errored"] == 1


def test_bare_metric_results_are_counted():
    """Paths that emit MetricResult dicts without a batch wrapper."""
    consolidated = {
        "targets": [target(evaluations=[
            {"metric_name": "m", "status": "errored"},
            {"metric_name": "m", "status": "scored"},
        ])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["scored"] == 1
    assert outcome["errored"] == 1


def test_deeply_nested_counts_are_found():
    consolidated = {
        "targets": [target(benchmarks=[{"suite": {"metrics": [batch(3, 1)]}}])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["scored"] == 3
    assert outcome["errored"] == 1


def test_failure_reason_is_populated():
    consolidated = {"targets": [target(evaluations=[batch(0, 10)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["reason"]
    assert "10" in outcome["reason"]
