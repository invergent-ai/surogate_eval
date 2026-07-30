from surogate_eval.metrics.base import BatchMetricResult, MetricResult, MetricType
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


def test_crashed_evaluation_fails_the_run():
    """runners.py records {"status": "failed"} with no counts at all."""
    consolidated = {
        "targets": [target(evaluations=[
            {"name": "e1", "status": "failed", "error": "boom"},
        ])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["errored"] == 1
    assert outcome["status"] == "failed"
    assert exit_code_for(outcome) == 1


def test_crashed_metric_fails_the_run():
    """A bare {"error": ..., "status": "failed"} node still fails the run.

    runners.py no longer emits this for a crashed metric - it emits a real
    batch with one errored result per unmeasured case - but the rule is kept
    as a fail-closed net for any node that carries only a failure status.
    """
    consolidated = {
        "targets": [target(evaluations=[{
            "name": "e1",
            "status": "completed",
            "metrics_summary": {"m": {"error": "boom", "status": "failed"}},
        }])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["errored"] == 1
    assert outcome["status"] == "failed"


def test_failed_benchmark_fails_the_run():
    consolidated = {
        "targets": [target(benchmarks=[
            {"benchmark_name": "b1", "status": "failed", "error": "boom"},
        ])]
    }
    assert compute_outcome(consolidated)["status"] == "failed"


def test_incompatible_benchmark_fails_the_run():
    consolidated = {
        "targets": [target(benchmarks=[
            {"benchmark": "b1", "status": "incompatible", "error": "wrong type"},
        ])]
    }
    assert compute_outcome(consolidated)["status"] == "failed"


def test_failed_red_teaming_fails_the_run():
    consolidated = {"targets": [target(red_teaming={"status": "failed", "error": "boom"})]}
    assert compute_outcome(consolidated)["status"] == "failed"


def test_failed_guardrails_fails_the_run():
    consolidated = {"targets": [target(guardrails={"status": "failed", "error": "boom"})]}
    assert compute_outcome(consolidated)["status"] == "failed"


def test_healthy_target_with_no_results_fails():
    """Zero evaluations configured: healthy, and nothing measured."""
    consolidated = {"targets": [target(evaluations=[])]}
    outcome = compute_outcome(consolidated)
    assert outcome["status"] == "failed"
    assert "nothing was measured" in outcome["reason"]
    assert exit_code_for(outcome) == 1


def test_one_empty_target_fails_even_when_another_is_clean():
    consolidated = {
        "targets": [
            {"name": "good", "status": "success", "evaluations": [batch(10, 0)]},
            {"name": "silent", "status": "success", "evaluations": []},
        ]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["status"] == "failed"
    assert "silent" in outcome["reason"]


def test_crashed_target_entry_fails_even_alongside_a_clean_one():
    """eval.py now records an entry when _run_target_evaluations raises."""
    consolidated = {
        "targets": [
            {"name": "good", "status": "success", "evaluations": [batch(10, 0)]},
            {"name": "crashed", "status": "failed", "error": "boom", "evaluations": []},
        ]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["errored"] == 1
    assert outcome["status"] == "failed"


def test_failed_status_does_not_hide_nested_counts():
    """A failed node still contributes whatever it did manage to measure."""
    consolidated = {
        "targets": [target(evaluations=[{
            "name": "e1",
            "status": "failed",
            "metrics_summary": {"m": batch(4, 0)},
        }])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["scored"] == 4
    assert outcome["errored"] == 1


def test_completed_statuses_are_not_counted_as_errors():
    """Only failure statuses count; 'completed' and 'success' are inert."""
    consolidated = {
        "targets": [target(evaluations=[{
            "name": "e1",
            "status": "completed",
            "metrics_summary": {"m": batch(10, 0)},
        }])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["scored"] == 10
    assert outcome["errored"] == 0
    assert outcome["status"] == "completed"
    assert exit_code_for(outcome) == 0


def test_summary_counts_do_not_swallow_a_failure_status():
    """No emitter carries both today. If one ever does, the failure must be
    counted on top of the counts rather than losing to them - a silently
    dropped failure is the exact bug this module exists to prevent.

    The counts here come from a real BatchMetricResult, not a hand-written
    shape; only the extra status is synthetic, because that is the
    hypothetical being pinned down.
    """
    real_batch = BatchMetricResult(
        metric_name="m",
        metric_type=MetricType.TOXICITY,
        results=[
            MetricResult(
                metric_name="m",
                metric_type=MetricType.TOXICITY,
                score=1.0,
                success=True,
            )
        ],
    ).to_dict()

    consolidated = {
        "targets": [target(evaluations=[{**real_batch, "status": "failed"}])]
    }
    outcome = compute_outcome(consolidated)
    assert outcome["scored"] == 1
    assert outcome["errored"] == 1
    assert outcome["status"] == "failed"


def test_failure_reason_is_populated():
    consolidated = {"targets": [target(evaluations=[batch(0, 10)])]}
    outcome = compute_outcome(consolidated)
    assert outcome["reason"]
    assert "10" in outcome["reason"]
