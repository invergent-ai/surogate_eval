"""A stress test that ran is evidence that something was measured.

``StressTestResult.to_dict()`` used to emit no counts at all, so a
stress-only target contributed 0/0 to the outcome walk, tripped the "healthy
target produced zero countable results" rule and exited 1 after a perfectly
good run.

The fixtures here are produced by the real ``StressTester`` against a fake
target - no socket, no hand-written copy of a shape this repo emits.
"""

from surogate_eval.datasets.test_case import TestCase
from surogate_eval.metrics.stress.stress_tester import (
    StressTestConfig,
    StressTester,
    StressTestResult,
)
from surogate_eval.outcome import compute_outcome, exit_code_for
from surogate_eval.targets.base import TargetResponse


class FakeTarget:
    """Answers every request. ``fail_after`` requests start erroring."""

    def __init__(self, fail_after=None):
        self.name = "stress-target"
        self.fail_after = fail_after
        self.seen = 0

    def send_request(self, request):
        self.seen += 1
        if self.fail_after is not None and self.seen > self.fail_after:
            return TargetResponse(content="", raw_response={}, error="connection reset")
        return TargetResponse(content="a fine answer", raw_response={})


def stress_dict(fail_after=None, num_requests=6):
    """Drive the real stress tester and return its real ``to_dict()``."""
    tester = StressTester(
        FakeTarget(fail_after=fail_after),
        [TestCase(input="hello"), TestCase(input="hi there")],
    )
    config = StressTestConfig(
        num_concurrent=2,
        num_requests=num_requests,
        monitor_resources=False,
        warmup_requests=0,
        max_failures=num_requests + 1,
    )
    return tester.run(config).to_dict()


def outcome_for(stress_result):
    """The target entry eval.py builds for a stress-only target."""
    consolidated = {
        "targets": [
            {"name": "t1", "status": "success", "stress_testing": stress_result}
        ]
    }
    return compute_outcome(consolidated)


def test_successful_stress_only_run_exits_zero():
    result = stress_dict()

    assert result["scored_n"] == result["total_requests"]
    assert result["errored_n"] == 0

    outcome = outcome_for(result)
    assert outcome["status"] == "completed"
    assert outcome["scored"] == result["total_requests"]
    assert exit_code_for(outcome) == 0


def test_failing_requests_are_counted_as_errored():
    """A request that never came back produced no latency sample."""
    result = stress_dict(fail_after=1)

    assert result["errored_n"] == result["failed_requests"]
    assert result["errored_n"] > 0

    outcome = outcome_for(result)
    assert outcome["status"] == "failed"
    assert exit_code_for(outcome) == 1


def test_stress_test_that_sent_nothing_fails_the_run():
    """Built rather than driven: a stress-only target that sent nothing
    still needs to fail the run rather than dissolve into a zero error
    rate. The result object and its ``to_dict()`` are still the real
    ones."""
    result = StressTestResult(
        config=StressTestConfig(num_requests=0, monitor_resources=False),
        total_duration=0.0,
        total_requests=0,
        successful_requests=0,
        failed_requests=0,
        requests_per_second=0.0,
        avg_latency_ms=0.0,
        median_latency_ms=0.0,
        p95_latency_ms=0.0,
        p99_latency_ms=0.0,
        min_latency_ms=0.0,
        max_latency_ms=0.0,
    ).to_dict()

    assert (result["scored_n"], result["errored_n"]) == (0, 1)
    assert exit_code_for(outcome_for(result)) == 1


def test_num_requests_zero_terminates_and_sends_nothing():
    """``num_requests=0`` used to be falsy, so the "stop now" guards in
    ``_execute_concurrent_requests`` never fired and the run spun in its
    ``while True`` loop forever, submitting nothing. Drive the real tester
    with a real (tiny) config and confirm it now returns immediately with
    no requests sent, instead of hanging.

    Guarded with a hard wall-clock timeout so a regression fails fast
    instead of hanging the suite.
    """
    import threading

    tester = StressTester(
        FakeTarget(),
        [TestCase(input="hello")],
    )
    config = StressTestConfig(
        num_concurrent=2,
        num_requests=0,
        monitor_resources=False,
        warmup_requests=0,
    )

    outcome = {}

    def run():
        outcome["result"] = tester.run(config).to_dict()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=5.0)

    assert not thread.is_alive(), "stress run with num_requests=0 did not terminate"
    result = outcome["result"]
    assert result["total_requests"] == 0
    assert result["scored_n"] == 0
    assert result["errored_n"] == 1
