"""A stress test is evidence that a target did work - and nothing more.

``StressTestResult.to_dict()`` used to emit no counts at all, so a
stress-only target contributed nothing to the outcome walk, tripped the
"healthy target produced zero countable results" rule and exited 1 after a
perfectly good run. Then it emitted its requests as ``scored_n``/
``errored_n``, which fixed that and broke something worse: a default
100-request stress test outvoted every metric in the run, so a target with
half its metric cases erroring came out at a 4.5% error rate and exited 0.

So the requests are load counts now, on their own channel with their own
error rate: enough to prove the target did work, unable to move the
quality-measurement error rate, still able to fail the run on their own.

The fixtures here are produced by the real ``StressTester`` against a fake
target - no socket, no hand-written copy of a shape this repo emits.
"""

from surogate_eval.datasets.test_case import TestCase
from surogate_eval.metrics.base import (
    BatchMetricResult,
    MetricResult,
    MetricType,
)
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

    assert result["load_scored_n"] == result["total_requests"]
    assert result["load_errored_n"] == 0
    # The load channel, and only the load channel.
    assert "scored_n" not in result and "errored_n" not in result

    outcome = outcome_for(result)
    assert outcome["status"] == "completed"
    assert outcome["load_scored"] == result["total_requests"]
    assert outcome["scored"] == 0
    assert exit_code_for(outcome) == 0


def test_failing_requests_are_counted_as_errored():
    """A request that never came back produced no latency sample.

    The stress test keeps its own pass/fail signal: these failures never
    touch the measurement error rate, but they exceed the threshold on the
    load channel and fail the run.
    """
    result = stress_dict(fail_after=1)

    assert result["load_errored_n"] == result["failed_requests"]
    assert result["load_errored_n"] > 0

    outcome = outcome_for(result)
    assert outcome["errored"] == 0
    assert outcome["error_rate"] == 0.0
    assert outcome["load_error_rate"] > outcome["max_error_rate"]
    assert outcome["status"] == "failed"
    assert exit_code_for(outcome) == 1


def test_passing_stress_cannot_hide_metric_errors():
    """The regression this split exists to prevent.

    Five of ten metric cases errored: a 50% error rate, a failed run. Add a
    perfectly healthy stress test at the runner's default 100 requests and
    the run used to come out at 4.5% and exit 0 - a passing stress test
    turning a failing run green.

    Both halves are real objects put through their real ``to_dict()``.
    """
    batch = BatchMetricResult(
        metric_name="toxicity",
        metric_type=MetricType.TOXICITY,
        results=[
            MetricResult(
                metric_name="toxicity",
                metric_type=MetricType.TOXICITY,
                score=0.9,
                success=True,
            )
            for _ in range(5)
        ] + [
            MetricResult.errored(
                metric_name="toxicity",
                metric_type=MetricType.TOXICITY,
                reason="judge unavailable",
            )
            for _ in range(5)
        ],
    ).to_dict()

    stress = stress_dict(num_requests=100)
    # Asserted off the raw request tallies, not the count keys, so that this
    # test fails on the dilution itself rather than on a missing key if the
    # requests ever go back into the measurement channel.
    assert (stress["total_requests"], stress["failed_requests"]) == (100, 0)

    consolidated = {
        "targets": [{
            "name": "t1",
            "status": "success",
            "evaluations": [batch],
            "stress_testing": stress,
        }]
    }
    outcome = compute_outcome(consolidated)

    assert (outcome["scored"], outcome["errored"]) == (5, 5)
    assert outcome["error_rate"] == 0.5
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

    assert (result["load_scored_n"], result["load_errored_n"]) == (0, 1)
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
    assert result["load_scored_n"] == 0
    assert result["load_errored_n"] == 1


def stress_failure(dataset=None):
    """Drive the real runner down a path that never reaches ``to_dict()``.

    A missing ``dataset`` key takes the config guard; a path that is not
    there takes the catch-all, via the real loader's FileNotFoundError.
    Both are reachable from a working config: ``_validate_file_paths`` only
    warns about a stress dataset it cannot find, so a typo survives
    validation and lands here.
    """
    from surogate_eval.runners import run_stress_testing

    config = {} if dataset is None else {"dataset": str(dataset)}
    return run_stress_testing(FakeTarget(), config)


def half_measured_batch():
    """Sixteen metric cases scored, four errored: 20.0%, exactly at the
    default threshold and therefore still a passing run."""
    return BatchMetricResult(
        metric_name="toxicity",
        metric_type=MetricType.TOXICITY,
        results=[
            MetricResult(
                metric_name="toxicity",
                metric_type=MetricType.TOXICITY,
                score=0.9,
                success=True,
            )
            for _ in range(16)
        ] + [
            MetricResult.errored(
                metric_name="toxicity",
                metric_type=MetricType.TOXICITY,
                reason="judge unavailable",
            )
            for _ in range(4)
        ],
    ).to_dict()


def test_a_stress_crash_declares_the_load_channel(tmp_path):
    """Both early returns skip ``StressTestResult.to_dict()``, so they have
    to name their own channel or the outcome walk guesses - and its guess
    is the measurement channel."""
    for crash in (stress_failure(), stress_failure(tmp_path / "typo.csv")):
        assert (crash["load_scored_n"], crash["load_errored_n"]) == (0, 0)
        assert "scored_n" not in crash and "errored_n" not in crash


def test_a_stress_crash_cannot_move_the_measurement_error_rate(tmp_path):
    """The verdict flip. Sixteen scored and four errored metric cases sit at
    20.0%, exactly at the threshold and passing. A stress test that died
    before its first request used to be charged to the measurement channel
    as a twenty-first evaluation, taking the run to 5 of 21 - 23.8% - and
    failing it for a reason that had nothing to do with the metrics.

    The crash still fails the run. It fails it on the load channel, where a
    load failure belongs.
    """
    batch = half_measured_batch()

    without_stress = compute_outcome({
        "targets": [{"name": "t1", "status": "success", "evaluations": [batch]}]
    })
    assert (without_stress["scored"], without_stress["errored"]) == (16, 4)
    assert without_stress["error_rate"] == 0.2
    assert without_stress["status"] == "completed"

    with_stress = compute_outcome({
        "targets": [{
            "name": "t1",
            "status": "success",
            "evaluations": [batch],
            "stress_testing": stress_failure(tmp_path / "typo.csv"),
        }]
    })

    assert (with_stress["scored"], with_stress["errored"]) == (16, 4)
    assert with_stress["error_rate"] == without_stress["error_rate"]
    assert (with_stress["load_scored"], with_stress["load_errored"]) == (0, 1)
    assert with_stress["load_error_rate"] == 1.0
    assert with_stress["status"] == "failed"
    assert "load" in with_stress["reason"].lower()
    assert exit_code_for(with_stress) == 1
