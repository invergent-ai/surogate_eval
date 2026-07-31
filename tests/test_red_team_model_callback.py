"""A red-team target failure must not read as a clean refusal.

``model_callback`` used to catch a target failure and return "". DeepTeam's
refusal judge then scored that empty string as if the target had answered
and declined the attack - a false PASS on an unreachable target, the worst
direction a security scan can fail in.

DeepTeam's scan cannot be driven end-to-end offline (attack simulation wants
a live simulator model), so - like test_deepteam_error_isolation.py - these
tests fake only DeepTeam's entry point (``deepteam.red_team``) to capture
the real callback ``RedTeamRunner.run()`` builds, then drive DeepTeam's real
per-attack loop (``RedTeamer._a_evaluate_vulnerability_type``) and our real
conversion code with it. Nothing here is a hand-written copy of a shape
DeepTeam or this repo emits.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("deepteam")

from deepteam.red_teamer import RedTeamer  # noqa: E402
from deepteam.test_case import RTTestCase  # noqa: E402
from deepteam.vulnerabilities import PIILeakage  # noqa: E402

from surogate_eval.models import DeepEvalTargetWrapper  # noqa: E402
from surogate_eval.security.red_team import RedTeamConfig, RedTeamRunner  # noqa: E402
from surogate_eval.targets.base import TargetResponse  # noqa: E402

#: Superset of the fields every schema the PII metric asks for needs.
JUDGE_JSON = (
    '{"purpose": "answer questions", "entities": [], '
    '"reason": "the model declined", "score": 1}'
)


class RefusalJudge:
    """A judge target that always reports the attack as refused."""

    name = "judge"
    config = {"base_url": "http://localhost:8000"}

    def send_request(self, request):
        return TargetResponse(content=JUDGE_JSON, raw_response={}, error=None)


class UnreachableTarget:
    """A target whose async request comes back carrying an error, not a
    raise - the same shape a real HTTP-layer failure takes."""

    name = "t1"
    config = {"base_url": "http://localhost:9"}

    async def send_request_async(self, request):
        return TargetResponse(content=None, raw_response={}, error="connection reset")


def capture_model_callback(monkeypatch):
    """Grab the real callback RedTeamRunner.run() builds and hands to
    DeepTeam, without running DeepTeam's actual scan (which needs a live
    simulator model)."""
    seen = {}

    def fake_red_team(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(overview="", test_cases=[])

    monkeypatch.setattr("deepteam.red_team", fake_red_team)
    return seen


def test_model_callback_raises_instead_of_returning_empty_string(monkeypatch):
    """The swallow this fix removes: awaiting the callback used to return ""
    for an unreachable target, never raising."""
    seen = capture_model_callback(monkeypatch)
    runner = RedTeamRunner(
        UnreachableTarget(),
        RedTeamConfig(vulnerabilities=["pii_leakage"], attacks=["prompt_injection"]),
    )

    asyncio.run(runner.run())
    model_callback = seen["model_callback"]

    with pytest.raises(Exception, match="connection reset"):
        asyncio.run(model_callback("attack prompt"))


class FakeMetric:
    """Stands in for DeepTeam's real per-vulnerability metric so the test
    isolates the callback's behaviour from the metric's own judgment. If it
    is ever asked to score empty output as a clean refusal, that is the
    false PASS this fix exists to prevent - so it is configured to do
    exactly that, and the assertion is that it must never be asked."""

    def __init__(self, score_for_empty_output=1.0):
        self.score = None
        self.reason = None
        self._score_for_empty_output = score_for_empty_output
        self.measured = False

    async def a_measure(self, test_case):
        self.measured = True
        self.score = self._score_for_empty_output if test_case.actual_output == "" else 0.0
        self.reason = "measured"


def attack_with_fake_metric(model_callback, fake_metric, monkeypatch):
    """Drive DeepTeam's real single-turn attack step (RedTeamer._a_attack)
    with the real callback and a controlled metric standing in for the
    judge, so the only variable is what the callback does on failure."""
    vulnerability = PIILeakage(types=["direct_disclosure"])
    vulnerability_type = vulnerability.types[0]
    monkeypatch.setattr(vulnerability, "_get_metric", lambda t: fake_metric)

    wrapper = DeepEvalTargetWrapper(RefusalJudge())
    red_teamer = RedTeamer(
        simulator_model=wrapper,
        evaluation_model=wrapper,
        async_mode=True,
        target_purpose="answer questions",
    )
    simulated_test_case = RTTestCase(
        vulnerability="PII Leakage",
        vulnerability_type=vulnerability_type,
        input="attack prompt",
        attack_method="prompt_injection",
    )

    return asyncio.run(
        red_teamer._a_attack(
            model_callback=model_callback,
            simulated_test_case=simulated_test_case,
            vulnerability="PII Leakage",
            vulnerability_type=vulnerability_type,
            vulnerabilities=[vulnerability],
            ignore_errors=RedTeamConfig().ignore_errors,
        )
    )


def test_target_failure_is_errored_not_a_passed_refusal(monkeypatch):
    """The callback's raise is caught by DeepTeam's own attack loop
    (RedTeamer._a_attack, ignore_errors=True) before the metric ever runs:
    the test case comes back with .error set and .score left None -
    unjudged - instead of the metric scoring an empty answer as a clean
    PASS. That is exactly the channel RiskAssessment.result_counts()
    already treats as errored.
    """
    seen = capture_model_callback(monkeypatch)
    runner = RedTeamRunner(
        UnreachableTarget(),
        RedTeamConfig(vulnerabilities=["pii_leakage"], attacks=["prompt_injection"]),
    )
    asyncio.run(runner.run())
    model_callback = seen["model_callback"]

    fake_metric = FakeMetric(score_for_empty_output=1.0)
    case = attack_with_fake_metric(model_callback, fake_metric, monkeypatch)

    # The metric was never consulted: a target failure must not be judged
    # as though it were a legitimate (empty) answer.
    assert fake_metric.measured is False
    assert case.score is None
    assert case.error is not None

    conversion_runner = RedTeamRunner(
        SimpleNamespace(name="t1"), RedTeamConfig(vulnerabilities=["pii_leakage"])
    )
    assessment = conversion_runner._convert_risk_assessment(
        SimpleNamespace(overview="scan", test_cases=[case])
    )
    result = assessment.to_dict()

    assert (result["scored_n"], result["errored_n"]) == (0, 1)
