"""One bad judge response must cost one attack, not the whole scan.

``DeepEvalTargetWrapper`` raises instead of inventing a score - that is the
point of this branch. But the wrapper is also handed to DeepTeam as the
simulator and evaluation model, and DeepTeam's attack loop has no try/except
of its own: it absorbs a model error only when ``ignore_errors=True``. With
that off, the first unparseable judge response aborts every remaining attack
in the batch and propagates, and the runner flattens the wreckage into a
status-only failure node. The outcome walk counts such a node as exactly one
errored unit, so thirty lost attacks read as one error, pass the 20%
threshold, and the run reports "completed" and exits 0 - the silent success
this branch exists to eliminate.

DeepTeam's scan cannot be driven end-to-end offline: attack simulation wants
a live simulator model. So these tests drive DeepTeam's real per-attack loop
(``RedTeamer._a_evaluate_vulnerability_type``) with a real wrapper over a
fake target, and take its real ``RTTestCase`` output onwards through our real
conversion, ``to_dict`` and ``compute_outcome``. Nothing here is a
hand-written copy of a shape DeepTeam or this repo emits, and nothing stubs
the error-handling path being tested.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("deepteam")

from deepteam.red_teamer import RedTeamer  # noqa: E402
from deepteam.test_case import RTTestCase  # noqa: E402
from deepteam.vulnerabilities import PIILeakage  # noqa: E402

from surogate_eval.models import DeepEvalTargetWrapper  # noqa: E402
from surogate_eval.outcome import compute_outcome, exit_code_for  # noqa: E402
from surogate_eval.security.guardrails import (  # noqa: E402
    GuardrailsConfig,
    GuardrailsEvaluator,
)
from surogate_eval.security.red_team import RedTeamConfig, RedTeamRunner  # noqa: E402
from surogate_eval.targets.base import TargetResponse  # noqa: E402

#: Superset of the fields every schema the PII metric asks for needs
#: (``Purpose``, ``Entities``, ``ReasonScore``). Pydantic ignores the extras.
JUDGE_JSON = (
    '{"purpose": "answer questions", "entities": [], '
    '"reason": "the model declined", "score": 1}'
)

POISONED = "attack-1"


class FakeJudge:
    """A judge target. ``poison`` marks the one prompt it garbles."""

    name = "judge"
    config = {"base_url": "http://localhost:8000"}

    def __init__(self, poison=None):
        self.poison = poison
        self.calls = 0

    def send_request(self, request):
        self.calls += 1
        if self.poison and self.poison in (request.prompt or ""):
            # Prose where JSON was asked for: the wrapper cannot parse it
            # into the metric's schema and raises JudgeParseError.
            return TargetResponse(
                content="Looks fine to me.", raw_response={}, error=None
            )
        return TargetResponse(content=JUDGE_JSON, raw_response={}, error=None)


async def model_callback(input: str, turns: list = None) -> str:
    return "I cannot help with that."


async def failing_model_callback(input: str, turns: list = None) -> str:
    """The real callback's contract: a target failure raises.

    See ``RedTeamRunner.run``'s ``model_callback`` - there is no error slot
    in DeepTeam's callback signature, so raising is how a target failure is
    signalled.
    """
    if POISONED in (input or ""):
        raise RuntimeError("Target request failed: connection reset")
    return "I cannot help with that."


def deepteam_batch(attacks=4, poison=POISONED, ignore_errors=None, callback=None):
    """Run DeepTeam's real per-attack loop over a batch of attacks.

    Returns the ``RTTestCase`` list DeepTeam hands back, exactly as
    ``RedTeamRunner`` receives it from a real scan.
    """
    if ignore_errors is None:
        ignore_errors = RedTeamConfig().ignore_errors

    wrapper = DeepEvalTargetWrapper(FakeJudge(poison=poison))
    vulnerability = PIILeakage(types=["direct_disclosure"])
    vulnerability_type = vulnerability.types[0]
    red_teamer = RedTeamer(
        simulator_model=wrapper,
        evaluation_model=wrapper,
        async_mode=True,
        target_purpose="answer questions",
    )
    simulated = [
        RTTestCase(
            vulnerability="PII Leakage",
            vulnerability_type=vulnerability_type,
            input=f"attack-{idx}",
            attack_method="prompt_injection",
        )
        for idx in range(attacks)
    ]

    return asyncio.run(
        red_teamer._a_evaluate_vulnerability_type(
            model_callback=callback or model_callback,
            vulnerabilities=[vulnerability],
            vulnerability_type=vulnerability_type,
            simulated_test_cases=simulated,
            ignore_errors=ignore_errors,
        )
    )


def red_team_dict(test_cases):
    """The dict the runner records, built by the real conversion code."""
    runner = RedTeamRunner(
        SimpleNamespace(name="t1"), RedTeamConfig(vulnerabilities=["pii_leakage"])
    )
    assessment = runner._convert_risk_assessment(
        SimpleNamespace(overview="scan", test_cases=test_cases)
    )
    return assessment, assessment.to_dict()


def outcome_for(red_team_result):
    """The target entry eval.py builds for a red-team-only target."""
    return compute_outcome(
        {"targets": [{"name": "t1", "status": "success", "red_teaming": red_team_result}]}
    )


def test_one_bad_judge_response_costs_one_attack_not_the_whole_scan():
    cases = deepteam_batch(attacks=4)

    # DeepTeam kept going: three attacks were judged, one came back marked
    # with an error and no score.
    assert [case.score for case in cases] == [1.0, None, 1.0, 1.0]
    assert sum(1 for case in cases if case.error) == 1

    assessment, result = red_team_dict(cases)
    assert (result["scored_n"], result["errored_n"]) == (3, 1)

    outcome = outcome_for(result)
    assert (outcome["scored"], outcome["errored"]) == (3, 1)


def test_the_lost_attack_is_errored_rather_than_scored_zero():
    """An attack the judge never scored is not an attack the target survived."""
    _, result = red_team_dict(deepteam_batch(attacks=4))

    unjudged = [case for case in result["detailed_results"] if case["score"] is None]
    assert len(unjudged) == 1
    assert unjudged[0]["success"] is False


def test_error_isolation_off_aborts_every_remaining_attack():
    """Why the default matters, asserted against DeepTeam itself.

    With ``ignore_errors=False`` the first raise takes the whole batch with
    it: no attack in it is measured, and nothing is left for the outcome to
    count except the failure node the runner writes in its place - worth
    exactly one error however many attacks were lost.
    """
    from surogate_eval.errors import JudgeParseError

    with pytest.raises(JudgeParseError):
        deepteam_batch(attacks=4, ignore_errors=False)


def test_a_collapsed_scan_is_worth_one_error_however_many_attacks_it_lost():
    """The arithmetic that made the abort invisible.

    A scan that collapses carries no counts at all, so a busy run absorbs it
    below the threshold. Paired with the test above, this is the whole bug:
    the loss is real, the signal is one unit.
    """
    _, healthy = red_team_dict(deepteam_batch(attacks=4, poison=None))
    assert (healthy["scored_n"], healthy["errored_n"]) == (4, 0)

    collapsed = compute_outcome({
        "targets": [{
            "name": "t1",
            "status": "success",
            # A second target section measured fine; the red-team section
            # is the status-only node runners.py writes when the scan raises.
            "benchmarks": [healthy],
            "red_teaming": {"status": "failed", "error": "judge exploded"},
        }]
    })

    assert collapsed["errored"] == 1
    assert collapsed["error_rate"] <= collapsed["max_error_rate"]
    assert collapsed["status"] == "completed"
    assert exit_code_for(collapsed) == 0


def captured_deepteam_kwargs(monkeypatch):
    """Replace DeepTeam's entry point, keeping every layer below it real."""
    seen = {}

    def fake_red_team(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(overview="", test_cases=[])

    monkeypatch.setattr("deepteam.red_team", fake_red_team)
    return seen


class FakeTarget:
    """Refuses everything, in a shape the refusal judge can read."""

    name = "t1"
    config = {"base_url": "http://localhost:8000"}

    def send_request(self, request):
        return TargetResponse(content="NO", raw_response={}, error=None)

    async def send_request_async(self, request):
        return self.send_request(request)


def test_red_team_runner_asks_deepteam_to_isolate_errors(monkeypatch):
    from surogate_eval.runners import run_red_teaming_async

    seen = captured_deepteam_kwargs(monkeypatch)

    asyncio.run(
        run_red_teaming_async(
            FakeTarget(),
            {"vulnerabilities": ["pii_leakage"], "attacks": ["prompt_injection"]},
            lambda name: None,
        )
    )

    assert seen["ignore_errors"] is True


def test_guardrails_asks_deepteam_to_isolate_errors(monkeypatch):
    from surogate_eval.runners import run_guardrails_testing_async

    seen = captured_deepteam_kwargs(monkeypatch)

    asyncio.run(
        run_guardrails_testing_async(
            FakeTarget(),
            {"vulnerabilities": ["pii_leakage"], "attacks": ["prompt_injection"]},
            lambda name: None,
        )
    )

    assert seen["ignore_errors"] is True


class RaisingJudgeTarget:
    """A refusal judge whose client dies on one prompt."""

    name = "refusal-judge"
    config = {}

    def __init__(self, poison):
        self.poison = poison

    def send_request(self, request):
        if self.poison in request.prompt:
            raise RuntimeError("judge client exploded")
        return TargetResponse(content="YES", raw_response={}, error=None)


def guardrails_evaluator(monkeypatch, judge_target, scan):
    """A real evaluator whose scan hands back real DeepTeam test cases."""
    async def fake_run(self):
        if isinstance(scan, Exception):
            raise scan
        return scan

    monkeypatch.setattr(RedTeamRunner, "run", fake_run)
    return GuardrailsEvaluator(
        FakeTarget(),
        GuardrailsConfig(vulnerabilities=["pii_leakage"], attacks=["prompt_injection"]),
        judge_target,
    )


def test_a_judge_failure_on_one_harmful_prompt_errors_only_that_prompt(monkeypatch):
    """The guardrails half of the same rule.

    The refusal judge dies on one of four prompts. That prompt was never
    tested and is counted as one error; the other three are still put to the
    model and judged. The counter used to sit behind the ``raise``, so this
    was either a lost count or a lost evaluation, never a counted one.
    """
    assessment, _ = red_team_dict(deepteam_batch(attacks=4, poison=None))
    evaluator = guardrails_evaluator(
        monkeypatch, RaisingJudgeTarget(poison="attack-1"), assessment
    )

    result = asyncio.run(evaluator.evaluate())

    assert result.harmful_prompts_tested == 3
    assert result.harmful_prompts_errored == 1
    assert result.result_counts() == (3, 1)
    assert result.to_dict()["harmful_prompts"]["errored"] == 1


class ReadingRefusalJudge:
    """A refusal judge that can only rule on a response it can read.

    Real judges behave this way: handed "AI Response: None" there is nothing
    to assess, so the answer comes back without a YES or a NO in it and the
    evaluator's unclear-answer fallback (``return not is_harmful``) decides
    for it - which for a harmful prompt means "the target complied".
    """

    name = "refusal-judge"
    config = {}

    def send_request(self, request):
        if "AI Response: None" in request.prompt:
            return TargetResponse(
                content="There is no response here to assess.",
                raw_response={},
                error=None,
            )
        return TargetResponse(content="YES", raw_response={}, error=None)


def test_a_target_failure_is_not_a_harmful_prompt_the_model_answered(monkeypatch):
    """Guardrails half of the callback fix that red-team already has.

    The callback raises on one of four attacks, so DeepTeam marks that case
    and leaves it without an ``actual_output``. Guardrails read the attribute
    regardless, judged "AI Response: None", and recorded a fabricated
    "complied with a harmful attack" verdict - one prompt counted as tested
    that was never put to the model, and a refusal rate dragged down by an
    unreachable target.
    """
    cases = deepteam_batch(attacks=4, poison=None, callback=failing_model_callback)

    # DeepTeam absorbed the target failure into one case and left it unscored.
    assert sum(1 for case in cases if case.error is not None) == 1
    assert sum(1 for case in cases if case.actual_output is None) == 1

    assessment, _ = red_team_dict(cases)
    evaluator = guardrails_evaluator(monkeypatch, ReadingRefusalJudge(), assessment)

    result = asyncio.run(evaluator.evaluate())

    assert result.harmful_prompts_tested == 3
    assert result.harmful_prompts_errored == 1
    assert result.result_counts() == (3, 1)
    # Every prompt the model actually answered was a refusal. The
    # unreachable one must not read as a guardrails failure.
    assert result.refusal_rate == 1.0


def test_a_failed_scan_is_counted_instead_of_collapsing_the_evaluation(monkeypatch):
    """``runner.run()`` used to be unguarded, so a scan that raised took the
    whole guardrails result with it and left the per-prompt counters
    unreached."""
    evaluator = guardrails_evaluator(
        monkeypatch,
        RaisingJudgeTarget(poison="never matches"),
        RuntimeError("deepteam exploded"),
    )

    result = asyncio.run(evaluator.evaluate())

    assert result.result_counts() == (0, 1)
    assert exit_code_for(outcome_for(result.to_dict())) == 1
