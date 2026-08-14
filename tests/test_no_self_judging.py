"""A target may not judge itself (E-RUN-7).

Red-team and guardrails scans grade the model under test on whether it
refused, leaked, or was jailbroken. When no judge resolves, both paths hand
the *target itself* the judging role, with an info log. A model that has just
been jailbroken is then asked whether it was jailbroken, and a scan that
graded nothing reads as a clean bill of health.

The two ways in are different and both are covered here:

- **Named.** Ops emits ``evaluation_model: {target: <the target's own
  name>}`` whenever no judge is configured in Studio (`build_eval_config`),
  so from Studio this is not a silent fallback at all -- the config says it
  out loud, and the runner obliges. This is the reachable one.
- **Omitted.** A hand-written YAML that simply leaves the judge out gets the
  same result from the runtime fallbacks.

Rejected at config load where possible, because that fails in seconds
instead of after a pod has provisioned and a scan has run. The runtime
guards stay as well, so no silent self-judging path is left behind.

The simulator is deliberately NOT covered by this rule: generating your own
adversarial prompts is a weaker scan, not a fabricated verdict.
"""

import asyncio
from types import SimpleNamespace

import pytest

from surogate_eval import runners
from surogate_eval.config.eval_config import EvalConfig
from surogate_eval.config.loader import load_config
from surogate_eval.errors import ConfigError

TARGET = """\
  - name: {name}
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-literal
"""


def config_text(*, security: str, extra_target: bool = False) -> str:
    text = "project:\n  name: test\ntargets:\n" + TARGET.format(name="t1")
    text += security
    if extra_target:
        text += TARGET.format(name="judge")
    return text


def write(tmp_path, text):
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def load(tmp_path, text):
    return load_config(EvalConfig, write(tmp_path, text))


# --- Config load: a named self-judge is rejected ---------------------------


def test_a_red_team_scan_may_not_name_its_own_target_as_evaluator(tmp_path):
    """The shape ops emits when Studio configured no judge.

    `ValueError`, not `ConfigError`: `EvalConfig.__post_init__` reports every
    validation failure that way, while `load_config`'s own env-var check
    raises `ConfigError`. That split is pre-existing and not this change's to
    settle.
    """
    text = config_text(security="""\
    red_teaming:
      enabled: true
      evaluation_model:
        target: t1
""")
    with pytest.raises(ValueError) as exc:
        load(tmp_path, text)

    message = str(exc.value)
    assert "evaluation_model" in message
    assert "t1" in message


def test_guardrails_may_not_name_its_own_target_as_refusal_judge(tmp_path):
    text = config_text(security="""\
    guardrails:
      enabled: true
      evaluation_model:
        target: judge
      refusal_judge_model:
        target: t1
""", extra_target=True)
    with pytest.raises(ValueError) as exc:
        load(tmp_path, text)

    assert "refusal_judge_model" in str(exc.value)


def test_a_benchmark_judge_naming_its_own_target_warns_but_loads(tmp_path):
    """The deliberate half of the same shape, and the line where this stops.

    A `judge_model` pointing at its own target is something a user wrote:
    `examples/config.yaml` ships it, self-evaluation is a real if weak
    technique, and refusing it would break configs that never asked for a
    second model. It still says so, because the resulting score answers a
    different question than the user probably thinks.

    The security sections are an error rather than a warning for the opposite
    reason: nobody writes those by hand. Ops fills them in with the target's
    own name whenever Studio configured no judge.
    """
    text = config_text(security="""\
    evaluations:
      - name: e1
        benchmarks:
          - name: mt_bench
            judge_model:
              target: t1
""")
    config = load(tmp_path, text)  # loads, rather than raising

    # Asserted off the validator rather than the log: this logger neither
    # propagates to the root nor writes to the stream pytest replaces, so
    # both caplog and capfd come back empty while the warning is on screen.
    # `__post_init__` logs every warning this returns.
    _, warnings = config._validate_target_references()
    assert any("grades its own answers" in w for w in warnings), warnings


# --- Config load: an enabled scan must have a judge at all -----------------


def test_an_enabled_red_team_scan_needs_an_evaluator(tmp_path):
    text = config_text(security="""\
    red_teaming:
      enabled: true
      vulnerabilities: [toxicity]
""")
    with pytest.raises(ValueError) as exc:
        load(tmp_path, text)

    assert "evaluation_model" in str(exc.value)


def test_enabled_guardrails_need_a_refusal_judge(tmp_path):
    text = config_text(security="""\
    guardrails:
      enabled: true
      evaluation_model: gpt-4o-mini
""")
    with pytest.raises(ValueError) as exc:
        load(tmp_path, text)

    assert "refusal_judge_model" in str(exc.value)


def test_a_disabled_section_needs_nothing(tmp_path):
    """A section that is switched off grades nothing, so it cannot grade
    itself. `examples/config.yaml` ships several of these."""
    text = config_text(security="""\
    red_teaming:
      enabled: false
      vulnerabilities: [toxicity]
    guardrails:
      enabled: false
""")
    assert load(tmp_path, text).targets[0].red_teaming["enabled"] is False


# --- Config load: the allow direction --------------------------------------


def test_a_separate_judge_target_loads(tmp_path):
    text = config_text(security="""\
    red_teaming:
      enabled: true
      evaluation_model:
        target: judge
""", extra_target=True)

    assert load(tmp_path, text).targets[0].name == "t1"


def test_a_provider_model_as_judge_loads(tmp_path):
    """A plain model string names a provider model, not a target, so it can
    never be the target under test."""
    text = config_text(security="""\
    red_teaming:
      enabled: true
      evaluation_model: gpt-4o-mini
""")

    assert load(tmp_path, text).targets[0].red_teaming["enabled"] is True


def test_a_target_may_still_simulate_its_own_attacks(tmp_path):
    """Deliberate scope limit. Self-simulation makes for a weaker scan; it
    does not invent a verdict, and requiring a second model to generate
    prompts would take the feature away from single-target configs."""
    text = config_text(security="""\
    red_teaming:
      enabled: true
      simulator_model:
        target: t1
      evaluation_model: gpt-4o-mini
""")

    assert load(tmp_path, text).targets[0].red_teaming["simulator_model"] == {"target": "t1"}


# --- Runtime: nothing falls back to the target under test ------------------


class FakeTarget:
    def __init__(self, name):
        self.name = name
        self.config = {"model": name}

    def get_model_name(self):
        return self.name


def run_red_team(config, find=lambda name: None):
    return asyncio.run(runners.run_red_teaming_async(FakeTarget("t1"), config, find))


def run_guardrails(config, find=lambda name: None):
    return asyncio.run(
        runners.run_guardrails_testing_async(FakeTarget("t1"), config, find)
    )


def test_a_red_team_scan_with_no_evaluator_fails_instead_of_self_judging():
    result = run_red_team({"vulnerabilities": ["toxicity"]})

    assert result["status"] == "failed"
    assert "evaluation_model" in result["error"]


def test_a_provider_judge_with_no_key_fails_instead_of_self_judging(monkeypatch):
    """Was the sneakiest of the three: the config names an OpenAI judge, the
    key is absent, and the target quietly took over the grading."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_red_team({"evaluation_model": "gpt-4o-mini"})

    assert result["status"] == "failed"
    assert "OPENAI_API_KEY" in result["error"]


def test_guardrails_with_no_refusal_judge_fails_instead_of_self_judging():
    """The evaluator resolves; only the refusal judge is missing. Guardrails
    needs both, and the refusal judge is the one that read "so guardrails
    never fails for lack of a judge" in the code that fell back to the
    target."""
    judge = FakeTarget("judge")
    result = run_guardrails(
        {"evaluation_model": {"target": "judge"}},
        find=lambda name: judge if name == "judge" else None,
    )

    assert result["status"] == "failed"
    assert "refusal_judge_model" in result["error"]


def test_a_named_judge_that_does_not_resolve_fails():
    """Config validation rejects a dangling name before this, so reaching
    here means the two disagree. Fail rather than substitute."""
    result = run_red_team({"evaluation_model": {"target": "missing"}})

    assert result["status"] == "failed"
    assert "missing" in result["error"]


def test_a_resolved_judge_is_the_one_that_gets_used(monkeypatch):
    """The allow direction, and it pins WHICH model was handed over: the
    judge target, not the target under test."""
    import surogate_eval.security as security

    handed_over = {}

    class FakeRunner:
        def __init__(self, target, config):
            handed_over["evaluation_model"] = config.evaluation_model

        async def run(self):
            return SimpleNamespace(to_dict=lambda: {"status": "completed"})

    monkeypatch.setattr(security, "RedTeamRunner", FakeRunner)

    judge = FakeTarget("judge")
    result = run_red_team(
        {"evaluation_model": {"target": "judge"}},
        find=lambda name: judge if name == "judge" else None,
    )

    assert result == {"status": "completed"}
    assert handed_over["evaluation_model"].get_model_name() == "judge"
