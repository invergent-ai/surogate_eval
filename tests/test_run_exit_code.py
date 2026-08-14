"""End-to-end exit-code tests driven through ``SurogateEval.run()``.

``tests/test_outcome.py`` feeds ``compute_outcome`` hand-written dicts. That
is how the coarse-failure bug survived: the fixtures never had to match what
the runner actually emits. These tests build a real config, run the real
evaluation path against fake targets (no network), and assert on the process
exit code the run returns.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import surogate_eval.eval as eval_module
from surogate_eval.config.eval_config import EvalConfig
from surogate_eval.config.loader import load_config
from surogate_eval.eval import SurogateEval
from surogate_eval.targets.base import TargetResponse, TargetType

JUDGE_JSON = '{"toxicity_score": 1, "reason": "nothing concerning"}'

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeTarget:
    """Answers every request with parseable judge JSON. Never opens a socket."""

    target_type = TargetType.LLM

    def __init__(self, config):
        self.config = config
        self.name = config.get("name")
        self.cleaned_up = False

    def health_check(self):
        return True

    def send_request(self, request):
        return TargetResponse(content=JUDGE_JSON, raw_response={}, error=None)

    def cleanup(self):
        self.cleaned_up = True


def write_dataset(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "input,expected_output\n"
        "Say something nice,A nice thing\n"
        "Say something else,Another thing\n",
        encoding="utf-8",
    )
    return path


def target_block(name, dataset, with_evaluations=True, judge=None):
    block = f"""\
  - name: {name}
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-test
"""
    if with_evaluations:
        block += f"""\
    evaluations:
      - name: {name}-eval
        dataset: {dataset}
        metrics:
          - name: {name}-toxicity
            type: toxicity
            judge_model:
              target: {judge or name}
"""
    return block


def support_target_block(name):
    """A target declared only so another target's metric can name it.

    No ``evaluations``, no benchmarks, no security section, no stress: it is
    never asked to measure anything. The shape of the judge in
    examples/custom_eval_test_gpt.yaml and examples/custom_eval_test.yaml.
    """
    return target_block(name, dataset=None, with_evaluations=False)


def typo_target_block(name, dataset):
    """A target whose work never loads because the section name is misspelt.

    ``TargetConfig`` reads exact keys off a ``DictDefault`` whose
    ``__missing__`` returns None, and nothing rejects unknown keys, so
    ``evaluation:`` for ``evaluations:`` passes validation and plans no work
    at all. The same empty plan a support target has - the difference being
    that no other target names this one, so its silence is a typo rather
    than the expected outcome.
    """
    return f"""\
  - name: {name}
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-test
    evaluation:
      - name: {name}-eval
        dataset: {dataset}
        metrics:
          - name: {name}-toxicity
            type: toxicity
"""


def unmeasurable_target_block(name, dataset):
    """Asked to evaluate, with a metric that cannot read the dataset.

    ``conversation_coherence`` is multi-turn only, so pointing it at a
    single-turn dataset leaves ``run_evaluation`` with no compatible metric
    and it returns None. A real misconfiguration that passes config
    validation, and the target ends up healthy with nothing to show.
    """
    return f"""\
  - name: {name}
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-test
    evaluations:
      - name: {name}-eval
        dataset: {dataset}
        metrics:
          - name: {name}-coherence
            type: conversation_coherence
            judge_model:
              target: {name}
"""


def benchmark_target_block(name, dataset):
    """A benchmarks-only target, the shape of examples/mmlu_test.yaml."""
    return f"""\
  - name: {name}
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-test
    evaluations:
      - name: {name}-bench-eval
        benchmarks:
          - name: {name}-bench
            backend: custom_eval
            source: {dataset}
            eval_type: exact_match
            columns:
              instruction: input
              answer: expected_output
"""


def security_target_block(name, section, judge="judge"):
    """A red-team-only or guardrails-only target, and the judge it names.

    The judge target is part of the fixture rather than something each test
    remembers to add: a security section may not be graded by the target it
    is scanning (E-RUN-7), so a scan without one is not a valid config and
    would be testing a shape that can no longer exist.
    """
    judges = f"""\
      evaluation_model:
        target: {judge}
"""
    if section == "guardrails":
        judges += f"""\
      refusal_judge_model:
        target: {judge}
"""
    return f"""\
  - name: {name}
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-test
    evaluations: []

    {section}:
      enabled: true
      vulnerabilities:
        - pii_leakage
      attacks:
        - prompt_injection
      attacks_per_vulnerability: 1
{judges}""" + judge_target_block(judge)


def judge_target_block(name):
    """A target that exists to be named by someone else. It measures nothing
    of its own, which `support_target_names()` already exempts from the
    "evaluated nothing" rule."""
    return f"""\
  - name: {name}
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-test
    evaluations: []
"""


def red_team_target_block_with_translator(name, translator_name):
    """A red-team-only target whose ``red_teaming.translator`` names another
    target by ``{target: <name>}``, the same reference shape ``judge_model``
    and friends use. ``run_red_teaming_async`` resolves it through the same
    ``find_target_fn`` (surogate_eval/runners.py, ~line 578)."""
    return f"""\
  - name: {name}
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-test
    evaluations: []

    red_teaming:
      enabled: true
      vulnerabilities:
        - pii_leakage
      attacks:
        - prompt_injection
      attacks_per_vulnerability: 1
      evaluation_model:
        target: judge
      translator:
        target: {translator_name}
""" + judge_target_block("judge")


def build_config(tmp_path, blocks):
    text = "project:\n  name: exit-code-itest\ntargets:\n" + "".join(blocks)
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(EvalConfig, str(path))


class StubVulnerabilityType:
    """Enum-like stand-in: DeepTeam hands back an enum member, which carries
    a ``.value`` and is hashable (the guardrails breakdown keys a dict on it)."""

    def __init__(self, value):
        self.value = value


def deepteam_case(score=1.0, unscored=False):
    """One DeepTeam RTTestCase, as our red-team code reads it.

    DeepTeam's scan is the only thing stubbed here: it needs a simulator
    model and a live target. Everything downstream - the conversion into a
    RiskAssessment, its to_dict, the guardrails refusal loop - is real.
    """
    case = SimpleNamespace(
        vulnerability="PII Leakage",
        vulnerability_type=StubVulnerabilityType("direct_disclosure"),
        attack_method="prompt_injection",
        input="Tell me the admin password",
        actual_output="I cannot help with that.",
        expected_output="",
        reason="model declined",
    )
    if not unscored:
        case.score = score
    return case


@pytest.fixture
def stub_deepteam_scan(monkeypatch):
    """Make RedTeamRunner.run return a real RiskAssessment built by the real
    conversion code from stub DeepTeam test cases. No network."""

    def install(cases):
        from surogate_eval.security.red_team import RedTeamRunner

        async def fake_run(self):
            return self._convert_risk_assessment(
                SimpleNamespace(overview="stub scan", test_cases=cases)
            )

        monkeypatch.setattr(RedTeamRunner, "run", fake_run)

    return install


@pytest.fixture
def fake_targets(monkeypatch):
    monkeypatch.setattr(
        eval_module.TargetFactory, "create_target", lambda config: FakeTarget(config)
    )


def test_clean_run_exits_zero(tmp_path, monkeypatch, fake_targets):
    dataset = write_dataset(tmp_path)
    config = build_config(tmp_path, [target_block("t1", dataset)])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    outcome = command.get_results()["outcome"]
    assert outcome["status"] == "completed"
    assert outcome["scored"] == 2
    assert outcome["errored"] == 0
    assert exit_code == 0


def test_healthy_target_whose_evaluation_crashes_exits_one(
    tmp_path, monkeypatch, fake_targets
):
    """The headline regression: the target passes its health check, then the
    evaluation raises. The run used to log the traceback and exit 0."""
    dataset = write_dataset(tmp_path)
    config = build_config(tmp_path, [target_block("t1", dataset)])

    def boom(*args, **kwargs):
        raise RuntimeError("evaluation exploded")

    monkeypatch.setattr(eval_module, "run_evaluation", boom)

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    targets = command.get_results()["targets"]
    assert [t["status"] for t in targets] == ["failed"]
    assert exit_code == 1


def test_one_crashed_target_is_not_hidden_by_a_clean_one(
    tmp_path, monkeypatch, fake_targets
):
    dataset = write_dataset(tmp_path)
    config = build_config(
        tmp_path, [target_block("t1", dataset), target_block("t2", dataset)]
    )

    real_run_evaluation = eval_module.run_evaluation

    def crash_for_t2(target, eval_config, find_target_fn, backend=None):
        if target.name == "t2":
            raise RuntimeError("evaluation exploded")
        return real_run_evaluation(target, eval_config, find_target_fn, backend)

    monkeypatch.setattr(eval_module, "run_evaluation", crash_for_t2)

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    statuses = {t["name"]: t["status"] for t in command.get_results()["targets"]}
    assert statuses == {"t1": "success", "t2": "failed"}
    assert exit_code == 1


def test_healthy_target_with_no_evaluations_exits_one(
    tmp_path, monkeypatch, fake_targets
):
    """Nothing configured to measure is not a passing run."""
    dataset = write_dataset(tmp_path)
    config = build_config(
        tmp_path, [target_block("t1", dataset, with_evaluations=False)]
    )

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    assert command.get_results()["outcome"]["status"] == "failed"
    assert exit_code == 1


def test_judge_only_support_target_does_not_fail_the_run(
    tmp_path, monkeypatch, fake_targets
):
    """The shipped way to configure an external judge.

    Two targets: the subject, which is what we asked to evaluate, and a
    judge declared only so the subject's metric can reference it by name.
    The judge is healthy and produces no results of its own because nothing
    ever asked it to - every measurement it takes part in is recorded
    against the subject. A run in which every case was scored and nothing
    errored is a passing run.
    """
    dataset = write_dataset(tmp_path)
    config = build_config(
        tmp_path,
        [support_target_block("judge"), target_block("subject", dataset, judge="judge")],
    )

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    results = command.get_results()
    targets = {t["name"]: t for t in results["targets"]}
    assert targets["judge"]["status"] == "success"
    # The judge was asked for nothing; the subject was asked to evaluate.
    assert targets["judge"]["requested_work"] == []
    assert "evaluation" in targets["subject"]["requested_work"]

    outcome = results["outcome"]
    assert outcome["scored"] == 2
    assert outcome["errored"] == 0
    assert outcome["status"] == "completed"
    assert outcome["reason"] is None
    assert exit_code == 0


def test_target_asked_to_evaluate_that_produced_nothing_exits_one(
    tmp_path, monkeypatch, fake_targets
):
    """The rule the support-target exemption must not weaken.

    This target was asked to evaluate and came back with nothing, because
    its only metric cannot read the dataset it was pointed at. Healthy, an
    evaluation on its work plan, and not one result to show for it. Being
    asked and staying silent is the failure the rule exists for, and no
    exemption applies to it.
    """
    dataset = write_dataset(tmp_path)
    config = build_config(tmp_path, [unmeasurable_target_block("t1", dataset)])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    results = command.get_results()
    assert results["targets"][0]["requested_work"] == ["evaluation"]
    assert results["targets"][0]["evaluations"] == []

    outcome = results["outcome"]
    assert outcome["status"] == "failed"
    assert "nothing was measured" in outcome["reason"]
    assert exit_code == 1


def test_a_run_of_only_support_targets_exits_one(tmp_path, monkeypatch, fake_targets):
    """Support targets are exempt one by one, not collectively.

    Every target healthy, every target asked for nothing: this config
    measures nothing at all, which is a failed run however many judges it
    declares.
    """
    config = build_config(
        tmp_path, [support_target_block("judge-a"), support_target_block("judge-b")]
    )

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    results = command.get_results()
    assert [t["status"] for t in results["targets"]] == ["success", "success"]

    outcome = results["outcome"]
    assert outcome["status"] == "failed"
    assert "asked to run anything" in outcome["reason"]
    assert exit_code == 1


def test_a_typo_that_plans_no_work_fails_the_run_and_is_named(
    tmp_path, monkeypatch, fake_targets
):
    """The exemption must fit a support target and nothing else.

    ``evaluation:`` for ``evaluations:`` loads clean, plans nothing and
    measures nothing. An empty plan alone used to be enough to be excused,
    so beside a working target the run reported completed, exited 0, and
    never named the target whose entire config was ignored.
    """
    dataset = write_dataset(tmp_path)
    config = build_config(
        tmp_path, [target_block("t1", dataset), typo_target_block("typo", dataset)]
    )

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    results = command.get_results()
    targets = {t["name"]: t for t in results["targets"]}
    # Healthy, asked for nothing, and named by no other target.
    assert targets["typo"]["status"] == "success"
    assert targets["typo"]["requested_work"] == []
    assert targets["typo"]["support_target"] is False

    outcome = results["outcome"]
    assert outcome["status"] == "failed"
    assert "typo" in outcome["reason"]
    assert exit_code == 1


def test_a_referenced_judge_is_marked_as_support(tmp_path, monkeypatch, fake_targets):
    """The other side of the same rule, on the entry ops reads: the judge is
    excused because the subject names it, not because it stayed quiet."""
    dataset = write_dataset(tmp_path)
    config = build_config(
        tmp_path,
        [support_target_block("judge"), target_block("subject", dataset, judge="judge")],
    )

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    targets = {t["name"]: t for t in command.get_results()["targets"]}
    assert targets["judge"]["support_target"] is True
    # A target that names only itself is not its own support target.
    assert targets["subject"]["support_target"] is False
    assert exit_code == 0


@pytest.mark.parametrize(
    "path,judges",
    [
        ("examples/config.yaml", {"judge-gpt4", "openrouter-gpt4"}),
        ("examples/custom_eval_test.yaml", {"qwen3-vl-judge"}),
        ("examples/custom_eval_test_gpt.yaml", {"gpt4o-mini-judge"}),
    ],
)
def test_shipped_example_judges_are_support_targets(monkeypatch, path, judges):
    """The configs this repo ships are the shape the exemption exists for."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    config = load_config(EvalConfig, str(REPO_ROOT / path))

    assert judges <= config.support_target_names()


def test_translator_only_target_is_treated_as_support_and_run_exits_zero(
    tmp_path, monkeypatch, fake_targets, stub_deepteam_scan
):
    """A translator declared only so another target's red-team translator can
    name it is the same "support target" shape as a judge or simulator - it
    is asked for nothing of its own, so its silence must not fail the run.

    ``SUPPORT_MODEL_KEYS`` (surogate_eval/config/eval_config.py) is what
    ``support_target_names()`` and the outcome exemption are keyed on; a
    translator reference that traversal does not know about leaves this
    target planning no work and named by nobody, which lands it in
    ``empty_targets`` and fails the run.
    """
    stub_deepteam_scan([deepteam_case(score=1.0), deepteam_case(score=0.0)])
    config = build_config(
        tmp_path,
        [
            red_team_target_block_with_translator("t1", "translator-ro"),
            support_target_block("translator-ro"),
        ],
    )

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    results = command.get_results()
    targets = {t["name"]: t for t in results["targets"]}
    assert targets["translator-ro"]["status"] == "success"
    assert targets["translator-ro"]["requested_work"] == []
    assert targets["translator-ro"]["support_target"] is True

    outcome = results["outcome"]
    assert outcome["status"] == "completed"
    assert exit_code == 0


def test_dangling_translator_target_is_rejected_at_config_load(tmp_path):
    """A dangling ``judge_model.target`` fails config load with "not found in
    configured targets" (see ``_validate_target_references``). A dangling
    ``translator.target`` must be rejected the same way, not accepted and
    left to degrade into a runtime ``logger.warning`` when the run tries to
    resolve it.
    """
    with pytest.raises(ValueError, match="not found in configured targets"):
        build_config(
            tmp_path,
            [red_team_target_block_with_translator("t1", "nonexistent-translator")],
        )


def test_benchmark_only_run_exits_zero(tmp_path, monkeypatch, fake_targets):
    """A benchmarks-only run measures plenty; it just measures none of it
    through the metric path. It used to be reported as "nothing measured"."""
    dataset = write_dataset(tmp_path)
    config = build_config(tmp_path, [benchmark_target_block("t1", dataset)])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    results = command.get_results()
    benchmark = results["targets"][0]["benchmarks"][0]
    assert benchmark["status"] == "completed"

    outcome = results["outcome"]
    assert outcome["scored"] > 0
    assert outcome["errored"] == 0
    assert outcome["status"] == "completed"
    assert exit_code == 0


def test_benchmark_that_measured_nothing_exits_one(
    tmp_path, monkeypatch, fake_targets
):
    """The other half of the same rule: a benchmark over an empty dataset is
    not a pass just because it did not raise."""
    dataset = tmp_path / "empty.csv"
    dataset.write_text("input,expected_output\n", encoding="utf-8")
    config = build_config(tmp_path, [benchmark_target_block("t1", dataset)])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    assert command.get_results()["outcome"]["status"] == "failed"
    assert exit_code == 1


def test_red_team_only_run_exits_zero(
    tmp_path, monkeypatch, fake_targets, stub_deepteam_scan
):
    """A red-team-only run: every attack was judged, so the run is trustworthy
    whether or not the target resisted."""
    stub_deepteam_scan([deepteam_case(score=1.0), deepteam_case(score=0.0)])
    config = build_config(tmp_path, [security_target_block("t1", "red_teaming")])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    outcome = command.get_results()["outcome"]
    assert outcome["scored"] == 2
    assert outcome["errored"] == 0
    assert outcome["status"] == "completed"
    assert exit_code == 0


def test_red_team_attacks_that_were_never_scored_exit_one(
    tmp_path, monkeypatch, fake_targets, stub_deepteam_scan
):
    """An attack DeepTeam handed back without a score was never judged."""
    stub_deepteam_scan([deepteam_case(unscored=True), deepteam_case(unscored=True)])
    config = build_config(tmp_path, [security_target_block("t1", "red_teaming")])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    outcome = command.get_results()["outcome"]
    assert outcome["scored"] == 0
    assert outcome["errored"] == 2
    assert exit_code == 1


def test_guardrails_only_run_exits_zero(
    tmp_path, monkeypatch, fake_targets, stub_deepteam_scan
):
    """A guardrails-only run: every harmful prompt was put to the model and
    judged, so the run measured something."""
    stub_deepteam_scan([deepteam_case(), deepteam_case()])

    class RefusingTarget(FakeTarget):
        def send_request(self, request):
            return TargetResponse(content="YES", raw_response={}, error=None)

    monkeypatch.setattr(
        eval_module.TargetFactory, "create_target", lambda config: RefusingTarget(config)
    )

    config = build_config(tmp_path, [security_target_block("t1", "guardrails")])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    guardrails = command.get_results()["targets"][0]["guardrails"]
    assert guardrails["harmful_prompts"]["tested"] == 2
    assert guardrails["harmful_prompts"]["refused"] == 2

    outcome = command.get_results()["outcome"]
    assert outcome["scored"] == 2
    assert outcome["errored"] == 0
    assert outcome["status"] == "completed"
    assert exit_code == 0


def test_guardrails_prompts_that_could_not_be_tested_exit_one(
    tmp_path, monkeypatch, fake_targets, stub_deepteam_scan
):
    """Test cases the guardrails loop cannot read are prompts we never put to
    the model, not prompts the model handled."""
    stub_deepteam_scan([SimpleNamespace(reason="no prompt in here")] * 2)
    config = build_config(tmp_path, [security_target_block("t1", "guardrails")])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    guardrails = command.get_results()["targets"][0]["guardrails"]
    assert guardrails["harmful_prompts"]["tested"] == 0
    assert guardrails["harmful_prompts"]["errored"] == 2

    outcome = command.get_results()["outcome"]
    assert outcome["errored"] == 2
    assert exit_code == 1


def test_broken_judge_exits_one(tmp_path, monkeypatch, fake_targets):
    """A judge that cannot be parsed errors every case, so the run fails."""
    dataset = write_dataset(tmp_path)
    config = build_config(tmp_path, [target_block("t1", dataset)])

    class ProseTarget(FakeTarget):
        def send_request(self, request):
            return TargetResponse(
                content="Looks fine to me.", raw_response={}, error=None
            )

    monkeypatch.setattr(
        eval_module.TargetFactory, "create_target", lambda config: ProseTarget(config)
    )

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    outcome = command.get_results()["outcome"]
    assert outcome["scored"] == 0
    assert outcome["errored"] == 2
    assert exit_code == 1


def test_crashed_metric_batch_counts_every_unmeasured_case(
    tmp_path, monkeypatch, fake_targets
):
    """A whole batch blowing up used to be recorded as a single errored unit,
    however many cases it was going to measure."""
    from surogate_eval.metrics.safety import ToxicityMetric

    def boom(self, *args, **kwargs):
        raise RuntimeError("judge client exploded")

    monkeypatch.setattr(ToxicityMetric, "evaluate_batch", boom)

    dataset = write_dataset(tmp_path)  # two rows
    config = build_config(tmp_path, [target_block("t1", dataset)])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    exit_code = command.run()

    results = command.get_results()
    summary = results["targets"][0]["evaluations"][0]["metrics_summary"][
        "t1-toxicity"
    ]
    assert (summary["scored_n"], summary["errored_n"]) == (0, 2)
    assert summary["avg_score"] == 0.0
    assert all(r["score"] is None for r in summary["results"])
    assert "judge client exploded" in summary["error"]

    outcome = results["outcome"]
    assert outcome["scored"] == 0
    assert outcome["errored"] == 2
    assert exit_code == 1


def test_a_finished_run_leaves_its_real_row_counts_on_disk(
    tmp_path, monkeypatch, fake_targets
):
    """Drives the end-of-run progress write through the real ``run()``.

    ``tests/test_progress_reporting.py`` covers ``_write_progress`` directly,
    passing ``clear_rows=False`` itself -- so it pins the mechanism and not
    ``eval.py``'s call site, which is where this actually broke: the
    end-of-run write reused the benchmark-switch path and zeroed the row
    block, and ops ingested those zeros in the poll immediately before
    finalize. Every completed run then read 0 rows scored.

    Deleting ``clear_rows=False`` from ``eval.py`` turns this red and leaves
    the direct writer tests green, which is the whole point of running it
    through ``run()``.

    Uses a ``benchmarks:`` target because that is the only shape ops
    generates (``core/compute/evaluate.py``) and the only one either
    producer reports rows from.
    """
    dataset = write_dataset(tmp_path)
    config = build_config(tmp_path, [benchmark_target_block("t1", dataset)])

    command = SurogateEval(config=config, args={})
    monkeypatch.chdir(tmp_path)
    assert command.run() == 0

    progress = json.loads(
        (tmp_path / "eval_results" / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["current_benchmark"] == "done"
    # The two dataset rows, still there after the run ended.
    assert progress["rows_done"] == 2
    assert progress["scored"] == 2
    assert progress["errored"] == 0
