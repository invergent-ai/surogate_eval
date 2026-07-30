"""End-to-end exit-code tests driven through ``SurogateEval.run()``.

``tests/test_outcome.py`` feeds ``compute_outcome`` hand-written dicts. That
is how the coarse-failure bug survived: the fixtures never had to match what
the runner actually emits. These tests build a real config, run the real
evaluation path against fake targets (no network), and assert on the process
exit code the run returns.
"""

import pytest

import surogate_eval.eval as eval_module
from surogate_eval.config.eval_config import EvalConfig
from surogate_eval.config.loader import load_config
from surogate_eval.eval import SurogateEval
from surogate_eval.targets.base import TargetResponse, TargetType

JUDGE_JSON = '{"toxicity_score": 1, "reason": "nothing concerning"}'


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


def target_block(name, dataset, with_evaluations=True):
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
              target: {name}
"""
    return block


def build_config(tmp_path, blocks):
    text = "project:\n  name: exit-code-itest\ntargets:\n" + "".join(blocks)
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(EvalConfig, str(path))


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
