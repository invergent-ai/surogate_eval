"""The runner never writes a target or judge API key to its logs.

Ops starts an eval pod with the target's key and any judge keys in the pod
config, and the pod's stdout becomes the run's logs, so a key in any log line
is a key in the run's logs. Each test passes canary keys in through the real
entry points, captures every log record the code path produces (the runner's
own logger and the libraries' loggers alike), and asserts no canary appears,
while the value handed to the library still carries the real key.
"""

import logging
import os
from contextlib import contextmanager

import pytest

from surogate_eval.targets.factory import TargetFactory

LOGGERS = ("", "surogate_eval", "evalscope", "lm_eval")


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(message)s"))
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


@contextmanager
def captured_logs():
    """Every record, at every level, from the runner's and libraries' loggers."""
    handler = _Collect()
    saved = []
    for name in LOGGERS:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level))
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
    try:
        yield handler.lines
    finally:
        for lg, level in saved:
            lg.removeHandler(handler)
            lg.setLevel(level)


def _assert_absent(canary: str, lines: list[str]):
    assert lines, "nothing was logged, so the test proves nothing"
    leaked = [line for line in lines if canary in line]
    assert not leaked, f"key leaked into {len(leaked)} log line(s): {leaked[:3]}"


def _api_target(name: str, key: str):
    return TargetFactory.create_target({
        "name": name,
        "type": "llm",
        "provider": "openai",
        "model": "proxy-model-instruct",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": key,
    })


# --- lm-eval ----------------------------------------------------------


class _Stop(Exception):
    pass


def test_lm_eval_run_logs_no_target_key_but_lm_eval_still_gets_it(monkeypatch):
    """`local-completions`, the branch every ops target takes.

    `simple_evaluate` runs for real up to model construction, so lm-eval's
    own "Initializing ... with arguments" line is exercised, not imitated.
    The stub model records what lm-eval would authenticate with, which in
    lm-eval 0.4.9 is `OPENAI_API_KEY` and never a `model_args` `api_key`.
    """
    import lm_eval.api.registry
    from surogate_eval.benchmarks.backends.lm_eval_backend import LMEvalBackend

    canary = "sk-agent-CANARYlmevalTarget0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pod-level-preexisting")
    seen = {}

    class StubModel:
        @classmethod
        def create_from_arg_string(cls, arg_string, additional_config=None):
            seen["model_args"] = arg_string
            seen["env_key"] = os.environ.get("OPENAI_API_KEY")
            raise _Stop("model construction stubbed")

    monkeypatch.setattr(lm_eval.api.registry, "get_model", lambda name: StubModel)

    target = _api_target("lm-eval-target", canary)
    with captured_logs() as lines:
        result = LMEvalBackend().evaluate(
            target, "canary_bench",
            {"source": "/nonexistent/data.jsonl", "task_type": "generation",
             "tokenizer": "gpt2"},
        )

    _assert_absent(canary, lines)
    assert canary not in seen["model_args"]
    assert canary not in str(result)
    # The library-facing value still carries the real key...
    assert seen["env_key"] == canary
    # ...only for the duration of the call.
    assert os.environ["OPENAI_API_KEY"] == "sk-pod-level-preexisting"


# --- evalscope --------------------------------------------------------


def test_evalscope_tau_bench_judge_key_not_logged_but_reaches_extra_params():
    """evalscope logs its whole TaskConfig (run.py `logger.info(task_config)`).

    Its `to_dict` drops the top level `api_key` and `judge_model_args.api_key`
    but not the judge key the runner puts in tau bench's `extra_params`.
    """
    from evalscope.utils.logger import get_logger as evalscope_logger
    from surogate_eval.benchmarks.backends.evalscope_backend import EvalScopeBackend

    target_canary = "sk-agent-CANARYevalscopeTarget0123456789"
    judge_canary = "sk-CANARYevalscopeJudge0123456789"
    target = _api_target("es-target", target_canary)
    judge = _api_target("es-judge", judge_canary)

    with captured_logs() as lines:
        task_config = EvalScopeBackend()._prepare_task_config(
            target, "tau_bench",
            {"backend_params": {"judge_target": judge, "use_sandbox": False}},
        )
        evalscope_logger().info(task_config)

    _assert_absent(judge_canary, lines)
    _assert_absent(target_canary, lines)
    assert task_config.dataset_args["tau_bench"]["extra_params"]["api_key"] == judge_canary
    assert task_config.judge_model_args["api_key"] == judge_canary
    assert task_config.api_key == target_canary


# --- the runner's own log sites -----------------------------------------


def test_config_debug_dump_logs_no_keys(tmp_path):
    from surogate_eval.config.eval_config import EvalConfig
    from surogate_eval.config.loader import load_config

    canary = "sk-CANARYloaderTarget0123456789"
    header_canary = "CANARYloaderHeader0123456789"
    path = tmp_path / "eval.yaml"
    path.write_text(
        "project:\n  name: test\n"
        "targets:\n"
        "  - name: t1\n    type: llm\n    provider: openai\n    model: m\n"
        f"    api_key: {canary}\n"
        f"    headers:\n      Authorization: Bearer {header_canary}\n",
        encoding="utf-8",
    )
    with captured_logs() as lines:
        config = load_config(EvalConfig, str(path))

    _assert_absent(canary, lines)
    _assert_absent(header_canary, lines)
    assert config.targets[0].api_key == canary


def test_target_build_failure_logs_no_key():
    canary = "sk-CANARYfactoryTarget0123456789"
    with captured_logs() as lines:
        TargetFactory.create_multiple_targets([
            {"name": "bad", "type": "not-a-type", "api_key": canary},
        ])
    _assert_absent(canary, lines)


def test_metric_build_failure_logs_no_inline_judge_key():
    """Metric configs never pass through the target factory, so this is the
    key-name redaction on its own, not the remembered-value net."""
    from surogate_eval.metrics.registry import MetricRegistry

    canary = "sk-CANARYmetricJudge0123456789"
    with captured_logs() as lines:
        MetricRegistry.create_metrics([
            {"type": "not-a-metric", "judge_model": {"model": "j", "api_key": canary}},
        ])
    _assert_absent(canary, lines)


# --- the helper -------------------------------------------------------


@pytest.mark.parametrize("value, secret", [
    ({"api_key": "s3cr3t-value-1"}, "s3cr3t-value-1"),
    ({"apiKey": "s3cr3t-value-2"}, "s3cr3t-value-2"),
    ({"headers": {"Authorization": "Bearer s3cr3t-value-3"}}, "s3cr3t-value-3"),
    ({"x": [{"auth_token": "s3cr3t-value-4"}]}, "s3cr3t-value-4"),
    ({"client_secret": "s3cr3t-value-5"}, "s3cr3t-value-5"),
    ("model=m,api_key=s3cr3t-value-6,base_url=u", "s3cr3t-value-6"),
    ("Authorization: Bearer s3cr3t-value-7", "s3cr3t-value-7"),
])
def test_redact_masks_secret_fields(value, secret):
    from surogate_eval.utils.redact import redact

    assert secret not in str(redact(value))


def test_redact_keeps_everything_else_and_leaves_the_original_alone():
    from surogate_eval.utils.redact import redact

    original = {"model": "m", "base_url": "u", "max_tokens": 5, "api_key": "s3cr3t-keep"}
    masked = redact(original)
    assert masked == {"model": "m", "base_url": "u", "max_tokens": 5, "api_key": "***"}
    assert original["api_key"] == "s3cr3t-keep"
    assert redact("model=m,base_url=u") == "model=m,base_url=u"
    # A value that only mentions a key name is not a secret field.
    assert redact({"max_tokens": 5}) == {"max_tokens": 5}
