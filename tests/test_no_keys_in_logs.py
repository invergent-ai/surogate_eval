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


def _api_target(name: str, key, base_url: str = "http://127.0.0.1:9/v1"):
    config = {
        "name": name,
        "type": "llm",
        "provider": "openai",
        "model": "proxy-model-instruct",
        "base_url": base_url,
    }
    if key is not None:
        config["api_key"] = key
    return TargetFactory.create_target(config)


# --- lm-eval ----------------------------------------------------------


class _Stop(Exception):
    pass


def _run_lm_eval(monkeypatch, target):
    """Drive the real backend and the real `simple_evaluate` up to model
    construction, so lm-eval's own "Initializing ... with arguments" line is
    exercised, not imitated.

    The stub model records what lm-eval would authenticate with. In lm-eval
    0.4.9 that is `OPENAI_API_KEY` and never a `model_args` `api_key`; the
    header is read off the real `LocalCompletionsAPI.api_key` property.
    """
    import lm_eval.api.registry
    from lm_eval.models.openai_completions import LocalCompletionsAPI
    from surogate_eval.benchmarks.backends.lm_eval_backend import LMEvalBackend

    seen = {}

    class StubModel:
        @classmethod
        def create_from_arg_string(cls, arg_string, additional_config=None):
            seen["model_args"] = arg_string
            seen["env_key"] = os.environ.get("OPENAI_API_KEY")
            seen["header"] = {"Authorization": f"Bearer {LocalCompletionsAPI.api_key.fget(None)}"}
            raise _Stop("model construction stubbed")

    monkeypatch.setattr(lm_eval.api.registry, "get_model", lambda name: StubModel)

    with captured_logs() as lines:
        result = LMEvalBackend().evaluate(
            target, "canary_bench",
            {"source": "/nonexistent/data.jsonl", "task_type": "generation",
             "tokenizer": "gpt2"},
        )
    return seen, lines, result


def test_lm_eval_run_logs_no_target_key_but_lm_eval_still_gets_it(monkeypatch):
    """`local-completions`, the branch every ops target takes."""
    canary = "sk-agent-CANARYlmevalTarget0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pod-level-preexisting")

    seen, lines, result = _run_lm_eval(monkeypatch, _api_target("lm-eval-target", canary))

    _assert_absent(canary, lines)
    assert canary not in seen["model_args"]
    assert canary not in str(result)
    # The library-facing value still carries the real key...
    assert seen["env_key"] == canary
    assert seen["header"] == {"Authorization": f"Bearer {canary}"}
    # ...only for the duration of the call.
    assert os.environ["OPENAI_API_KEY"] == "sk-pod-level-preexisting"


@pytest.mark.parametrize("target_key", [None, "", "EMPTY"])
def test_keyless_target_is_not_sent_the_pods_openai_key(monkeypatch, target_key):
    """A target with no key must not authenticate with whatever key the pod
    holds (a judge's, say): that sends the pod's key to the user's endpoint."""
    pod_canary = "sk-CANARYpodJudgeKey0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", pod_canary)

    seen, _, _ = _run_lm_eval(monkeypatch, _api_target("keyless-target", target_key))

    assert pod_canary not in str(seen["header"])
    assert seen["env_key"] is None
    assert os.environ["OPENAI_API_KEY"] == pod_canary


def test_openai_branch_key_is_scoped_to_the_call(monkeypatch):
    """`openai-completions` reads the same variable; a target key left in it
    would reach a later provider-model judge."""
    canary = "sk-CANARYopenaiBranchTarget0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pod-level-preexisting")

    seen, lines, _ = _run_lm_eval(
        monkeypatch, _api_target("openai-target", canary, "https://api.openai.com/v1"),
    )

    _assert_absent(canary, lines)
    assert seen["env_key"] == canary
    assert os.environ["OPENAI_API_KEY"] == "sk-pod-level-preexisting"


def test_openai_branch_without_target_key_keeps_the_pods_openai_key(monkeypatch):
    """Talking to api.openai.com itself, the pod's OpenAI key is the right
    one; that path is left as it was."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-pod-level-openai-key")

    seen, _, _ = _run_lm_eval(
        monkeypatch, _api_target("openai-keyless", None, "https://api.openai.com/v1"),
    )

    assert seen["env_key"] == "sk-pod-level-openai-key"


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


@pytest.mark.parametrize("text, secret", [
    # repr and JSON forms, as a dict or a TaskConfig lands in an f-string
    ("{'api_key': 's3cr3t-repr-1', 'model': 'm'}", "s3cr3t-repr-1"),
    ('{"api_key": "s3cr3t-json-2", "model": "m"}', "s3cr3t-json-2"),
    ('{\n    "apiKey": "s3cr3t-json-3"\n}', "s3cr3t-json-3"),
    ("{'auth_token': 's3cr3t-repr-4'}", "s3cr3t-repr-4"),
    # any Authorization scheme, not only Bearer
    ("Authorization: Basic s3cr3t-basic-5", "s3cr3t-basic-5"),
    ("Authorization: Token s3cr3t-token-6", "s3cr3t-token-6"),
    ("{'Authorization': 'Digest s3cr3t-digest-7'}", "s3cr3t-digest-7"),
    ('"authorization": "ApiKey s3cr3t-apikey-8"', "s3cr3t-apikey-8"),
])
def test_redact_masks_secret_values_in_repr_json_and_header_text(text, secret):
    from surogate_eval.utils.redact import redact

    masked = redact(text)
    assert secret not in masked
    assert "***" in masked


@pytest.mark.parametrize("key", [
    "access_token", "api_token", "auth_token", "refresh_token", "hf_token",
    "bearer_token", "token", "HF_TOKEN", "GITHUB_TOKEN", "x-auth-token", "authToken",
    "api_key", "apiKey", "OPENAI_API_KEY", "client_secret", "password", "Authorization",
])
def test_credential_field_names_are_secret(key):
    from surogate_eval.utils.redact import is_secret_key

    assert is_secret_key(key)


@pytest.mark.parametrize("key", [
    "eos_token", "bos_token", "pad_token", "unk_token", "sep_token", "cls_token",
    "mask_token", "pad_token_id", "token_budget", "max_tokens", "max_token",
    "tokenizer", "tokenizer_backend", "num_tokens", "decoder_start_token",
])
def test_tokenizer_and_budget_fields_are_not_secret(key):
    from surogate_eval.utils.redact import is_secret_key, redact

    assert not is_secret_key(key)
    assert redact({key: "</s>"}) == {key: "</s>"}
    assert redact(f"{key}=</s>") == f"{key}=</s>"


def test_provider_keys_in_the_environment_at_startup_are_masked():
    """A key that only ever lives in the pod's environment is never in a
    config the runner registers, so it has to be picked up from there."""
    import importlib
    import surogate_eval.utils.redact as redact_mod

    canary = "sk-ant-CANARYenvOnlyKey0123456789"
    hf_canary = "hf_CANARYenvOnlyToken0123456789"
    benign = "CANARYnotAsecret0123456789"
    saved = {k: os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "HF_TOKEN", "SOME_SETTING")}
    os.environ.update(ANTHROPIC_API_KEY=canary, HF_TOKEN=hf_canary, SOME_SETTING=benign)
    try:
        redact_mod.register_env_secrets()
        assert redact_mod.scrub(f"x {canary} y {hf_canary}") == "x *** y ***"
        assert benign in redact_mod.scrub(benign)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert importlib.import_module("surogate_eval.utils.redact") is redact_mod


def test_scrub_masks_overlapping_secrets_longest_first():
    """The precomputed pattern must keep the old longest-first behaviour, so
    a secret that contains another is masked whole, not left half visible."""
    from surogate_eval.utils.redact import register_secrets, scrub

    short = "CANARYoverlap01"
    long = short + "-and-the-rest-of-it"
    register_secrets({"api_key": short})
    register_secrets({"api_key": long})
    assert scrub(f"a {long} b {short} c") == "a *** b *** c"


# --- tracebacks go through logging, masked ------------------------------


def test_no_traceback_bypasses_logging():
    """`traceback.print_exc()` writes straight to stderr, past the masking
    record factory. Every traceback the runner reports goes through the
    logger instead."""
    import pathlib
    import surogate_eval

    root = pathlib.Path(surogate_eval.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for path in root.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "print_exc(" in line or "print_exception(" in line
    ]
    assert not offenders, offenders


def _raising_loader(canary):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError(f"upstream rejected api_key={canary}")
    return Boom


@pytest.mark.parametrize("site", ["evaluation", "stress"])
def test_a_crash_traceback_is_logged_with_the_key_masked(monkeypatch, capfd, site):
    import surogate_eval.datasets as datasets
    from surogate_eval import runners
    from surogate_eval.utils.redact import register_secrets

    canary = f"sk-CANARYtraceback{site}0123456789"
    register_secrets({"api_key": canary})
    monkeypatch.setattr(datasets, "DatasetLoader", _raising_loader(canary))
    target = type("T", (), {"name": "t"})()

    with captured_logs() as lines:
        if site == "evaluation":
            runners.run_evaluation(
                target, {"name": "e", "dataset": "x.jsonl", "metrics": [{"type": "x"}]},
                find_target_fn=lambda name: None,
            )
        else:
            runners.run_stress_testing(target, {"dataset": "x.jsonl"})

    out = capfd.readouterr()
    assert canary not in out.out + out.err
    _assert_absent(canary, lines)
    assert any("Traceback" in line and "RuntimeError" in line for line in lines), \
        "the traceback must still be reported"
