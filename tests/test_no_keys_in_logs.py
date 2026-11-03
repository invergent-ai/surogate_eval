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


@pytest.fixture(autouse=True, scope="module")
def _masking_installed():
    """What the runner's entry points do first. Importing the package must
    not do it (see test_importing_the_runner_leaves_the_process_hooks_alone),
    so these tests do it themselves, and put the process back afterwards."""
    import sys
    import threading
    from surogate_eval.utils.redact import install_log_masking

    saved = (logging.getLogRecordFactory(), sys.excepthook, threading.excepthook)
    install_log_masking()
    yield
    import surogate_eval.utils.redact as redact_mod

    logging.setLogRecordFactory(saved[0])
    sys.excepthook, threading.excepthook = saved[1], saved[2]
    redact_mod._share_with_children = False
    os.environ.pop(redact_mod._SHARED_ENV, None)

LOGGERS = ("", "surogate_eval", "evalscope", "lm_eval")


def _runner_formatter() -> logging.Formatter:
    """The formatter on the runner's real console handler, the one whose
    output becomes the run's logs. A plain Formatter would print tracebacks
    the real one drops, and a test using it would miss exactly that."""
    from surogate_eval.utils.logger import get_logger

    get_logger()
    for handler in logging.getLogger("surogate_eval").handlers:
        if type(handler) is logging.StreamHandler and handler.formatter:
            return handler.formatter
    raise AssertionError("the runner's console handler is missing")


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.setFormatter(_runner_formatter())
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
    """`traceback.print_exc()` writes straight to stderr, past the masking.
    Every traceback the runner reports goes through the logger, and by one
    path, `exc_info`, which the runner's formatter prints masked, rather
    than a pre-formatted `format_exc()` string."""
    import pathlib
    import surogate_eval

    root = pathlib.Path(surogate_eval.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for path in root.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if any(call in line for call in ("print_exc(", "print_exception(", "format_exc("))
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
            result = runners.run_evaluation(
                target, {"name": "e", "dataset": "x.jsonl", "metrics": [{"type": "x"}]},
                find_target_fn=lambda name: None,
            )
        else:
            result = runners.run_stress_testing(target, {"dataset": "x.jsonl"})

    out = capfd.readouterr()
    assert canary not in out.out + out.err
    _assert_absent(canary, lines)
    assert any("Traceback" in line and "RuntimeError" in line for line in lines), \
        "the traceback must still be reported"
    # What ops stores as the run's result carries no key either.
    assert canary not in str(result)
    assert "api_key=***" in str(result)


def test_cli_launch_failure_logs_the_error_with_the_key_masked(monkeypatch, capfd):
    """The launcher process holds the pod's env keys too."""
    from surogate_eval.cli import main
    from surogate_eval.utils.redact import register_secrets

    canary = "sk-CANARYcliLaunch0123456789"
    register_secrets({"api_key": canary})

    def boom(*a, **k):
        raise RuntimeError(f"launch refused with {canary}")

    monkeypatch.setattr(main.subprocess, "Popen", boom)
    monkeypatch.setattr(main, "get_system_info", lambda: {})
    monkeypatch.setattr(main, "print_system_diagnostics", lambda info: None)
    monkeypatch.setattr(main, "get_torchrun_args", lambda: None)
    monkeypatch.setattr(main, "torch", None)
    monkeypatch.setattr("sys.argv", ["surogate-eval", "eval", "--config", "x.yaml"])
    installed = []
    monkeypatch.setattr(main, "install_log_masking", lambda: installed.append(True))

    with captured_logs() as lines:
        assert main.cli_main() == 1

    assert installed, "the launcher entry point must install the masking"

    out = capfd.readouterr()
    assert canary not in out.out + out.err
    _assert_absent(canary, lines)
    assert any("An error occurred: RuntimeError: launch refused with ***" in line
               for line in lines), "the error itself must be logged"


def test_importing_the_runner_leaves_the_process_hooks_alone():
    """Masking and the crash hooks are installed by the runner's entry
    points, not as a side effect of importing a library module: ops or a
    notebook importing surogate_eval keeps its own hooks. The eval entry
    script, surogate_eval.cli.eval, does install on import; that is what
    covers its spawn children."""
    import subprocess
    import sys

    code = (
        "import logging, sys, threading\n"
        "import surogate_eval, surogate_eval.cli.main\n"
        "import surogate_eval.eval, surogate_eval.runners, surogate_eval.utils.redact\n"
        "print(sys.excepthook is sys.__excepthook__,\n"
        "      threading.excepthook is threading.__excepthook__,\n"
        "      logging.getLogRecordFactory() is logging.LogRecord)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, timeout=300).stdout
    assert out.split()[-3:] == ["True", "True", "True"], out


def test_a_logged_traceback_reaches_the_runners_console_output():
    """The runner's formatter used to print the message only, so a record
    logged with exc_info lost its traceback on the way to the run's logs."""
    from surogate_eval.utils.logger import get_logger

    try:
        raise ValueError("the cause")
    except ValueError:
        with captured_logs() as lines:
            get_logger().error("it failed", exc_info=True)

    [line] = [line for line in lines if "it failed" in line]
    assert "Traceback" in line and "ValueError: the cause" in line


def test_unregistered_keys_in_library_lines_and_tracebacks_are_masked_by_name():
    """The record factory masks secret fields by name too, so a key the
    runner never registered (a library's own, say) is still masked."""
    canary = "sk-CANARYunregisteredLib0123456789"
    lib = logging.getLogger("some_library")
    with captured_logs() as lines:
        lib.warning("retrying with {'api_key': '%s', 'model': 'm'}", canary)
        lib.warning("headers: Authorization: Basic %s", canary)
        try:
            raise RuntimeError(f"401 for url https://h/v1?api_key={canary}&x=1")
        except RuntimeError:
            lib.error("request failed", exc_info=True)
        # Ordinary prompt text is left alone.
        lib.info("The bearer of the token said: hello, and the password field was empty")

    _assert_absent(canary, lines)
    assert any("The bearer of the token said: hello" in line for line in lines)


@pytest.mark.parametrize("text, secret", [
    ("url: https://h/v1?api_key=s3cr3t-query-1&x=1", "s3cr3t-query-1"),
    ("GET https://h/v1?token=s3cr3t-query-2", "s3cr3t-query-2"),
    ("{'password': 'correct horse s3cr3t-space-3'}", "s3cr3t-space-3"),
    ('{"api_key": "sk two s3cr3t-space-4"}', "s3cr3t-space-4"),
])
def test_redact_reaches_keys_inside_other_values_and_whole_quoted_values(text, secret):
    from surogate_eval.utils.redact import redact

    masked = redact(text)
    assert secret not in masked
    assert "correct horse" not in masked and "sk two" not in masked


def test_redact_leaves_non_secret_values_intact():
    from surogate_eval.utils.redact import redact

    text = "url: https://h/v1?model=m&x=1, 'name': 'two words'"
    assert redact(text) == text


def test_stress_tester_failure_records_carry_no_key(monkeypatch):
    """The per-request error rows the stress tester stores."""
    from surogate_eval.metrics.stress.stress_tester import StressTester

    canary = "sk-CANARYstressRow0123456789"

    class Target:
        name = "t"

        def send_request(self, request):
            raise RuntimeError(f"rejected Authorization: Bearer {canary}")

    from surogate_eval.datasets.test_case import TestCase

    tester = StressTester.__new__(StressTester)
    tester.target = Target()
    row = tester._send_single_request(TestCase(input="hi"), 1)
    assert row["success"] is False
    assert canary not in str(row)


def test_an_uncaught_exception_in_the_runner_process_is_logged_masked(tmp_path):
    """The eval child process: nothing catches an exception that escapes its
    __main__ block, so Python's default hook printed it raw to stderr."""
    import subprocess
    import sys

    canary = "sk-CANARYuncaught0123456789"
    env_canary = "sk-CANARYuncaughtEnvKey0123456789"
    code = (
        "import os, runpy, sys\n"
        "sys.argv = ['eval', '--config', 'x.yaml']\n"
        "import surogate_eval.config.loader as loader\n"
        "def boom(*a, **k):\n"
        f"    raise RuntimeError('upstream api_key={canary} env ' + os.environ['OPENAI_API_KEY'])\n"
        "loader.load_config = boom\n"
        "runpy.run_module('surogate_eval.cli.eval', run_name='__main__')\n"
    )
    env = dict(os.environ, OPENAI_API_KEY=env_canary, PYTHONPATH=str(Path_root()))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=tmp_path, timeout=300)
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert canary not in output and env_canary not in output, output[-2000:]
    assert "Traceback" in output and "RuntimeError" in output, output[-2000:]


def Path_root():
    import pathlib
    import surogate_eval

    return pathlib.Path(surogate_eval.__file__).parent.parent


# --- exception text stored in results, reasons and rows -------------------


def _boom(canary):
    def raise_it(*a, **k):
        raise RuntimeError(f"upstream rejected api_key={canary}")
    return raise_it


def _metric_reason(monkeypatch, tmp_path, canary):
    from surogate_eval.datasets.test_case import TestCase
    from surogate_eval.metrics.base import BaseMetric

    class Exploding(BaseMetric):
        def _validate_config(self):
            pass

        evaluate = staticmethod(_boom(canary))

    batch = Exploding({"name": "boom", "type": "toxicity"}).evaluate_batch(
        [TestCase(input="a")], ["x"],
    )
    return batch.results[0].reason


def _evalscope_row(monkeypatch, tmp_path, canary):
    import json
    from surogate_eval.benchmarks.backends.evalscope_backend import EvalScopeBackend

    reviews = tmp_path / "reviews" / "m"
    reviews.mkdir(parents=True)
    (reviews / "gsm8k_default.jsonl").write_text(json.dumps({"x": 1}) + "\n")
    backend = EvalScopeBackend.__new__(EvalScopeBackend)
    monkeypatch.setattr(backend, "_review_row_to_record", _boom(canary))
    return backend._load_predictions(str(tmp_path), "m", "gsm8k")


def _adapter_row(monkeypatch, tmp_path, canary):
    from surogate_eval.benchmarks import BenchmarkConfig
    from surogate_eval.benchmarks.adapters import vita_bench

    monkeypatch.setattr(vita_bench, "_download_dataset", lambda: tmp_path)
    monkeypatch.setattr(vita_bench, "_load_tasks", lambda d, limit=None: [{"id": "t1"}])
    monkeypatch.setattr(vita_bench, "_format_prompt", lambda task: "p")
    monkeypatch.setattr(vita_bench, "_format_expected", lambda task: "e")
    target = type("T", (), {"name": "t", "send_request": staticmethod(_boom(canary))})()
    result = vita_bench.VitaBenchmark(BenchmarkConfig(name="vita_bench")).evaluate(target)
    return result.detailed_results


def _validator_errors(monkeypatch, tmp_path, canary):
    from surogate_eval.datasets import loader
    from surogate_eval.datasets.validator import DatasetValidator

    monkeypatch.setattr(loader.DatasetLoader, "load", _boom(canary))
    return DatasetValidator().validate_file("x.jsonl")


@pytest.mark.parametrize("stored", [
    _metric_reason, _evalscope_row, _adapter_row, _validator_errors,
], ids=["metrics", "benchmark-backends", "benchmark-adapters", "datasets"])
def test_exception_text_stored_in_results_carries_no_key(monkeypatch, tmp_path, stored):
    """One site per module family: what a failure stores for ops to show
    is the exception text, and that text can hold a key."""
    canary = f"sk-CANARYstored{stored.__name__}0123456789"
    value = str(stored(monkeypatch, tmp_path, canary))
    assert "upstream rejected api_key=***" in value, value
    assert canary not in value


# --- masking happens when a line is written, and only on credential forms --


class _CountingArg:
    calls = 0

    def __str__(self):
        type(self).calls += 1
        return "value"


def test_a_record_no_handler_writes_is_never_formatted():
    """Masking runs when a handler writes the line, so a record every handler
    drops costs nothing, not a format and a regex pass at creation."""
    lg = logging.getLogger("surogate_eval_test_dropped")
    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    try:
        lg.info("dropped %s", _CountingArg())
    finally:
        lg.removeHandler(handler)
    assert _CountingArg.calls == 0


@pytest.mark.parametrize("name", ["evalscope", "lm_eval.evaluator", "some.other.library"])
def test_library_loggers_are_masked(name):
    """evalscope's logger has its own handlers and does not propagate;
    lm-eval's propagates to the root. Both must be masked."""
    from surogate_eval.utils.redact import register_secrets

    canary = f"sk-CANARYlib{name.replace('.', '')}0123456789"
    register_secrets({"api_key": canary})
    with captured_logs() as lines:
        logging.getLogger(name).warning("config %s", {"key": canary})
    _assert_absent(canary, lines)


@pytest.mark.parametrize("text", [
    "time_to_first_token: 0.123s",
    "input_token: 5",
    "Q: What is a token: a unit?",
    "secret: the meaning of life",
    "token=abc123def456",
    # Prompt prose is output text: masked only if it holds a registered key.
    "Please enter your password: hunter2",
    "The bearer of the token said: hello, and the password field was empty",
])
def test_ordinary_lines_are_written_unchanged(text):
    with captured_logs() as lines:
        logging.getLogger("some_library").warning(text)
    assert any(line.endswith(text) for line in lines), lines


@pytest.mark.parametrize("template", [
    "api_key={}",
    "apikey: {}",
    "x-api-key: {}",
    "OPENAI_API_KEY={}",
    "{{'access_token': '{}'}}",
    '{{"client_secret": "{}"}}',
    "GET https://h/v1?hf_token={}&x=1",
    "refresh_token={}",
    "auth_token: {}",
    "api_token={}",
    "bearer_token={}",
    "aws_secret_key={}",
    "password={}",
    "{{'password': '{}'}}",
    "Authorization: Basic {}",
    "headers: Authorization: Token {}",
    "sending Bearer {}",
])
def test_credential_forms_are_masked_in_written_lines(template):
    canary = "CANARYcred0123456789"
    with captured_logs() as lines:
        logging.getLogger("some_library").warning(template.format(canary))
    _assert_absent(canary, lines)


def test_a_credential_after_a_colour_code_is_masked_and_the_reset_survives():
    """The runner's logger wraps every message in ANSI colour codes."""
    from surogate_eval.utils.logger import get_logger

    canary = "CANARYansi0123456789"
    with captured_logs() as lines:
        get_logger().info(f"api_key={canary}")
        get_logger().info(f"access_token: {canary}")
    _assert_absent(canary, lines)
    for line in lines[-2:]:
        assert line.endswith("***\x1b[0m"), repr(line)


def test_a_thread_crash_is_logged_masked_with_its_thread_name():
    import threading
    from surogate_eval.utils.redact import register_secrets

    canary = "sk-CANARYthreadCrash0123456789"
    register_secrets({"api_key": canary})

    def crash():
        raise RuntimeError(f"worker saw {canary}")

    with captured_logs() as lines:
        worker = threading.Thread(target=crash, name="progress-watcher")
        worker.start()
        worker.join()
    _assert_absent(canary, lines)
    assert any("Exception in thread progress-watcher" in line and "RuntimeError" in line
               for line in lines), lines


def test_a_judge_parse_error_cannot_keep_half_a_key():
    """The judge wrapper cuts the underlying error to 200 characters. Cut
    first, and a key straddling the cut survives as a prefix no mask
    recognises."""
    from surogate_eval.errors import JudgeParseError
    from surogate_eval.models.deepeval_wrapper import DeepEvalTargetWrapper
    from surogate_eval.targets.base import TargetResponse
    from surogate_eval.utils.redact import register_secrets

    canary = "sk-CANARYjudgeCut0123456789abcdef"
    register_secrets({"api_key": canary})

    class Target:
        name = "judge"
        config = {"base_url": "http://127.0.0.1:9/v1", "provider": "vllm"}

        def send_request(self, request):
            return TargetResponse(content='{"a": 1} trailing', raw_response={})

    class Schema:
        @classmethod
        def model_json_schema(cls):
            return {}

        def __init__(self, **kwargs):
            raise ValueError("x" * 185 + f" got {canary} back")

    wrapper = DeepEvalTargetWrapper(Target())
    with pytest.raises(JudgeParseError) as err:
        wrapper.generate("p", Schema)
    assert canary[:10] not in str(err.value), str(err.value)


# --- last round: linear time, spawn children, names, cuts, one mask per record


@pytest.mark.parametrize("text", [
    "'A' * 200_000",
    "'QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkrLz0' * 4_300",
    "'api_key' * 30_000",
])
def test_masking_a_long_line_is_linear(text):
    """A long unbroken run (a base64 blob, a dumped tensor) once took seconds
    per 10k characters and never finished at 20k: every start position
    rescanned the rest of the word. Run in a child so a regression fails on
    the timeout instead of hanging the suite."""
    import subprocess
    import sys

    code = (
        "import time\n"
        "from surogate_eval.utils.redact import mask_line, redact_text\n"
        f"text = {text}\n"
        "for mask in (mask_line, redact_text):\n"
        "    start = time.perf_counter(); mask(text)\n"
        "    print(mask.__name__, time.perf_counter() - start)\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, timeout=60).stdout
    except subprocess.TimeoutExpired:
        pytest.fail("masking a 200k-character line did not finish in 60s")
    for name, seconds in (line.split() for line in out.splitlines()):
        assert float(seconds) < 1.0, (name, seconds)


def test_a_spawned_child_of_the_runner_masks_its_logs(tmp_path):
    """evalscope runs code benchmarks in multiprocessing spawn children,
    which import the entry script as __mp_main__ and so skip its
    `if __name__ == '__main__'` block."""
    import subprocess
    import sys

    env_canary = "sk-CANARYspawnEnvKey0123456789"
    named_canary = "CANARYspawnNamed0123456789"
    # Registered by the parent from its config (an inline judge key, say):
    # the child never loads that config, so it has to be handed down.
    config_canary = "sk-CANARYspawnConfigKey0123456789"
    (tmp_path / "spawn_probe.py").write_text(
        "import os\n"
        "def child():\n"
        "    from surogate_eval.utils.logger import get_logger\n"
        "    get_logger().warning('child env ' + os.environ['OPENAI_API_KEY'])\n"
        f"    get_logger().warning('child api_key={named_canary}')\n"
        f"    get_logger().warning('child config {config_canary}')\n"
    )
    (tmp_path / "sitecustomize.py").write_text(
        "import multiprocessing, sys\n"
        "def _patch():\n"
        "    import surogate_eval.config.loader as loader\n"
        "    def start_child(*a, **k):\n"
        "        from surogate_eval.utils.redact import register_secrets\n"
        f"        register_secrets({{'judge': {{'api_key': '{config_canary}'}}}})\n"
        "        import spawn_probe\n"
        "        p = multiprocessing.get_context('spawn').Process(target=spawn_probe.child)\n"
        "        p.start(); p.join()\n"
        "        sys.exit(p.exitcode)\n"
        "    loader.load_config = start_child\n"
        "if multiprocessing.parent_process() is None:\n"
        "    _patch()\n"
    )
    env = dict(os.environ, OPENAI_API_KEY=env_canary,
               PYTHONPATH=f"{tmp_path}{os.pathsep}{Path_root()}")
    proc = subprocess.run(
        [sys.executable, "-m", "surogate_eval.cli.eval", "--config", "x.yaml"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output[-3000:]
    assert "child env ***" in output and "child api_key=***" in output, output[-3000:]
    assert "child config ***" in output, output[-3000:]
    assert env_canary not in output and named_canary not in output
    assert config_canary not in output


@pytest.mark.parametrize("template", [
    "AWS_SECRET_ACCESS_KEY={}",
    "aws_secret_access_key: {}",
    "private_key={}",
    "{{'client_secret': '{}'}}",
    "passwd={}",
    "GITHUB_TOKEN={}",
    "github_token: {}",
    "NOTION_TOKEN={}",
    "HF_TOKEN={}",
    "hf_token: {}",
    "CLOUDFLARE_API_TOKEN={}",
    "MYSERVICE_SECRET_KEY={}",
    "SLACK_ACCESS_TOKEN={}",
    "DB_PASSWORD: {}",
    "db_password: {}",
    "PASSWORD: {}",
])
def test_explicit_credential_names_are_masked(template):
    canary = "CANARYnamed0123456789"
    with captured_logs() as lines:
        logging.getLogger("some_library").warning(template.format(canary))
    _assert_absent(canary, lines)


def test_mcpmark_output_is_masked_before_it_is_cut(monkeypatch, tmp_path):
    """The adapter logs 500 characters of the tool's stderr and stdout. Cut
    first, and a registered key straddling the cut survives as a prefix."""
    import subprocess
    from surogate_eval.benchmarks.adapters import mcp_mark
    from surogate_eval.utils.redact import register_secrets

    canary = "sk-CANARYmcpmarkCut0123456789abcdef"
    register_secrets({"api_key": canary})
    # 19 characters of the key either side of the 500-character cut.
    straddle_head = "e" * 480 + f" {canary} tail"
    straddle_tail = f"head {canary} " + "o" * 480

    def fake_run(cmd, **kwargs):
        if cmd[0] == "git":
            (tmp_path / "mcpmark").mkdir(exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, straddle_tail, straddle_head)

    monkeypatch.setattr(mcp_mark.subprocess, "run", fake_run)
    with captured_logs() as lines:
        mcp_mark._clone_repo(tmp_path)
        mcp_mark._run_mcpmark_pipeline(
            tmp_path / "mcpmark", "m", "http://127.0.0.1:9/v1", "k", ["filesystem"],
        )
    assert lines
    for line in lines:
        assert canary[:12] not in line and canary[-12:] not in line, line[-300:]


def test_a_record_is_masked_once_however_many_handlers_write_it(monkeypatch):
    import surogate_eval.utils.redact as redact_mod

    calls = []
    real = redact_mod.mask_line
    monkeypatch.setattr(redact_mod, "mask_line", lambda text: calls.append(1) or real(text))
    record = logging.getLogRecordFactory()(
        "x", logging.INFO, __file__, 1, "api_key=%s", ("s3cr3t-value",), None,
    )
    assert record.getMessage() == "api_key=***"
    assert record.getMessage() == "api_key=***"
    assert len(calls) == 1
    # A filter that rewrites the message after the first read is honoured.
    record.msg, record.args = "plain", None
    assert record.getMessage() == "plain"


# --- final round: redact's key boundary, the record cache, spawn hand-down


@pytest.mark.parametrize("text", [
    "--api-key=sk-CANARYdash0123456789",
    "--api_key=sk-CANARYdash0123456789",
    "--hf-token=sk-CANARYdash0123456789",
    "--password=sk-CANARYdash0123456789",
    "\x1b[32mapi_key=sk-CANARYdash0123456789\x1b[0m",
    "\x1b[32mpassword=sk-CANARYdash0123456789\x1b[0m",
    "é_api_key=sk-CANARYdash0123456789",
    "2fa_secret=sk-CANARYdash0123456789",
])
def test_redact_masks_a_key_after_a_dash_colour_code_letter_or_digit(text):
    from surogate_eval.utils.redact import error_text, redact_text

    assert "CANARYdash" not in redact_text(text), redact_text(text)
    assert "CANARYdash" not in error_text(RuntimeError(text))


def _masked_record(msg, args):
    return logging.getLogRecordFactory()("x", logging.INFO, __file__, 1, msg, args, None)


def test_a_pickled_record_carries_no_key_and_unpicklable_args_still_pickle():
    """SocketHandler and QueueHandler pickle the record's __dict__ after
    replacing msg and args with the formatted message; a cache holding the
    original args put the raw key back in, and made unpicklable args fatal."""
    import logging.handlers
    import pickle
    import threading
    from surogate_eval.utils.redact import register_secrets

    canary = "sk-CANARYpickled0123456789"
    register_secrets({"api_key": canary})
    handler = logging.handlers.SocketHandler("127.0.0.1", 9)

    record = _masked_record("upstream said %s", (canary,))
    record.getMessage()
    payload = handler.makePickle(record)
    assert canary.encode() not in payload
    assert pickle.loads(payload[4:])["msg"] == "upstream said ***"

    lock = threading.Lock()
    record = _masked_record("holding %s", (lock,))
    record.getMessage()
    assert handler.makePickle(record)


@pytest.mark.parametrize("args, mutate, before, after", [
    ({"k": "one"}, lambda a: a.__setitem__("k", "two"), "value one", "value two"),
    ([["one"]], lambda a: a[0].__setitem__(0, "two"), "value ['one']", "value ['two']"),
])
def test_a_message_rewritten_in_place_is_not_served_from_the_cache(args, mutate, before, after):
    msg = "value %(k)s" if isinstance(args, dict) else "value %s"
    record = _masked_record(msg, (args,) if isinstance(args, dict) else tuple(args))
    assert record.getMessage() == before
    mutate(record.args)
    assert record.getMessage() == after
