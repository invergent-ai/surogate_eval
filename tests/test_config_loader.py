import pytest

from surogate_eval.config.eval_config import EvalConfig
from surogate_eval.config.loader import load_config
from surogate_eval.errors import ConfigError

CONFIG = """\
project:
  name: test
targets:
  - name: t1
    type: llm
    provider: openai
    model: gpt-4
    api_key: ${SPIKE_MISSING_KEY}
    judge_key: ${SPIKE_OTHER_MISSING}
"""


def write(tmp_path, text):
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_unresolved_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("SPIKE_MISSING_KEY", raising=False)
    monkeypatch.delenv("SPIKE_OTHER_MISSING", raising=False)
    with pytest.raises(ConfigError):
        load_config(EvalConfig, write(tmp_path, CONFIG))


def test_error_lists_every_missing_var(tmp_path, monkeypatch):
    """One run per typo is a bad loop. Report them all at once."""
    monkeypatch.delenv("SPIKE_MISSING_KEY", raising=False)
    monkeypatch.delenv("SPIKE_OTHER_MISSING", raising=False)
    with pytest.raises(ConfigError) as exc:
        load_config(EvalConfig, write(tmp_path, CONFIG))
    message = str(exc.value)
    assert "SPIKE_MISSING_KEY" in message
    assert "SPIKE_OTHER_MISSING" in message


def test_resolved_vars_load_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("SPIKE_MISSING_KEY", "sk-real")
    monkeypatch.setenv("SPIKE_OTHER_MISSING", "sk-other")
    config = load_config(EvalConfig, write(tmp_path, CONFIG))
    assert config.targets[0].api_key == "sk-real"


def test_config_without_vars_is_unaffected(tmp_path):
    text = CONFIG.replace("${SPIKE_MISSING_KEY}", "sk-literal").replace(
        "${SPIKE_OTHER_MISSING}", "sk-literal2"
    )
    config = load_config(EvalConfig, write(tmp_path, text))
    assert config.targets[0].api_key == "sk-literal"
