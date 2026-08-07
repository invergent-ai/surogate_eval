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


# --- user prose is not an environment variable -------------------------

PROSE_CONFIG = """\
project:
  name: test
targets:
  - name: t1
    type: llm
    provider: openai
    model: gpt-4
    api_key: sk-literal
    evaluations:
      - name: run
        benchmarks:
          - name: custom-support-qa
            backend: custom_eval
            source: hub://p/ds/main
            eval_type: judge
            judge_criteria: "Score 1 if the reply greets the user by ${name}."
            system_prompt: "Answer as ${persona} would."
"""


def test_a_placeholder_in_judge_criteria_is_left_alone(tmp_path, monkeypatch):
    """`${...}` is ordinary template syntax in a prompt. Expanding the whole
    document treated a sentence the user typed in the Studio as a reference
    to a pod environment variable, and failed the entire run at config load
    with a message naming a variable they never mentioned."""
    monkeypatch.delenv("name", raising=False)
    monkeypatch.delenv("persona", raising=False)

    config = load_config(EvalConfig, write(tmp_path, PROSE_CONFIG))

    benchmark = config.targets[0].evaluations[0].benchmarks[0]
    assert benchmark.judge_criteria == (
        "Score 1 if the reply greets the user by ${name}."
    )
    assert benchmark.system_prompt == "Answer as ${persona} would."


def test_prose_is_not_expanded_even_when_the_variable_exists(tmp_path, monkeypatch):
    """Silently rewriting a prompt because the pod happens to export a
    matching name is worse than failing: the benchmark would score against
    text the user never wrote."""
    monkeypatch.setenv("name", "SHOULD-NOT-APPEAR")

    config = load_config(EvalConfig, write(tmp_path, PROSE_CONFIG))

    criteria = config.targets[0].evaluations[0].benchmarks[0].judge_criteria
    assert "SHOULD-NOT-APPEAR" not in criteria
    assert "${name}" in criteria


def test_a_missing_credential_still_raises(tmp_path, monkeypatch):
    """The allow direction: narrowing where expansion happens must not lose
    the protection it was added for (E-RUN-2, an unresolved key counting as
    a valid credential)."""
    monkeypatch.delenv("SPIKE_MISSING_KEY", raising=False)
    monkeypatch.delenv("SPIKE_OTHER_MISSING", raising=False)
    with pytest.raises(ConfigError):
        load_config(EvalConfig, write(tmp_path, CONFIG))
