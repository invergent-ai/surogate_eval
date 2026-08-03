import pytest

from surogate_eval.errors import (
    BenchmarkSchemaError,
    ConfigError,
    EvalError,
    JudgeError,
    JudgeParseError,
    JudgeUnavailableError,
    TargetUnhealthyError,
)


@pytest.mark.parametrize(
    "cls",
    [ConfigError, TargetUnhealthyError, JudgeError,
     JudgeUnavailableError, JudgeParseError, BenchmarkSchemaError],
)
def test_every_error_is_an_eval_error(cls):
    assert issubclass(cls, EvalError)


@pytest.mark.parametrize("cls", [JudgeUnavailableError, JudgeParseError])
def test_judge_errors_share_a_base(cls):
    """Catch sites catch JudgeError to handle both judge failure modes."""
    assert issubclass(cls, JudgeError)


def test_config_error_is_not_a_judge_error():
    """A bad config must not be swallowed by judge-failure handling."""
    assert not issubclass(ConfigError, JudgeError)
