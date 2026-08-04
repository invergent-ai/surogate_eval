"""Typed errors for the eval runner.

The runner must fail closed: a broken judge, an unreachable target or an
unresolved config variable has to surface as an error, never as a score.
These types let each catch site tell those cases apart from a genuine bug
in our own code.
"""


class EvalError(Exception):
    """Base for every error the eval runner raises deliberately."""


class ConfigError(EvalError):
    """Config is unusable. Raised at load, before any evaluation runs."""


class TargetUnhealthyError(EvalError):
    """A target failed its health check and cannot be evaluated."""


class JudgeError(EvalError):
    """Base for judge failures. Catch this to handle both modes below."""


class JudgeUnavailableError(JudgeError):
    """The judge could not be reached, or returned no content."""


class JudgeParseError(JudgeError):
    """The judge returned content that could not be parsed into a schema."""


class BenchmarkSchemaError(EvalError):
    """A backend's result payload did not have the shape we know how to read.

    Raised rather than defaulted past, because the alternative is publishing a
    number we did not measure. Not retryable: an upstream schema change will
    not resolve on the next attempt.
    """


class MatchTimeout(EvalError):
    """A user-supplied pattern took longer than its budget on one row.

    Raised rather than returning "no match", because a pattern we abandoned
    tells us nothing about whether the answer was right. The caller records
    the row as unmeasured.
    """


class UnscorableRow(EvalError):
    """A row carries nothing to score against, so no verdict is available.

    Raised rather than returning a verdict, because both verdicts are wrong:
    "correct" is the fail-open case a blank answer key would otherwise
    produce under ``contains`` (every output contains the empty string), and
    "incorrect" records a row nobody measured as one the model got wrong.
    The caller records the row as unmeasured.
    """
