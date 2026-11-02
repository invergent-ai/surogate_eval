"""Keep API keys out of the runner's logs.

The pod's stdout becomes the run's logs in ops, so a key in any log line is a
key in the run's logs. Two layers:

- ``redact(obj)`` masks secret fields by name. Every runner log site that
  prints a config, a model block or library arguments goes through it.
- ``register_secrets(obj)`` remembers the secret values of a config where
  keys enter the process (config load, target creation; judges are
  targets). A log record factory then masks those values in every record,
  from any logger, which is the only way to reach lines a library logs
  itself (evalscope logs its whole TaskConfig, judge key included).
"""

import logging
import re
from typing import Any

MASK = "***"

# Shorter values (and the "EMPTY" placeholder) are not remembered: masking
# them everywhere would eat ordinary words in unrelated log lines.
_MIN_SECRET_LEN = 8

_SECRET_SEGMENTS = {"secret", "password", "passwd", "token", "authorization", "apikey"}
_SECRET_SUFFIXES = ("api_key", "access_key", "private_key")

_KEY_VALUE_RE = re.compile(
    r"(?i)(\b[\w-]*(?:api[_-]?key|token|secret|password|authorization)\b\s*[=:]\s*)[^,\s'\"}]+"
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[^,\s'\"}]+")

_secrets: set[str] = set()


def is_secret_key(key: Any) -> bool:
    """True for a field name that holds a credential (api_key, apiKey,
    Authorization, auth_token, client_secret, ...). ``max_tokens`` and
    ``tokenizer`` are not."""
    if not isinstance(key, str):
        return False
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower().replace("-", "_")
    return snake.endswith(_SECRET_SUFFIXES) or bool(_SECRET_SEGMENTS & set(snake.split("_")))


def scrub(text: str) -> str:
    """``text`` with every remembered secret value replaced by the mask."""
    for secret in sorted(tuple(_secrets), key=len, reverse=True):
        text = text.replace(secret, MASK)
    return text


def redact(obj: Any) -> Any:
    """A copy of ``obj`` safe to log: secret fields masked, the original untouched.

    Dicts and lists are walked; strings lose ``key=value`` credentials,
    ``Bearer`` tokens and remembered values. Anything else is returned as is.
    """
    if isinstance(obj, dict):
        return {k: MASK if is_secret_key(k) and v else redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact(v) for v in obj)
    if isinstance(obj, str):
        text = _BEARER_RE.sub(rf"\g<1>{MASK}", obj)
        text = _KEY_VALUE_RE.sub(rf"\g<1>{MASK}", text)
        return scrub(text)
    return obj


def register_secrets(obj: Any) -> None:
    """Remember the values of every secret field in ``obj`` for the log net."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if is_secret_key(k) and isinstance(v, str):
                _remember(v)
            else:
                register_secrets(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            register_secrets(v)


def _remember(value: str) -> None:
    value = value.strip()
    if value.lower().startswith("bearer "):
        _remember(value[7:])
    if len(value) >= _MIN_SECRET_LEN and value != "EMPTY":
        _secrets.add(value)


_base_factory = logging.getLogRecordFactory()


def _scrubbing_factory(*args, **kwargs) -> logging.LogRecord:
    record = _base_factory(*args, **kwargs)
    if not _secrets:
        return record
    try:
        message = record.getMessage()
    except Exception:
        return record  # logging reports the bad format itself
    clean = scrub(message)
    if clean != message:
        record.msg, record.args = clean, None
    if record.exc_info:
        # Formatter.format uses a preset exc_text instead of formatting the
        # traceback again, so the masked copy is what every handler prints.
        record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
    return record


logging.setLogRecordFactory(_scrubbing_factory)
