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
  Provider keys already in the environment at import are remembered too.
"""

import logging
import os
import re
from typing import Any

MASK = "***"

# Shorter values (and the "EMPTY" placeholder) are not remembered: masking
# them everywhere would eat ordinary words in unrelated log lines.
_MIN_SECRET_LEN = 8

_SECRET_SEGMENTS = {"secret", "password", "passwd", "authorization", "apikey"}
_SECRET_SUFFIXES = ("api_key", "access_key", "private_key")
# A field named ``*_token`` is a credential (access_token, hf_token, ...)
# unless the word before it makes it a vocabulary token or a count.
_NOT_CREDENTIAL_TOKENS = {
    "eos", "bos", "pad", "unk", "sep", "cls", "mask", "start", "end", "stop",
    "eot", "max", "min", "num", "special", "image", "video", "audio", "vision",
    "think", "default",
}

# Any Authorization header value, whatever its scheme (Bearer, Basic,
# Token, Digest, ...): the whole value goes, scheme included.
_AUTH_HEADER_RE = re.compile(
    r"""(?i)(['"]?(?:proxy[_-])?authorization['"]?\s*[=:]\s*['"]?)[^'"\n,}]+"""
)
# key=value, key: value, 'key': 'value', "key": "value". Whether the key is
# a secret is decided by is_secret_key, so strings and dicts follow one rule.
_KEY_VALUE_RE = re.compile(
    r"""(?P<q>['"]?)(?P<key>[A-Za-z_][\w-]*)(?P=q)(?P<sep>\s*[=:]\s*['"]?)(?P<val>[^\s,'"}\])]+)"""
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[^,\s'\"}]+")

_secrets: set[str] = set()
# One alternation over every remembered value, longest first so a secret
# that contains another is masked whole. Rebuilt only when a value is added.
_secrets_re: re.Pattern | None = None


def is_secret_key(key: Any) -> bool:
    """True for a field name that holds a credential (api_key, apiKey,
    Authorization, auth_token, hf_token, client_secret, ...). ``max_tokens``,
    ``eos_token``, ``token_budget`` and ``tokenizer`` are not."""
    if not isinstance(key, str):
        return False
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower().replace("-", "_")
    segments = snake.split("_")
    if segments[-1] == "token":
        return len(segments) == 1 or segments[-2] not in _NOT_CREDENTIAL_TOKENS
    return snake.endswith(_SECRET_SUFFIXES) or bool(_SECRET_SEGMENTS & set(segments))


def scrub(text: str) -> str:
    """``text`` with every remembered secret value replaced by the mask."""
    pattern = _secrets_re
    return pattern.sub(MASK, text) if pattern else text


def _mask_key_value(match: re.Match) -> str:
    if not is_secret_key(match["key"]):
        return match[0]
    return f"{match['q']}{match['key']}{match['q']}{match['sep']}{MASK}"


def redact(obj: Any) -> Any:
    """A copy of ``obj`` safe to log: secret fields masked, the original untouched.

    Dicts and lists are walked; strings lose ``key=value`` credentials (also
    in repr and JSON form), Authorization values, ``Bearer`` tokens and
    remembered values. Anything else is returned as is.
    """
    if isinstance(obj, dict):
        return {k: MASK if is_secret_key(k) and v else redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact(v) for v in obj)
    if isinstance(obj, str):
        text = _AUTH_HEADER_RE.sub(rf"\g<1>{MASK}", obj)
        text = _BEARER_RE.sub(rf"\g<1>{MASK}", text)
        text = _KEY_VALUE_RE.sub(_mask_key_value, text)
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


def register_env_secrets() -> None:
    """Remember credentials that live only in the environment.

    The runner and its libraries read provider keys from it (OPENAI_API_KEY
    for lm-eval and deepeval/deepteam provider models, the LakeFS and hub
    ``*_SECRET_ACCESS_KEY`` pairs, GITHUB_TOKEN / NOTION_TOKEN for MCPMark,
    HF_TOKEN, and the other ``*_API_KEY`` variables litellm and deepeval
    accept). Any variable whose name is a secret field name counts, so a
    provider added later is covered without a list to keep in step.
    """
    register_secrets(dict(os.environ))


def _remember(value: str) -> None:
    global _secrets_re
    value = value.strip()
    _scheme, _, credential = value.partition(" ")
    if credential:  # "Bearer x", "Basic x": the credential alone shows up too
        _remember(credential)
    if len(value) >= _MIN_SECRET_LEN and value != "EMPTY" and value not in _secrets:
        _secrets.add(value)
        _secrets_re = re.compile("|".join(
            re.escape(s) for s in sorted(_secrets, key=len, reverse=True)
        ))


_base_factory = logging.getLogRecordFactory()


def _scrubbing_factory(*args, **kwargs) -> logging.LogRecord:
    record = _base_factory(*args, **kwargs)
    if _secrets_re is None:
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
register_env_secrets()
