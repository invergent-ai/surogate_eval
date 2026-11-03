"""Keep API keys out of the runner's logs.

The pod's stdout becomes the run's logs in ops, so a key in any log line is a
key in the run's logs. Three parts:

- ``redact(obj)`` masks secret fields by name in a copy of a config, a model
  block or library arguments. The runner's own log sites that print such a
  structure go through it, and ``error_text`` applies it to exception text
  stored in results.
- ``register_secrets(obj)`` remembers the secret values of a config where
  keys enter the process (config load, target creation; judges are
  targets).
- ``install_log_masking()``, called by the runner's entry points and never
  on import, masks every line a handler writes, from any logger, the
  libraries' included (evalscope logs its whole TaskConfig, judge key in
  it): remembered values, plus a narrow set of unmistakable credential forms.
  It also routes uncaught exceptions through that logger and remembers the
  provider keys in the environment.
"""

import base64
import json
import logging
import os
import re
import sys
import threading
import traceback
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

# One pass over a string finds, leftmost first, any of:
# - an Authorization value, whatever its scheme (Bearer, Basic, Token,
#   Digest, ...): the whole value goes, scheme included;
# - a bare "Bearer <credential>", where the credential looks like one (8 or
#   more characters with a digit), so "the bearer of" in prose survives;
# - key=value, key: value, 'key': 'value', "key": "value", or a URL query
#   parameter. Only the key and separator are consumed; the value is looked
#   at ahead. Whether it is masked is decided by is_secret_key, so strings and
#   dicts follow one rule, and a value that is not a secret is scanned on in
#   turn, which is how "url: https://h/v1?api_key=..." is reached. A key
#   starts only where a word does (linear time on long runs), may itself
#   start with dashes, digits or any letter (--api-key, 2fa_secret,
#   é_api_key), and a colour code in front of it is stepped over.
_NAMED_RE = re.compile(
    r"""(?P<auth>(?i:['"]?(?:proxy[_-])?authorization['"]?\s*[=:]\s*['"]?))(?P<aval>[^'"\n,}]+)"""
    r"""|(?P<bearer>(?i:\bbearer\s+))(?P<bval>(?=[^\s,'"}]*\d)[^\s,'"}]{8,})"""
    r"""|(?<![\w-])(?:\x1b\[[0-9;]*m)?(?P<q>['"]?)(?P<key>[\w-]+)(?P=q)(?P<sep>\s*[=:]\s*)"""
    r"""(?=(?P<vq>['"])(?P<qval>(?:(?!(?P=vq))[^\n])*)(?P=vq)|(?P<val>[^\s,'"}\])&;]+))"""
)

# The pass over every written line is narrower than ``redact``: a log line is
# mostly prose and metrics, so it masks only forms that are unmistakably a
# credential. Generic ``*_token``, a bare ``token`` or ``secret`` are left
# alone (time_to_first_token, prompt text). Values never take an ESC or "["
# so the colour reset after a value survives. A key only starts where a word
# does: without that left edge every position inside a long unbroken run
# (a base64 blob) rescanned the rest of it, and a 20k-character line never
# finished.
_CREDENTIAL_NAMES = (
    r"(?:x[_-])?api[_-]?key|access[_-]token|auth[_-]token|refresh[_-]token"
    r"|api[_-]token|hf[_-]token|github[_-]token|notion[_-]token|bearer[_-]token"
    r"|client[_-]secret|secret[_-]key|secret[_-]access[_-]key|private[_-]key|password|passwd"
)
_LINE_RE = re.compile(
    r"""(?P<auth>(?i:['"]?(?:proxy[_-])?authorization['"]?\s*[=:]\s*['"]?))(?P<aval>[^'"\n,}\x1b]+)"""
    r"""|(?P<bearer>(?i:\bbearer\s+))(?P<bval>(?=[^\s,'"}\x1b\[]*\d)[^\s,'"}\x1b\[]{8,})"""
    r"""|(?<![A-Za-z0-9_-])(?P<q>['"]?)(?P<key>(?i:[A-Za-z0-9_-]*?(?:""" + _CREDENTIAL_NAMES + r""")))(?P=q)"""
    r"""(?P<sep>\s*[=:]\s*)(?:(?P<vq>['"])(?P<qval>(?:(?!(?P=vq))[^\n\x1b])*)(?P=vq)"""
    r"""|(?P<val>[^\s,'"}\])\[&;\x1b]+))"""
)

_secrets: set[str] = set()
# The variable a child process reads the parent's remembered values from,
# once install_log_masking() has run in the parent. The parent's config is
# never loaded in a multiprocessing spawn child, so a key registered from it
# (an inline judge key) would otherwise reach the child's logs unmasked. The
# encoded value is itself masked, in case something logs the environment.
_SHARED_ENV = "SUROGATE_EVAL_LOG_MASK_SECRET"
_share_with_children = False
_shared_blob: str | None = None
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


def _mask_names(text: str) -> str:
    out, pos = [], 0
    while match := _NAMED_RE.search(text, pos):
        if match["auth"] is not None:
            value = "aval"
        elif match["bearer"] is not None:
            value = "bval"
        elif is_secret_key(match["key"]):
            value = "qval" if match["vq"] else "val"
        else:
            out.append(text[pos:match.end("sep")])
            pos = match.end("sep")
            continue
        out.append(text[pos:match.start(value)] + MASK)
        pos = match.end(value)
    out.append(text[pos:])
    return "".join(out)


def redact_text(text: str) -> str:
    """``text`` with secret fields masked by name and remembered values masked."""
    return scrub(_mask_names(text))


def _mask_line_match(match: re.Match) -> str:
    if match["auth"] is not None:
        return match["auth"] + MASK
    if match["bearer"] is not None:
        return match["bearer"] + MASK
    # "password: hunter2" is how prompt prose reads; a config or header
    # writes it as password=..., quotes the key, prefixes it (db_password)
    # or capitalises it (DB_PASSWORD), and all of those are masked.
    if (match["key"] == "password" and not match["q"]
            and ":" in match["sep"] and not match["vq"]):
        return match[0]
    quote = match["vq"] or ""
    return f"{match['q']}{match['key']}{match['q']}{match['sep']}{quote}{MASK}{quote}"


def mask_line(text: str) -> str:
    """A written log line with credential forms and remembered values masked."""
    return scrub(_LINE_RE.sub(_mask_line_match, text))


def error_text(exc: BaseException) -> str:
    """``str(exc)`` fit to store in a result ops shows: keys masked."""
    return redact_text(str(exc))


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
        return redact_text(obj)
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
    register_secrets({k: v for k, v in os.environ.items() if k != _SHARED_ENV})


def _remember(value: str) -> None:
    value = value.strip()
    _scheme, _, credential = value.partition(" ")
    if credential:  # "Bearer x", "Basic x": the credential alone shows up too
        _remember(credential)
    if len(value) >= _MIN_SECRET_LEN and value != "EMPTY" and value not in _secrets:
        _secrets.add(value)
        _rebuild()


def _rebuild() -> None:
    global _secrets_re, _shared_blob
    patterns = set(_secrets)
    if _share_with_children:
        _shared_blob = base64.b64encode(json.dumps(sorted(_secrets)).encode()).decode()
        os.environ[_SHARED_ENV] = _shared_blob
        patterns.add(_shared_blob)
    if patterns:
        _secrets_re = re.compile("|".join(
            re.escape(s) for s in sorted(patterns, key=len, reverse=True)
        ))


def _immutable_args(args) -> bool:
    return args is None or (isinstance(args, tuple) and all(
        isinstance(a, (str, bytes, int, float, bool, type(None))) for a in args
    ))


class _MaskedRecord(logging.LogRecord):
    """A record masked when it is written, not when it is made.

    Every formatter reads the message through ``getMessage`` and caches the
    traceback in ``exc_text``, so masking both here covers every handler,
    including ones a library attaches later to a logger that does not
    propagate (evalscope's), and a record no handler writes costs nothing.
    """

    def getMessage(self) -> str:
        # Masked once, however many handlers write the record. The cache
        # keeps only the masked string and id() markers: a pickling handler
        # (SocketHandler, QueueHandler) copies __dict__, and the original
        # args there would put the raw key back, or fail to pickle. A filter
        # that replaces msg or args changes the markers; args that could be
        # rewritten in place (a mapping, a list, any object) are not cached.
        marker = (id(self.msg), id(self.args))
        cached = self.__dict__.get("_masked_message")
        if cached is not None and cached[0] == marker:
            return cached[1]
        masked = mask_line(super().getMessage())
        if isinstance(self.msg, str) and _immutable_args(self.args):
            self.__dict__["_masked_message"] = (marker, masked)
        return masked

    @property
    def exc_text(self):
        return self.__dict__.get("_masked_exc_text")

    @exc_text.setter
    def exc_text(self, value):
        self.__dict__["_masked_exc_text"] = mask_line(value) if value else value


def _log_uncaught(exc_type, exc, tb, where: str = "Uncaught exception") -> None:
    """Report an exception nothing caught through the masked runner logger.

    Python's default hook prints the traceback raw to stderr, which is the
    run's log in ops.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    try:
        from surogate_eval.utils.logger import get_logger

        get_logger().critical(where, exc_info=(exc_type, exc, tb))
    except Exception:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(f"{where}\n{mask_line(text)}")


def _log_uncaught_in_thread(args) -> None:
    if args.exc_type is SystemExit:  # as threading's own hook does
        return
    name = args.thread.name if args.thread is not None else threading.get_ident()
    _log_uncaught(args.exc_type, args.exc_value, args.exc_traceback,
                  where=f"Exception in thread {name}")


def install_log_masking() -> None:
    """Mask every written log line and report uncaught exceptions masked.

    For the runner's entry points only: it replaces process-wide hooks, which
    importing a module must not do to whoever imported it. It also hands
    every value it remembers, before or after this call, to multiprocessing
    spawn children through the environment, and takes over what a parent
    handed down.
    """
    global _share_with_children
    base = logging.getLogRecordFactory()
    if not getattr(base, "_masks_records", False):
        def factory(*args, **kwargs):
            record = base(*args, **kwargs)
            record.__class__ = _MaskedRecord
            return record

        factory._masks_records = True
        logging.setLogRecordFactory(factory)
    sys.excepthook = _log_uncaught
    threading.excepthook = _log_uncaught_in_thread
    _share_with_children = True
    inherited = os.environ.get(_SHARED_ENV)
    if inherited:  # a spawn child: take over what its parent remembered
        try:
            for value in json.loads(base64.b64decode(inherited)):
                _remember(value)
        except ValueError:
            pass
    register_env_secrets()
    _rebuild()
