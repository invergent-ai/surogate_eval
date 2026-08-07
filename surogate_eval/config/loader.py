import json
import os
import re
from typing import Any
import yaml

from surogate_eval.config.eval_config import EvalConfig
from surogate_eval.errors import ConfigError
from surogate_eval.utils.dict import DictDefault
from surogate_eval.utils.logger import get_logger

logger = get_logger()

type SurogateConfig = EvalConfig

def load_config(config_cls: SurogateConfig, path: str) -> SurogateConfig:
    with open(path, encoding="utf-8") as file:
        cfg_dict = yaml.safe_load(file)

        # Expand environment variables. An unresolved ${VAR} used to be
        # left in place as a literal, which then passed the target health
        # check as a non-empty credential and 401'd on every request
        # (E-RUN-2). Fail here instead, and report every missing name at
        # once so a user fixes them in one pass.
        missing: set[str] = set()
        cfg_dict = _expand_env_vars(cfg_dict, missing)
        if missing:
            raise ConfigError(
                f"Unresolved environment variable(s) in {path}: "
                f"{', '.join(sorted(missing))}. "
                "Export them, or remove the ${...} reference."
            )
        cfg: DictDefault = DictDefault(cfg_dict)

    config = config_cls(cfg)
    cfg.config_path = path

    cfg_to_log = {
        k: v for k, v in cfg.items() if v is not None
    }

    logger.debug(
        "config:\n%s",
        json.dumps(cfg_to_log, indent=2, default=str, sort_keys=True),
    )

    return config

#: Fields holding text the user wrote, where `${...}` is ordinary template
#: syntax rather than a reference to the environment. Expansion skips these;
#: everywhere else it applies as before.
#:
#: A denylist rather than an allowlist of credential fields, and the
#: direction matters more than the contents. Both lists will be incomplete
#: eventually, so the question is how each one fails. Miss a prose field here
#: and the run stops at config load naming the variable, which is loud and
#: obvious. Miss a credential field in an allowlist and the literal `${VAR}`
#: ships to the provider and 401s a minute into the run -- silent, and
#: exactly the E-RUN-2 failure this check exists to prevent.
#:
#: An allowlist was tried first, keyed on `_key`/`_url` suffixes because
#: every reference in the repo's configs happens to use one. The schema
#: already disagreed: `endpoint`, `health_endpoint` and `headers` hold
#: credentials and URLs and match no suffix, and `headers` is merged
#: verbatim into every request, so `Authorization: Bearer ${TOKEN}` would
#: have silently shipped the literal.
_PROSE_FIELDS = frozenset({
    'judge_criteria',
    'system_prompt',
    'prompt_template',
    'pattern',        # a matcher's regex, inside the `matcher` block
    'description',
    'comment',        # "additional comments about this target"; read by nothing
    'purpose',        # the red-team persona, fed to the attack simulator
})


def _expand_env_vars(obj: Any, missing: set[str], expand: bool = True) -> Any:
    """
    Recursively expand environment variables in config.
    Supports ${VAR_NAME} syntax.

    Everywhere except the free-text fields in ``_PROSE_FIELDS``, where a
    `${...}` is content: passed through untouched, neither expanded nor
    reported missing. Untouched matters as much as unreported -- rewriting a
    prompt because the pod happens to export a matching name would score the
    benchmark against text the user never wrote.

    Names that cannot be resolved are collected into *missing* and left
    untouched; the caller raises once it has the full set.
    """
    if isinstance(obj, dict):
        # `expand and ...`, so a prose verdict carries down into a nested
        # block instead of being recomputed away at the next level. Without
        # the conjunction `{"description": {"note": "${X}"}}` expands, which
        # contradicts what this function promises about prose.
        return {
            k: _expand_env_vars(v, missing, expand and k not in _PROSE_FIELDS)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        # A list has no key of its own, so it inherits the verdict of the
        # key it hangs off rather than defaulting back to "expand".
        #
        # `stop_sequences` is the list a reader will think of, and it is
        # deliberately NOT
        # prose: leaving it out means a `${...}` there fails loudly at load,
        # which is the direction this whole check errs in.
        return [_expand_env_vars(item, missing, expand) for item in obj]
    elif isinstance(obj, str) and expand:
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'

        def replace_env_var(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                missing.add(var_name)
                return match.group(0)
            logger.debug(f"Expanded ${{{var_name}}} (length: {len(value)})")
            return value

        return re.sub(pattern, replace_env_var, obj)
    else:
        return obj



