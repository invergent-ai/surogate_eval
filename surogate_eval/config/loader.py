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

def _expand_env_vars(obj: Any, missing: set[str]) -> Any:
    """
    Recursively expand environment variables in config.
    Supports ${VAR_NAME} syntax.

    Names that cannot be resolved are collected into *missing* and left
    untouched; the caller raises once it has the full set.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v, missing) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item, missing) for item in obj]
    elif isinstance(obj, str):
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



