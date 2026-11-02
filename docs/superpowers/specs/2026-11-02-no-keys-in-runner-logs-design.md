# Design: the runner never writes an API key to its logs

**Date:** 2026-11-02
**Repo:** `surogate_eval`

## Problem

Ops starts an eval pod with the target's API key (a proxy model's upstream key, or a minted
`sk-agent-...` key) and any judge keys in the pod config, and the pod's stdout becomes the
run's logs. Several paths put those keys into the log stream.

### Leak sites

| Site | Whose log | What leaks |
|---|---|---|
| `benchmarks/backends/lm_eval_backend.py:314` builds `model_args` with `api_key=<key>`; `:345` logs `lm-eval args: {eval_args}` | runner | target key, every lm-eval run against a local or proxy endpoint |
| same `model_args`, logged by `lm_eval/evaluator.py:242` (`Initializing local-completions model, with arguments: ...`) | lm-eval | target key |
| `benchmarks/backends/evalscope_backend.py:1337` puts the judge key in `dataset_args[tau_bench / tau2_bench].extra_params.api_key`; evalscope `run.py:144` logs the whole `TaskConfig`, whose `to_dict` drops only the top level `api_key` and `judge_model_args.api_key` | evalscope | judge key, tau bench runs |
| `config/loader.py:42` debug dump of the whole loaded config | runner (DEBUG) | every key in the YAML |
| `targets/factory.py:71` `Config: {config}` when a target fails to build | runner | that target's key |
| `metrics/registry.py:76` `Config: {config}` when a metric fails to build | runner (DEBUG) | an inline judge key, if any |
| `security/red_team.py:173,175` `simulator_model value` / `evaluation_model value` | runner | a raw dict model block, if one is passed through unresolved |

Checked and safe: the evalscope top level `api_key` and `judge_model_args.api_key` (popped by
`TaskConfig.to_dict`); the runner's own evalscope logs (model name only); `APIModelTarget`
request logging (URL, model, payload; no headers); `targets/base.py` `__str__`.

### A second finding in the same line

lm-eval 0.4.9.2 (the version `uv.lock` pins) does not read `api_key` from `model_args`.
`LocalCompletionsAPI` swallows it in `**kwargs` and builds its header from `OPENAI_API_KEY`
only. So the key in `model_args` was logged and then thrown away: lm-eval sent
`Authorization: Bearer ` (or whatever `OPENAI_API_KEY` the pod happened to have).

## Design

One module, `surogate_eval/utils/redact.py`:

- `redact(obj)`: a copy of a dict / list / string with every value under a secret looking key
  (`api_key`, `apiKey`, `authorization`, anything containing `token`, `secret`, `password`)
  replaced by `***`, plus `key=value` pairs and `Bearer ...` inside strings. Used at every
  runner log site above that prints a structure.
- `register_secrets(obj)`: walks a config and remembers the values under those keys.
  Called where keys enter the process: `load_config` and `TargetFactory.create_target`
  (judges are targets, so both target and judge keys are covered).
- A `logging` record factory, installed on import, that replaces every remembered value in a
  record's message and formatted exception text with `***`. It runs when a record is created,
  so it applies to every logger and handler, including evalscope's and lm-eval's own, which the
  runner cannot otherwise reach. Values shorter than 8 characters and `EMPTY` are not
  remembered, so ordinary words are never masked.

lm-eval local-completions: drop `api_key` from `model_args` and hand the key to lm-eval the way
it actually reads it, `OPENAI_API_KEY`, set only for the duration of `simple_evaluate` and
restored afterwards, so a later deepeval / deepteam provider-model judge never picks up a target
key. Benchmarks run one at a time, so the scoped variable is not shared with a concurrent run.

## Behaviour change

lm-eval runs against a key protected endpoint now send the target key instead of an empty
bearer. That is the intended behaviour; before, those runs authenticated only if the pod's
environment carried a matching `OPENAI_API_KEY`. Scores of runs that authenticated before are
unchanged.

## Out of scope

- Output a library writes with `print` rather than `logging` (none found carrying a key).
- evalscope's `task_config.yaml` in the run's work dir still carries the tau bench judge key;
  it is a file in the pod, not the log stream.
