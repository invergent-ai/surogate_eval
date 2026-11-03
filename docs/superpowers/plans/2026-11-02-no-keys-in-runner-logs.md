# Plan: the runner never writes an API key to its logs

Spec: `docs/superpowers/specs/2026-11-02-no-keys-in-runner-logs-design.md`

1. Tests first, in `tests/test_no_keys_in_logs.py`, with a canary target key and a canary judge
   key. Capture every log record (root, `surogate_eval`, `lm-eval`, `evalscope`):
   - lm-eval backend: `simple_evaluate` mocked; the canary appears in no log line and not in
     `model_args`, while `OPENAI_API_KEY` holds the canary during the call and is restored after.
   - lm-eval itself: the real `evaluator.simple_evaluate` logging line, driven with the real
     `model_args` the backend builds (model construction stubbed), carries no canary.
   - evalscope: a tau bench `TaskConfig` built by the backend with the judge canary, logged the
     way evalscope logs it, carries no canary, while `extra_params.api_key` still holds it.
   - `load_config` debug dump, `TargetFactory` failure log, metric registry failure log.
   - `redact` unit cases: nested keys, header case, `key=value` and `Bearer` strings.
   Run each against unchanged code and see it fail.
2. Add `surogate_eval/utils/redact.py` (redact, register_secrets, scrub, record factory).
3. Wire it in: loader, factory, registry, red_team, lm-eval backend (env var, no `api_key` in
   `model_args`).
4. Prove each test red with its fix removed; restore.
5. Full suite, baseline 488 passed.
6. Review the diff against the spec.
