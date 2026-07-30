# Design: make the eval runner fail closed

**Date:** 2026-07-30
**Repo:** `surogate_eval`
**Ships as:** PR 6 in the evaluations stream (see `Misc/Training-Dataset-Eval/eval-pr-order.md`)
**Findings covered:** E-RUN-1, E-RUN-2, E-RUN-3

## Problem

When something breaks mid-run, the runner returns a confident-looking number instead of an
error. The eval finishes, reports scores, and exits 0. Nothing anywhere records that part of
the result is fabricated.

Three sites, one posture:

- **E-RUN-1.** Judge failures become `score=0.0`. A zero from "the model answered badly" and a
  zero from "the judge was down" are written identically.
- **E-RUN-2.** An unresolved `${VAR}` stays in the config as a literal string. The health check
  tests the API key with `bool(self.api_key)`, and a non-empty literal passes, so a target with
  no credential is declared healthy.
- **E-RUN-3.** Safety metrics fall back to keyword heuristics on judge parse failure.
  `_simple_bias_check` never reads the text at all: it returns `success=True` unconditionally.

These chain. Forget to export the key, the placeholder passes the health check, every request
401s, every response is empty, every judged metric scores 0, and safety metrics pass on
keywords. You get a clean-looking report for a run that never contacted the target model.

### Two corrections to the findings doc

Established by spike (deepeval 3.7.9 installed in a scratch venv, 2026-07-30):

1. **E-RUN-1's mechanism is misattributed.** The doc points at
   `models/deepeval_wrapper.py` returning a 0 score. In fact exceptions raised from
   `generate()` propagate cleanly out of `measure()` in both sync and async mode. The site that
   actually flattens failures to 0.0 is **our own adapter**, `metrics/adapters/deepeval_adapter.py:365-375`,
   whose blanket `except Exception` returns `score=0.0, success=False`. Fixing only the wrapper
   would have changed nothing for DeepEval-driven metrics.

2. **`_empty_schema` does not even produce a clean zero for GEval.** It builds a malformed
   `Steps` object, and the metric dies with `AttributeError: 'Steps' object has no attribute
   'steps'`, which the adapter then flattens to 0.0. Same visible outcome, different route. The
   consequence worth noting is that the blanket `except` is also hiding genuine code bugs as
   "the model scored zero".

A third structural fact that shapes the design: `metrics/safety.py` does **not** go through the
wrapper. It calls `self.judge_target.send_request(...)` directly, so we own that loop. Combined
with (1), we own every catch site in the system.

### The fail-open goes one level higher than recorded

`eval.py:86-93` writes a failed target into the results as `status: "unhealthy"` and `continue`s.
`run()` then reaches `logger.success("Surogate Eval completed")` and exits 0. So fixing the
health check alone would not fail the run. The run would complete with zero evaluations and ops
would ingest it as fine. The run-level outcome is a fourth touchpoint, not covered by any of the
three findings.

## Approach

One uniform mechanism: **raise a typed error at the source, catch it at the `MetricResult`
boundary, carry an errored status through aggregation, and let the run-level outcome set the
exit code.**

A ledger or other side-channel was considered and rejected: since we own every catch site, the
existing `MetricResult` funnel is sufficient, and a raise makes it structurally impossible to
record a failure and still return a score.

### 1. Error taxonomy

New module `surogate_eval/errors.py`:

```python
class EvalError(Exception): ...
class ConfigError(EvalError): ...             # unresolved ${VAR}, raised at load
class TargetUnhealthyError(EvalError): ...    # health check failed
class JudgeError(EvalError): ...              # base for judge failures
class JudgeUnavailableError(JudgeError): ...  # transport error / empty content
class JudgeParseError(JudgeError): ...        # got content, could not parse
```

Typed so catch sites can distinguish "the judge broke" from "our code broke" from "the config is
wrong". The last must fail before any evaluation work begins.

### 2. `MetricResult` carries status

```python
class MetricStatus(str, Enum):
    scored = "scored"
    errored = "errored"
```

`MetricResult.status: MetricStatus = MetricStatus.scored`, and `score` becomes
`Optional[float]`, set to `None` when errored. Nullable rather than 0.0 so it is structurally
impossible to average an error into a score. `to_dict()` emits `status`.

**Consumer impact:** ops reads benchmark-level scores via `_extract_scores`, not per-metric
scores, so the ops ingestion contract is unaffected by the nullable field.

### 3. `BatchMetricResult` aggregates over scored results only

`avg_score` and `success_rate` computed over `status is scored` only. A judge outage no longer
drags the score down. New fields on `to_dict()`: `scored_n`, `errored_n`, `error_rate`.

Score now means "of what we could actually measure", and the errored count plus the run outcome
are what tell you whether the run is trustworthy.

### 4. Run-level outcome and exit code

After `_process_targets`, compute an outcome block into the consolidated results:

```python
consolidated_results["outcome"] = {
    "status": "completed" | "failed",
    "reason": str | None,
    "scored": int,
    "errored": int,
    "error_rate": float,
    "max_error_rate": float,
}
```

`run()` returns an exit code; `cli/eval.py` passes it to `sys.exit`. The run fails (exit 1) when:

- zero targets are healthy (always, regardless of threshold), or
- any configured target did not complete its evaluations (health check failed, target creation
  failed, or the evaluation raised), or
- a healthy target produced no countable results at all, or
- `error_rate` exceeds `max_error_rate`.

`error_rate` is **run-wide**, not per-metric: `errored_n / (scored_n + errored_n)` summed across
every `MetricResult` produced by every metric on every target in the run.

**A run with no results is not covered by the zero-healthy-targets rule.** An earlier version of
this section claimed it was; that is false. The failures that produce no countable results
happen *after* a target has passed its health check: an evaluation crashing wholesale, a metric
raising, a benchmark or red-team or guardrails run failing, or a target with no evaluations
configured. In every one of those cases the target is healthy, so the zero-healthy-targets rule
never fires, and `total = 0` divides to `error_rate = 0.0` and reports "completed". Two rules
close this. First, any node carrying a failure status (`failed`, `error`, `validation_failed`,
`incompatible`, `unhealthy`) counts as one errored unit, so coarse failures above the metric
level become countable. Second, a healthy target that produced zero countable results fails the
run outright: "we measured nothing" is not a success.

`max_error_rate` is configurable at the top level of the eval config, **default 0.2**:

```yaml
project:
  name: my-eval
max_error_rate: 0.2   # optional; omit to accept the default
targets:
  ...
```

This is the signal ops already watches. The exit-code gate shipped in ops PR #308
(`eval_monitor.py:727-740`) marks a run failed on a non-zero exit, so no ops change is needed
for the run-level outcome to become visible.

### 5. Config loader fails hard on unresolved variables

`config/loader.py::_expand_env_vars` collects **all** unresolved `${VAR}` names and raises a
single `ConfigError` listing them together, so a user fixes every typo in one pass rather than
one run per variable.

**Verified safe for the ops path:** ops writes literal values into the eval YAML and passes
secrets as pod env vars (`core/compute/evaluate.py:823-867`), never `${VAR}` placeholders. Hard
failing therefore only affects hand-written local configs, which is exactly where the bug bites.

### 6. Health check stops being optimistic

In `targets/model.py`:

- Remove `bool(self.api_key)` as the OpenAI/Anthropic health test.
- Remove the "be optimistic if we have credentials" fallback (`:233-235`). A probe that cannot
  confirm the target is now **unhealthy**, not assumed fine.

With (5) in place the literal-placeholder case is already dead at config load. This is defence
in depth for the case where a credential is present but wrong.

### 7. An empty target response is an error, not a zero

`deepeval_adapter.py:200-207` returns `score=0.0` when `actual_output` is empty. This is the
same fail-open one layer out: the *target* failed rather than the judge, and the result is still
written as a score.

Rule: if the `TargetResponse` carries an `error`, or content is empty **and** an error is set,
the result is errored. A genuinely empty completion with no transport error stays a scored 0.0,
because an empty answer is a real (bad) answer. The distinction is `target_response.error`, not
emptiness alone.

### 8. safety.py and conversation.py fail closed

Delete `_simple_toxicity_check`, `_simple_bias_check` and `_simple_harm_check`. A judge transport
failure raises `JudgeUnavailableError` and a parse failure raises `JudgeParseError`; both surface
as an errored `MetricResult`. The existing outer `except Exception` returns errored rather than
`score=0.0`.

These heuristics are not a safety net. Toxicity was five keywords, harm was sixteen across four
categories, and bias was an unconditional pass.

`metrics/conversation.py` carries the identical pattern and gets the identical treatment:
`_simple_coherence_check`, `_simple_retention_check` and `_simple_turn_analysis` scored 0.7/0.75
with `success=True` from response length alone, so a judge outage produced a passing
conversation score.

## Error handling summary

| Condition | Before | After |
|---|---|---|
| Unresolved `${VAR}` | warning, literal kept | `ConfigError` at load, run never starts |
| Target unhealthy | recorded, run continues, exit 0 | recorded; exit 1 if no target healthy |
| Target request failed (`response.error`) | `score=0.0` | errored result |
| Target returned empty, no error | `score=0.0` | `score=0.0` (unchanged, a real bad answer) |
| Judge transport error | `score=0.0` | `JudgeUnavailableError` -> errored result |
| Judge unparseable | `score=0.0` or keyword heuristic | `JudgeParseError` -> errored result |
| Internal bug in metric | `score=0.0` | errored result, reason marked internal |
| Error rate over threshold | run completes, exit 0 | run fails, exit 1 |

## Testing

The repo has **no test infrastructure today**: no `tests/`, no pytest dependency. This PR
bootstraps it, per the E-X-1 policy that every eval-stream PR lands with the failing-then-passing
test for what it fixes.

- Add pytest to `pyproject.toml` and create `tests/`.
- All tests use fake targets. Nothing touches the network.

Coverage, one test per behaviour:

1. Loader raises `ConfigError` on an unresolved `${VAR}`, and the message lists every
   unresolved name, not just the first.
2. Health check reports unhealthy when the probe fails, and does not fall back to optimism.
3. Wrapper raises `JudgeUnavailableError` on target error and on empty content.
4. Wrapper raises `JudgeParseError` on unparseable content.
5. Adapter converts a `JudgeError` into an errored `MetricResult`, not `score=0.0`.
6. **Bias regression:** unparseable judge output never yields `success=True`.
7. `BatchMetricResult.avg_score` excludes errored results from the average.
8. A target response carrying an `error` yields an errored result; a genuinely empty completion
   with no error still yields a scored 0.0.
9. Run outcome: zero healthy targets exits 1.
10. Run outcome: error rate over threshold exits 1; under threshold exits 0.

## Scope

**In:** E-RUN-1, E-RUN-2, E-RUN-3, the run-level outcome and exit code, pytest bootstrap.

**Out, deliberately:**

- **Retry/backoff on transient judge errors.** Would reduce errored counts, but this PR is about
  honest reporting, not resilience. Follow-up finding.
- **Surfacing errored counts in the ops UI.** The run-level exit code is covered by PR #308.
  Per-metric error counts land in the results JSON for a later consumer.
- **E-RUN-4/5** (config passthrough) and **E-RUN-9/10** (metric correctness) belong to PRs 11
  and 12.

## Risks

- **Threshold default changes behaviour for existing runs.** Set too low, a flaky judge fails
  otherwise-good runs. 0.2 is a starting point and is configurable; worth revisiting once real
  error rates are observable.
- **The health probe could false-negative** on unusual endpoints that reject a probe request but
  serve real traffic. Mitigated by probing the same endpoint the eval will actually use.
- **`score` becoming nullable** touches every consumer that reads it. Checked against ops; any
  in-repo arithmetic on `score` needs a None guard.

## Follow-up

Correct the E-RUN-1 entry in `Misc/Training-Dataset-Eval/eval-findings.md`: the stated mechanism
(wrapper returns 0) sends the next reader to the wrong file. The load-bearing site is the
adapter's blanket `except`.
