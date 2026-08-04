# Design: named match modes for custom-eval string scoring

**Date:** 2026-08-03
**Repo:** `surogate_eval`
**Findings covered:** E-RUN-9

## Problem

A custom benchmark scored by string comparison uses substring containment, and calls it exact match.
`_evaluate_exact_match_direct` (`benchmarks/backends/custom_eval_backend.py:449-455`):

```python
expected_clean = expected.strip().lower()
output_clean = normalized_output.strip().lower()
success = (
    expected_clean == output_clean
    or expected_clean in output_clean
    or output_clean.startswith(expected_clean)
)
```

`startswith` is subsumed by the `in` test, so the effective rule is `expected in output`, after a
normalisation pass. Run against the real code:

| expected | model output | scored |
|---|---|---|
| `A` | `The answer is C.` | **correct** |
| `no` | `Nobody knows.` | **correct** |
| `4` | `It took 14 minutes.` | **correct** |
| `Paris` | `Paris is not the capital; Berlin is.` | **correct** |

The multiple-choice case is the worst and is not hypothetical for anyone building a benchmark from an
MCQ dataset: expected `"A"` matches any output containing the letter "a", so every row passes and the
benchmark reports 100%.

There is a second, quieter half. `_normalize_output` (`:177`) does two different jobs under one name:
it strips markdown and collapses whitespace, which is formatting cleanup, and it also *guesses the
answer* out of the output with hardcoded heuristics for emails, percentages and dates. So even the
equality branch is comparing against something the runner rewrote on the model's behalf.

## Scope note

This is not user-visible today. No user-created custom benchmark can reach the platform: the create
page discards its payload at the call site, and the only other code that builds one is unmounted. The
sole custom-eval benchmark in production is the onboarding `greeting-check`, whose expected value is
a long greeting, where containment is arguably what is wanted.

The fix is still the right one, and it has to exist before custom benchmark creation ships, because
every non-judge benchmark a user builds lands on this comparison.

## Design

### One `matcher` block, not four loose keys

```yaml
eval_type: exact_match      # who scores the row: string comparison, judge, hybrid
matcher:
  mode: regex               # how the string comparison decides
  pattern: '\b([ABCD])\b'
  flags: i
  group: 1
```

`eval_type` keeps meaning *who scores this*; `matcher.mode` means *how*. The alternative, adding a
`regex` value to `eval_type`, was rejected: `hybrid` reads a per-row `eval_type` column to route rows
between string and judge scoring, so a new value there would have to be understood by that column
too, and it conflates two orthogonal choices.

`matcher` travels as a `BenchmarkConfig` field passed through `generic.py`, mirroring how `eval_type`
already reaches the backend (`base.py:60`, `generic.py:72`, `runners.py:381`).

### Three modes

**`contains`** — the default, and today's behaviour minus the heuristics. Formatting cleanup, then
substring. Chosen as the default so nothing regresses before callers set a mode explicitly.

**`exact`** — formatting cleanup, then case-insensitive equality.

**`regex`** — apply the user's pattern to the cleaned output, take the capture group (`group`,
default 1, or 0 when the pattern has no groups), then compare that to `expected` by the same rule
`exact` uses: case-insensitive equality after stripping. The pattern sees the cleaned output rather
than the raw one, so a pattern does not have to anticipate markdown the runner is about to remove.

Extraction rather than matching is deliberate. Models bury the answer in prose, which is the problem
the heuristics in `_normalize_output` were reaching for. A regex extractor is the same idea with the
user in control and the rule visible, and it composes with the `expected` value every row already
has, where a pure "does the output match this pattern" mode would ignore `expected` entirely.

### What "exact" means, and where the heuristics go

Formatting cleanup stays in all three modes: markdown stripping and whitespace collapsing are about
presentation, not about deciding the answer.

The email/percentage/date extraction retires. It is exactly what `regex` mode does explicitly, and
keeping both means two competing extractors and results nobody can explain. `_normalize_output`
splits into a formatting-only helper the string modes share.

**Verified safe for the one benchmark in production:** those heuristics are gated on the *expected*
value's shape (containing an `@`, being a bare percentage, matching a date). `greeting-check`'s
expected value is a long greeting, so none of them fire for it.

### Outcomes

| Situation | Result | Why |
|---|---|---|
| Pattern matches, extracted value equals expected | score 1.0 | |
| Pattern matches, extracted value differs | score 0.0 | wrong answer |
| Pattern does not match the output | score 0.0 | the pattern is the answer format the benchmark asked for; a model that produced nothing matching it did not answer correctly |
| Pattern is invalid | benchmark fails before scoring | config error, every row would hit it |
| Match exceeds the timeout | that row is errored | a failure to measure, not the model's fault |
| `matcher.mode` is unrecognised | benchmark fails before scoring | it currently falls into the containment bucket silently (`custom_eval_backend.py:250-262`) |

Not a load-time check: it runs when the benchmark first scores a string row, so a matcher block on a
benchmark that scores none is never validated. Harmless, since it has no effect there either way.

### Engine

The `regex` module, already installed as a transitive dependency (2025.11.3), used for its `timeout`
parameter. Patterns are user input and stdlib `re` backtracks with no timeout, so a catastrophic
pattern would hang the eval pod. No new dependency is added, and `google-re2` is not needed.

Worth stating plainly: this is a self-inflicted foot-gun rather than an attack surface. The pattern
comes from the tenant running their own benchmark, on their own per-run pod. The timeout exists so a
bad pattern costs a row instead of the run.

### Both scoring paths, one matcher

`exact_match` rows take one of two paths: direct inference when no tokenizer is configured, and
lm-eval when one is (`custom_eval_backend.py:287-292`). Only the direct path does its own comparison.
The lm-eval path delegates correctness to lm-eval's own metric:

```python
score = 1.0 if detail.get('metrics', {}).get('exact_match', 0) else 0.0
```

So the same benchmark is scored by different rules depending on whether a tokenizer happens to be
set, and neither rule is the one the user asked for.

**lm-eval keeps generation; the comparison comes back to us.** Its `exact_match` metric is no longer
read. Each returned row is scored by the same matcher the direct path uses, against the same cleaned
text, so a benchmark's results stop depending on its tokenizer setting.

Compare `raw_output`, not `output`. `LMEvalBackend` derives `output` by running its own
`_extract_answer` heuristic (`lm_eval_backend.py:466-472`), which is a *third* extractor beside the
matcher and the normaliser heuristics being retired here. Taking `raw_output` through the same
cleanup as the direct path makes the two identical: raw model text, our cleanup, our matcher. As on
the direct path, the record's `output` then holds our cleaned value and `raw_output` the model's.

`_extract_answer` is left in place; lm-eval's own benchmark path still uses it. It simply stops
deciding custom-eval scores.

**A row lm-eval never returned is unmeasured.** When it returns fewer rows than were sent, the
remainder currently become `score: 0.0, success: False, reason: 'No result'`
(`custom_eval_backend.py:357-362`) — a row that was never scored, recorded as an answer the model got
wrong. It gains `score: None`, `success: False`, `status: 'errored'` and a reason, matching the
convention its sibling `_evaluate_exact_match_direct` already follows. It also currently carries no
`status` key at all, unlike every other row this backend emits.

## Testing

No network; all cases are plain strings through the real comparison.

- The four false positives above must fail under `exact` and under `regex`, and still pass under
  `contains`, so the default is pinned as today's behaviour.
- MCQ extraction: expected `A`, output `The answer is C.`, pattern `\b([ABCD])\b` → extracts `C`,
  scores 0.0.
- Formatting cleanup survives in all modes: expected `42`, output `**42**` → passes under `exact`.
- The retired heuristics: expected `a@b.com`, output `Contact: a@b.com` → now fails under `exact`,
  and passes under a `regex` mode with an email pattern. This is the behaviour change, stated as a
  test.
- A pattern with no capture group falls back to group 0.
- No match scores 0.0 rather than erroring.
- An invalid pattern and an unknown mode each fail the benchmark, naming the offending value.
- A catastrophic pattern is bounded by the timeout and errors that row without taking the run.
- `greeting-check`'s shape (long expected, output containing it) keeps passing under the default.
- The same rows, scored through both paths, agree: a case that fails under `exact` on the direct path
  fails on the lm-eval path too, rather than depending on the tokenizer setting.
- A row lm-eval did not return is unmeasured, not a wrong answer.

## Out of scope

- Sending an explicit mode from the platform, and flipping the default to `exact`. Both follow once
  custom benchmark creation exists.
- The create form's pattern field, the Custom Python option, and its hardcoded `columns`. They edit a
  form whose payload is discarded today, so they land with the work that makes it survive.
- Sending an explicit mode from the platform, and flipping the default to `exact`, are both listed
  above and remain out of scope here.
