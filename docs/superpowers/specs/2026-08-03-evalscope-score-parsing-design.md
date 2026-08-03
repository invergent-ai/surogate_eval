# Design: stop the evalscope parser inventing per-sample scores

**Date:** 2026-08-03
**Repo:** `surogate_eval`
**Findings covered:** E-RUN-10

## Problem

`_load_predictions` reads a per-sample score out of each evalscope review row
(`benchmarks/backends/evalscope_backend.py:478-506`). When it cannot find a score it guesses,
and the guess is reported as if it were measured.

```python
main_name = score_obj.get('main_score_name')
if main_name and main_name in value:
    score = float(value[main_name])
else:
    score = value.get('acc', 0.0)
if score == 0.0:
    for key in ['accuracy', 'correct', 'score', ..., 'prompt_level_strict', 'em', 'f1', ...]:
        if key in value:
            score = float(value[key])
            break
    if score == 0.0 and value:
        nums = [v for v in value.values() if isinstance(v, (int, float))]
        if nums:
            score = sum(nums) / len(nums)
```

Two defects, one on top of the other.

**`if score == 0.0` conflates "no score found" with "the score is zero".** A sample evalscope
scored `0.0`, meaning the model got it wrong, is indistinguishable from a sample whose key was
missing. So a correctly-parsed wrong answer falls into the key sweep, and then into the
average-everything fallback, and comes back out with a fabricated non-zero score.

**The fallback averages unrelated numbers.** Whatever is in the value dict gets summed and
divided, whether or not the entries are commensurable, or even scores.

### Reproduced

`IFEval` emits four numbers per sample and names the first as the main score
(`evalscope/benchmarks/ifeval/utils.py:123-128`). A sample that **failed** the prompt-level
constraint while following three of its four instructions:

```
value = {'prompt_level_strict': 0.0, 'inst_level_strict': 0.75,
         'prompt_level_loose': 0.0, 'inst_level_loose': 0.75}
main_score_name = 'prompt_level_strict'

  -> parsed score 0.375, success True
```

evalscope said the sample failed. We record `0.375` and flag it as a pass.

The general case:

| value dict | parsed score |
|---|---|
| `{acc: 0.0}` | `0.0` (correct) |
| `{acc: 0.0, num_tokens: 512}` | `256.0` |
| `{acc: 0.0, latency_ms: 900.0}` | `450.0` |
| `{acc: 0.0, f1: 0.62}`, main `acc` | `0.62` |
| `{grade: 3, confidence: 0.9}` | `1.95` |

A per-sample score of `256.0` on a metric whose range is 0 to 1.

### Where the wrong number goes

`evalscope_backend.py:565` sets `'success': float(score) > 0`, so a fabricated `0.375` becomes
`success: True`, which is what the consuming UI renders as a correct sample.

The **run-level score is not affected**: it comes from evalscope's own aggregate, not from these
per-sample values. What is corrupted is the per-sample evidence. A run can report 41% while the
sample list shows a wall of passes. Someone opening a run to decide whether they believe the
score is shown fabricated evidence, which is worse than a wrong number they could have
distrusted.

### A third, quieter defect

`_parse_results` reads `overall_score = results.get('score', 0.0)` (`:659`). Two cases hide
behind that default:

- **The report file is missing.** `results` is `{}` (`:394`), no task parses, and
  `BenchmarkResult.result_counts()` already charges one errored unit. Handled.
- **The report loads and its subsets parse, but there is no top-level `score` key.** This is what
  an upstream key rename looks like. `overall_score` silently becomes `0.0` while `scored_n` is
  positive, so the run completes and reports that the model scored zero.

`evalscope>=1.7.0` is an open upper bound, so the second case is a live upgrade hazard.

## Decisions taken

**An unrecognised schema affects the evidence, not the run outcome.** The sample is recorded as
unmeasured; `task_results` keeps its current shape and no `scored_n`/`errored_n` are emitted, so
`error_rate` and the exit code are unchanged.

The alternative was to emit per-sample counts, matching what `custom_eval_backend` does. Rejected
for now on two grounds. `BenchmarkResult.result_counts()` documents why per-sample counting is
unsafe on this path: several backends report a task without a trustworthy sample count, so
counting samples can report a healthy benchmark as having measured nothing. And adding hundreds
of benchmark samples to a single run-wide denominator worsens the dilution problem that the
run-wide `error_rate` rule already has, which needs its own decision rather than a side effect of
this change.

**`success = float(score) > 0` is out of scope.** It is wrong for any graded metric: a weak
answer with a low non-zero score reads as a pass. Fixing it properly needs a per-benchmark notion
of what a pass is, and the stored scale is unverified for several benchmarks. The headline case
here is fixed without it, because once a legitimate `0.0` stops being overridden, `success`
follows correctly.

## Approach

**Per-sample extraction.** Replace the `score == 0.0` cascade with an explicit found/not-found
lookup using `None` as the sentinel, and delete the averaging fallback. When no known key
matches, the sample is unmeasured:

```python
'score': None,
'success': False,
'status': 'errored',
'reason': "unrecognised score schema: {...keys seen...}",
```

This is the shape `custom_eval_backend` already uses for a row it could not compare
(`custom_eval_backend.py:462-471`), so it extends an existing convention rather than adding one.

**Keep the known-key list.** Two alternatives were considered:

- *Trust `main_score_name` alone.* Simpler, but it is confirmed present on only a few adapters
  and cannot be assumed for all of them.
- *A per-benchmark extraction table.* The most correct long-term shape, but validating it needs a
  real run per benchmark, which is not available here. Recorded as the eventual direction.

The list encodes real accumulated per-benchmark knowledge. The fix is to make its failure
explicit rather than to guess past it.

**Benchmark-level score.** Distinguish the two cases above. A report that loads and parses but
carries no usable top-level `score` raises a named schema-mismatch error, so the benchmark fails
loudly instead of publishing a fabricated zero. This fires only when an upstream change breaks
the parser, which is exactly when a run should stop. The parse runs inside the backend's retry
loop, so the error must be non-retryable: a schema mismatch will not resolve on attempt two.
"No usable" covers a present-but-null score as well as an absent key, since guarding on absence
alone lets `"score": null` through and the fabrication happens anyway one step later.

**Subset-level score, added after review.** The same read exists one loop down
(`subset.get('score', 0.0)`), and `BenchmarkResult.result_counts()` then counts that subset as
scored on the defaulted number. Mitigating context, recorded so the priority is not overstated:
evalscope declares `score` at report, metric, category and subset level in one model, so a rename
trips the report-level guard first. This is consistency rather than a second live hole.

It is **recorded as failed rather than raised**. Raising was the first implementation and an
independent review was right to call it out: it discarded every subset already parsed in the same
report, so one bad subset in fifty-six cost the whole benchmark. That made the subset case the only
place the module escalated rather than isolated, when a bad row costs one row. A failed subset is
charged as one errored unit by `result_counts()`, which is not the same as defaulting it to zero:
it is marked unmeasured, not scored.

The report-level guard still raises, because a report with no usable score has nothing left to
salvage.

**Row-level isolation, added after review.** The field guards above close the instances of the
non-dict hazard we found, but not the class: `instruction_id_list`, for one, is fed straight to
`enumerate()`, so a non-list there raises from a spot no guard covers. The review-file loop wrapped
every row of a file in one `try`, so any such raise abandoned every remaining row in that file,
visible only as a warning. The row builder moves into its own method and each row is read inside
its own `try`; an unreadable row is recorded as unmeasured and costs exactly one row. A file-level
handler remains for what is genuinely file-level (open, permissions, a read failing mid-iteration),
where there is no row to attribute the failure to.

The field guards stay rather than being replaced by the row handler. They are more precise: when
only a row's score is malformed they salvage its input, expected output and subset and mark just
the score unmeasured, where the row-level handler would lose all of it.

An unreadable row produces a placeholder record rather than being skipped. Dropping it would
understate the sample count and hide the failure, which is the same defect this design exists to
remove, one level up.

## Downstream

No companion change is needed in the consuming service. It already guards the sample score with
`if s.get("score") is not None` before copying it, and derives its `correct` flag from `success`,
which will be `False`. A null score is therefore safe on that side.

## Testing

Table-driven over real score shapes, no network:

- IFEval's four-key dict: must go from `0.375` and passing to `0.0` and failing.
- DROP's `{em, f1}`.
- MBPP's `{acc: <bool>}`. Note `isinstance(False, int)` is `True` in Python, so booleans currently
  flow into the numeric average.
- An unrecognised schema: must be unmeasured, never averaged.
- A legitimate zero: must stay `0.0` and must not fall through.
- A report missing its top-level `score`, a subset missing its own, and either of them present but
  null: must raise rather than report `0.0`.
- A healthy report: must parse without raising, with the right `overall_score` and `task_results`.
  Without this the raise has no allow-direction coverage, and inverting its condition so that every
  real report fails would leave the suite green.
- A truncated JSON line, and a field no guard was written for (`instruction_id_list` as a non-list):
  each must cost one row, with the rest of the file intact.

## Out of scope

- `success = float(score) > 0` for graded metrics.
- Feeding per-sample errors into the run-wide error rate.
- A per-benchmark score-extraction table.
