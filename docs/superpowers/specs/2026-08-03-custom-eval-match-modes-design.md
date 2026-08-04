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
substring. Chosen as the default so the *direct* path does not regress before callers set a mode
explicitly.

Stated precisely, because "nothing regresses" is not true of both paths: the lm-eval path previously
took its verdict from lm-eval's own `exact_match` metric, which is stricter than containment. Rows
scored there get *looser* under the default, and an MCQ benchmark with a tokenizer configured gains
the same false positive the direct path already had. That is the cost of having one rule instead of
two, and it is bounded: the platform sets no tokenizer on eval targets, so only a runner-direct
config with a local model reaches that path, and setting `mode: exact` removes it.

**`exact`** — formatting cleanup, then case-insensitive equality, ignoring a trailing run of sentence
punctuation (`. ! , ; :`) on both sides.

That last part is not pedantry avoidance, it is the difference between measuring knowledge and
measuring formatting compliance. A model answering `D.` to an answer key of `D` has answered D, and
an `exact` benchmark reporting 0.0 across every row is as uninformative as one reporting 1.0.
Stripping on both sides rather than one keeps a key that legitimately ends in punctuation (`etc.`,
`Yes.`) able to match itself.

Brackets are deliberately excluded, so `D)` still fails. `)` can be part of an answer (`f(x)`,
`sin(x)`), and stripping it would score a truncated generation of `f(x` as correct: a silent false
positive, which is the failure this whole design exists to prevent. A model that writes `D)` is
better served by `regex`, which handles it along with every other shape.

The tolerance stops at the equality modes. `contains` is untouched, because the one promise the
default makes is that it reproduces the historical behaviour, and substring already tolerates
trailing punctuation anyway.

**`regex`** — apply the user's pattern to the cleaned output, take the capture group (`group`,
default 1, or 0 when the pattern has no groups), then compare that to `expected` by the same rule
`exact` uses: case-insensitive equality after stripping. The pattern sees the cleaned output rather
than the raw one, so a pattern does not have to anticipate markdown the runner is about to remove.

Only the `i` flag is accepted. `m` and `s` are the two that exist to change behaviour around
newlines, and the cleanup collapses every newline into a space before the pattern runs, so they
would validate and then do nothing: an anchored `^Answer: (\w+)$` with `flags: m` scores every row
0.0 with no signal the flag was inert. Rejected at build time rather than accepted-and-ignored.

Extraction rather than matching is deliberate. Models bury the answer in prose, which is the problem
the heuristics in `_normalize_output` were reaching for. A regex extractor is the same idea with the
user in control and the rule visible, and it composes with the `expected` value every row already
has, where a pure "does the output match this pattern" mode would ignore `expected` entirely.

### What "exact" means, and where the heuristics go

Formatting cleanup stays in all three modes: markdown stripping and whitespace collapsing are about
presentation, not about deciding the answer.

**It applies to the model's output only, never to `expected`.** The cleanup is lossy by design: it
renders markdown and drops anything HTML-shaped. That is right for model prose and wrong for an
answer key, which is a literal the dataset chose. Run on `expected`, a literal `<answer>` cleans to
`''` and `1. Paris` to `Paris`, so the benchmark silently measures something the dataset never asked
for. The `<answer>` case is the worst, because it compounds with the rule below. `expected` gets
`.strip().lower()`, which is all it ever got before this change.

**A row whose `expected` is blank cannot be scored at all.** Every output contains the empty string,
so `contains` would score an entire benchmark 1.0 off one empty answer column, and a null cell or the
literal string `null` reaches the comparison as `''`. Neither verdict is honest: "correct" is that
fail-open, and "incorrect" records a row nobody measured as one the model got wrong. It raises, and
both call sites already record a raised comparison as `status: 'errored'`.

The email/percentage/date extraction retires. It is exactly what `regex` mode does explicitly, and
keeping both means two competing extractors and results nobody can explain. `_normalize_output`
splits into a formatting-only helper the string modes share.

**Verified safe for the one benchmark in production:** those heuristics are gated on the *expected*
value's shape (containing an `@`, being a bare percentage, matching a date). `greeting-check`'s
expected value is a long greeting, so none of them fire for it.

**The judge path is affected too, though it is otherwise out of scope here.** It also called
`_normalize_output`, so retiring that method changes what G-Eval receives: where the expected value
had one of those shapes, the judge used to be handed a *fragment* of the response and now gets the
whole cleaned text. That is the better input, since a judge scoring an answer should see the answer
rather than a regex's guess at it, but it is a real change and not a no-op.

### Outcomes

| Situation | Result | Why |
|---|---|---|
| Pattern matches, extracted value equals expected | score 1.0 | |
| Pattern matches, extracted value differs | score 0.0 | wrong answer |
| Pattern does not match the output | score 0.0 | the pattern is the answer format the benchmark asked for; a model that produced nothing matching it did not answer correctly |
| Pattern is invalid | benchmark fails before scoring | config error, every row would hit it |
| Match exceeds the timeout | that row is errored | a failure to measure, not the model's fault |
| `matcher.mode` is unrecognised | benchmark fails before scoring | it currently falls into the containment bucket silently (`custom_eval_backend.py:250-262`) |
| `matcher` is present but not a mapping | benchmark fails before scoring | including the falsy ones (`matcher: []`), which a `cfg or {}` would quietly read as "no matcher" |
| `expected` is blank for a row | that row is errored | nothing to score against; see above |
| `expected` is punctuation only (`.`) | that row is errored | survives the blank check but weighs nothing once the trailing run is stripped, so it would match any punctuation-only output |

The matcher is built once per benchmark, before the direct/lm-eval path choice. Building it inside
the lm-eval path put it under that call's blanket `except Exception`, so a matcher typo was reported
as `lm-eval exact_match failed, falling back to direct inference: ...` and then raised again from the
direct path, pointing the first diagnostic at something that was not broken.

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

**Pairing rows to results becomes score-bearing, so it fails closed.** Results are matched to sent
rows by position. While the score came from lm-eval's own per-sample metric that was cosmetic: a
reordered sample list only mislabeled the displayed columns. Now `detail[i]`'s generation is compared
against `row[i]`'s answer key, so a reorder would score one row's output against another row's
answer, silently and plausibly. lm-eval returns each sample's own `target`, so a disagreement is
detectable; a row that cannot be paired confidently is recorded unmeasured rather than guessed at.

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
- An `expected` that the formatting cleanup would rewrite (`1. Paris`, `# 42`, `<answer>`) is
  compared as written: an output matching only the cleaned form scores 0.0.
- A blank, whitespace-only or absent `expected` raises rather than scoring the row correct.
- A non-mapping `matcher` is rejected whether it is truthy or falsy.
- `flags: m` and `flags: s` are rejected; `flags: i` still works.
- A reordered lm-eval result set errors both rows instead of scoring them against each other's keys.
- A bad matcher never enters the lm-eval path, so it is not reported as an lm-eval failure.
- Under `exact`, `D.` `D!` `D,` and `**D**.` all match a key of `D`, while `D)` does not, and a
  truncated `f(x` does not match a key of `f(x)`.
- Every key matches itself whatever its punctuation, including `etc.` and `f(x)`.
- A punctuation-only key raises rather than matching a punctuation-only output.
- `contains` is unchanged by the tolerance.

## Out of scope

- Sending an explicit mode from the platform, and flipping the default to `exact`. Both follow once
  custom benchmark creation exists.
- The create form's pattern field, the Custom Python option, and its hardcoded `columns`. They edit a
  form whose payload is discarded today, so they land with the work that makes it survive.
- Sending an explicit mode from the platform, and flipping the default to `exact`, are both listed
  above and remain out of scope here.
