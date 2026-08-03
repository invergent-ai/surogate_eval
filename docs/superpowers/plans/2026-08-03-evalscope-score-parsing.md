# Evalscope per-sample score parsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the evalscope result parser reporting a fabricated score for a sample it could not read, and stop a schema change reading as a benchmark that scored zero.

**Architecture:** The score-extraction logic is currently ~40 lines inlined nine levels deep inside the review-file loop in `_load_predictions`, which is why it has never been tested. Task 1 lifts it into a module-level pure function with corrected semantics (`None` means "not measured", never "scored zero") and deletes the average-everything fallback. Tasks 2 and 3 extend the same posture to two adjacent cases. Each task is independently testable and independently rejectable.

**Tech Stack:** Python 3.12, pytest. No new dependencies.

## Global Constraints

- **No test may make a network call.** `tests/conftest.py` enforces the offline and telemetry
  environment; every test in this plan operates on plain dicts.
- **A failure to measure is never a score.** An unreadable sample is `score=None`,
  `success=False`, `status='errored'`, matching `custom_eval_backend.py:462-471`.
- **`task_results` shape is unchanged.** No `scored_n`/`errored_n` is emitted from this backend, so
  the run-wide `error_rate` and the process exit code are untouched by this plan.
- **`success = float(score) > 0` is out of scope** except where `None` would raise.
- Run tests with `./.venv/bin/python -m pytest`.
- Commit messages use Conventional Commits (`fix:`, `test:`, `refactor:`).

---

### Task 1: Pure score extraction that distinguishes absent from zero

**Files:**
- Modify: `surogate_eval/benchmarks/backends/evalscope_backend.py:466-508` (extraction block), `:558-568` (sample record)
- Test: `tests/test_evalscope_score_parsing.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: module-level `_extract_sample_score(score_obj: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any], Optional[str]]`, returning `(score, score_details, reason)`. `score is None` means not measured; `reason` is non-`None` exactly when `score is None`. Task 2 calls this same function.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evalscope_score_parsing.py`:

```python
"""A sample we cannot read a score for is unmeasured, not scored zero.

The extraction used ``if score == 0.0`` as its "keep looking" signal, which
made a sample evalscope genuinely scored 0.0 indistinguishable from one whose
key was missing. A correctly-parsed wrong answer therefore fell through to a
fallback that averaged every number in the row, and came back out with a
fabricated non-zero score: an IFEval sample that failed its prompt-level
constraint was recorded at 0.375 and flagged as a pass.

No network: these are plain dicts shaped like evalscope review rows.
"""

import pytest

from surogate_eval.benchmarks.backends.evalscope_backend import _extract_sample_score

# (label, score_obj, expected_score)
CASES = [
    (
        "ifeval failed prompt-level but followed 3 of 4 instructions",
        {
            "main_score_name": "prompt_level_strict",
            "value": {
                "prompt_level_strict": 0.0,
                "inst_level_strict": 0.75,
                "prompt_level_loose": 0.0,
                "inst_level_loose": 0.75,
            },
        },
        0.0,
    ),
    (
        "ifeval passed",
        {
            "main_score_name": "prompt_level_strict",
            "value": {"prompt_level_strict": 1.0, "inst_level_strict": 1.0},
        },
        1.0,
    ),
    (
        "drop reports em and f1, main score is f1",
        {"main_score_name": "f1", "value": {"em": 0.0, "f1": 0.62}},
        0.62,
    ),
    (
        "mbpp writes a bool under acc",
        {"main_score_name": "acc", "value": {"acc": False}},
        0.0,
    ),
    (
        "mbpp passed",
        {"main_score_name": "acc", "value": {"acc": True}},
        1.0,
    ),
    (
        "no main_score_name, acc is the fallback",
        {"value": {"acc": 0.0}},
        0.0,
    ),
    (
        "no main_score_name and no acc, a known key is used",
        {"value": {"winrate": 0.4}},
        0.4,
    ),
    (
        "a bare numeric value rather than a dict",
        {"value": 0.75},
        0.75,
    ),
    (
        "a zero score sitting beside an unrelated count is still zero",
        {"main_score_name": "acc", "value": {"acc": 0.0, "num_tokens": 512}},
        0.0,
    ),
]


@pytest.mark.parametrize(
    "label, score_obj, expected",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_known_schemas_are_read_as_written(label, score_obj, expected):
    score, details, reason = _extract_sample_score(score_obj)

    assert score == expected
    assert reason is None
    if isinstance(score_obj["value"], dict):
        assert details == score_obj["value"]


UNREADABLE = [
    ("schema we do not recognise", {"value": {"grade": 3, "confidence": 0.9}}),
    ("a value that is not a number", {"value": {"acc": "n/a"}}),
    ("an empty value dict", {"value": {}}),
    ("no value at all", {}),
]


@pytest.mark.parametrize("label, score_obj", UNREADABLE, ids=[c[0] for c in UNREADABLE])
def test_unreadable_samples_are_unmeasured_never_averaged(label, score_obj):
    score, _details, reason = _extract_sample_score(score_obj)

    assert score is None, "an unreadable sample must not be given a number"
    assert reason, "an unmeasured sample must say why"


def test_the_averaging_fallback_is_gone():
    """The regression this task exists to prevent, stated as its own case."""
    score, _details, _reason = _extract_sample_score(
        {"value": {"acc": 0.0, "latency_ms": 900.0}}
    )

    assert score == 0.0, "must not average the latency into the score"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_evalscope_score_parsing.py -v`
Expected: FAIL at import, `ImportError: cannot import name '_extract_sample_score'`

- [ ] **Step 3: Write the extraction function**

In `surogate_eval/benchmarks/backends/evalscope_backend.py`, add near the top of the module (after the existing imports and module-level constants, before the backend class):

```python
#: Keys we know how to read a sample score from, tried in this order after
#: ``main_score_name`` and ``acc``. Each entry is a benchmark we have seen
#: write its score under that name.
_KNOWN_SCORE_KEYS = (
    'accuracy', 'correct', 'score',
    'resolved', 'passed', 'pass',
    'is_correct',           # simple_qa, simple_vqa
    'prompt_level_strict',  # IFEval/IFBench
    'em', 'exact_match',    # DROP, drivelology
    'f1',                   # DROP, tool_bench
    'winrate',              # alpaca_eval
    'overall_score',        # healthbench
    'eq_bench_score',       # eq_bench
    'total_score',          # mia_bench
)


def _extract_sample_score(
        score_obj: Dict[str, Any]
) -> Tuple[Optional[float], Dict[str, Any], Optional[str]]:
    """Read one sample's score out of an evalscope review row.

    Returns ``(score, score_details, reason)``. A ``score`` of ``None`` means
    the row could not be read, which is not the same thing as a row that
    scored zero, and the two must never be conflated: the previous version
    used ``score == 0.0`` as its "keep looking" signal, so a correctly-parsed
    wrong answer fell through to a fallback that averaged every number in the
    row and returned a fabricated non-zero score.

    ``reason`` is set exactly when ``score`` is ``None``.
    """
    value = score_obj.get('value')

    # Some adapters write a bare number rather than a dict of named metrics.
    if isinstance(value, (int, float)):
        return float(value), {}, None

    if not isinstance(value, dict) or not value:
        return None, {}, f"no readable score value (got {type(value).__name__})"

    main_name = score_obj.get('main_score_name')
    candidates = ([main_name] if main_name else []) + ['acc'] + list(_KNOWN_SCORE_KEYS)

    for key in candidates:
        if key not in value:
            continue
        try:
            return float(value[key]), value, None
        except (TypeError, ValueError):
            # A key we recognise carrying something we cannot read is a
            # failure to measure, not a reason to go looking for a number
            # elsewhere in the row.
            return None, value, f"score key {key!r} is not a number: {value[key]!r}"

    return None, value, f"unrecognised score schema, keys: {sorted(value)}"
```

Change line 5 from `from typing import Dict, Any, List` to
`from typing import Dict, Any, List, Optional, Tuple`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_evalscope_score_parsing.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Replace the inline block with a call**

In `_load_predictions`, replace lines 466-508 (from `# Extract score from EvalScope's nested structure` through `score = float(value)`) with:

```python
                            # Extract score from EvalScope's nested structure
                            # sample_score.score.value.<metric>
                            score = 0.0
                            score_details = {}
                            score_reason = None
                            sample_score = sample.get('sample_score') or {}
                            if sample_score:
                                score_obj = sample_score.get('score') or {}
                                if score_obj:
                                    score, score_details, score_reason = (
                                        _extract_sample_score(score_obj)
                                    )
```

- [ ] **Step 6: Update the sample record so a null score is representable**

Replace the `detailed_results.append({...})` call (currently `:558-568`) with:

```python
                            detailed_results.append({
                                'input': input_text,
                                'expected': expected,
                                'output': extracted or prediction[:500],
                                'raw_output': prediction,
                                'score': score,
                                'score_details': score_details,
                                'success': score is not None and score > 0,
                                'status': 'errored' if score is None else 'scored',
                                'reason': score_reason,
                                'subset': (sample_score.get('sample_metadata') or {}).get('subject', ''),
                                'metadata': sample
                            })
```

`float(score)` is gone from both places: it raises on `None`.

- [ ] **Step 7: Add the record-level test**

Append to `tests/test_evalscope_score_parsing.py`:

```python
def test_an_unmeasured_sample_is_not_reported_as_a_pass():
    """The record the UI reads, not just the extraction helper.

    ``success`` is what the consuming service renders as a correct sample, so
    an unreadable row must not set it. The score must stay ``None`` rather
    than becoming 0.0, because the consumer distinguishes them.
    """
    score, _details, reason = _extract_sample_score({"value": {"grade": 3}})

    record = {
        'score': score,
        'success': score is not None and score > 0,
        'status': 'errored' if score is None else 'scored',
        'reason': reason,
    }

    assert record['score'] is None
    assert record['success'] is False
    assert record['status'] == 'errored'
    assert 'unrecognised score schema' in record['reason']
```

- [ ] **Step 8: Run the full suite**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS, no regressions against the 195 currently passing

- [ ] **Step 9: Commit**

```bash
git add surogate_eval/benchmarks/backends/evalscope_backend.py tests/test_evalscope_score_parsing.py
git commit -m "fix(benchmarks): stop the evalscope parser inventing a score it could not read

The extraction used \`score == 0.0\` as its keep-looking signal, so a sample
evalscope genuinely scored zero was indistinguishable from one whose key was
missing. The wrong answer then fell through to a fallback that averaged every
number in the row. An IFEval sample that failed its prompt-level constraint
while following three of four instructions was recorded at 0.375 and flagged
as a pass; a row carrying a token count could score 256.0 on a 0-1 metric.

Lift the extraction out of the review loop into a pure function, use None as
the not-measured sentinel, and delete the averaging. An unreadable row is now
score=None, success=False, status=errored, with a reason naming what it saw."
```

---

### Task 2: A review row with no score object is unmeasured too

**Files:**
- Modify: `surogate_eval/benchmarks/backends/evalscope_backend.py` (the `score = 0.0` initialiser added in Task 1, Step 5)
- Test: `tests/test_evalscope_score_parsing.py`

**Interfaces:**
- Consumes: `_extract_sample_score` from Task 1.
- Produces: nothing new.

**Reviewer note:** this is the one step in the plan with real behavioural reach, and it is a separate task so it can be rejected on its own. Task 1 leaves a row with no `sample_score` (or no `score` inside it) scoring 0.0, which is the same fabrication in a quieter place. Changing it is correct, but if any benchmark omits `sample_score` routinely, it moves many samples to unmeasured at once. Nothing in the repo indicates that, and it cannot be confirmed without a real run.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evalscope_score_parsing.py`:

```python
def test_a_row_with_no_score_object_is_unmeasured(tmp_path):
    """A row carrying no score at all did not score zero either.

    Task 1 fixed the case where the score object exists but cannot be read.
    A row with no ``sample_score`` took a different path and kept the old
    0.0 default, which reports it as an answer the model got wrong.

    Driven through the real review-file loop rather than the helper, because
    the defect is in the loop's initialiser, not in the extraction. No
    network: this reads one file from tmp_path.
    """
    import json

    from surogate_eval.benchmarks.backends.evalscope_backend import EvalScopeBackend

    reviews = tmp_path / "reviews" / "model-under-test"
    reviews.mkdir(parents=True)
    (reviews / "gsm8k_default.jsonl").write_text(
        json.dumps({"input": "2+2?", "target": "4"}) + "\n"
    )

    backend = EvalScopeBackend.__new__(EvalScopeBackend)
    rows = backend._load_predictions(str(tmp_path), "model-under-test", "gsm8k")

    assert len(rows) == 1
    assert rows[0]["score"] is None, "a row with no score object is not a zero"
    assert rows[0]["success"] is False
    assert rows[0]["status"] == "errored"
    assert rows[0]["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_evalscope_score_parsing.py::test_a_row_with_no_score_object_is_unmeasured -v`
Expected: FAIL, `assert 0.0 is None` — the loop's initialiser still supplies a zero

- [ ] **Step 3: Change the initialiser**

In `_load_predictions`, change the block from Task 1 Step 5 to:

```python
                            # Extract score from EvalScope's nested structure
                            # sample_score.score.value.<metric>. A row we
                            # cannot read a score from is unmeasured, which
                            # includes a row carrying no score object at all.
                            sample_score = sample.get('sample_score') or {}
                            score_obj = sample_score.get('score') or {}
                            score, score_details, score_reason = _extract_sample_score(score_obj)
```

`_extract_sample_score({})` already returns `(None, {}, "no readable score value ...")`, so the empty cases need no special handling.

`score_obj` is now always bound (to `{}` when absent), so the later prediction block a few
lines down loses its guard. Change:

```python
                            # Get the model's prediction/output
                            prediction = ''
                            extracted = ''
                            if sample_score:
                                score_obj = sample_score.get('score') or {}
                                prediction = score_obj.get('prediction', '') or ''
                                extracted = score_obj.get('extracted_prediction', '') or ''
```

to:

```python
                            # Get the model's prediction/output
                            prediction = score_obj.get('prediction', '') or ''
                            extracted = score_obj.get('extracted_prediction', '') or ''
```

This removes a second, now-redundant derivation of `score_obj` from the same `sample_score`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_evalscope_score_parsing.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add surogate_eval/benchmarks/backends/evalscope_backend.py tests/test_evalscope_score_parsing.py
git commit -m "fix(benchmarks): treat a review row with no score object as unmeasured

The loop initialised score to 0.0 before looking, so a row carrying no
sample_score at all was reported as a sample the model got wrong rather than
one that was never scored. Same fabrication as the averaging fallback, in a
quieter place."
```

---

### Task 3: A missing top-level score is a schema mismatch, not a zero

**Files:**
- Modify: `surogate_eval/errors.py`, `surogate_eval/benchmarks/backends/evalscope_backend.py:659`
- Test: `tests/test_evalscope_score_parsing.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `surogate_eval.errors.BenchmarkSchemaError(EvalError)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evalscope_score_parsing.py`:

```python
def test_a_report_without_a_top_level_score_raises_rather_than_reporting_zero():
    """An evalscope key rename must not read as "the model scored zero".

    A missing report file is already handled: results is {}, no task parses,
    and BenchmarkResult.result_counts() charges one errored unit. The silent
    case is a report that loads and whose subsets parse, but that carries no
    top-level ``score`` key.
    """
    from surogate_eval.benchmarks.backends.evalscope_backend import EvalScopeBackend
    from surogate_eval.errors import BenchmarkSchemaError

    backend = EvalScopeBackend.__new__(EvalScopeBackend)
    renamed = {
        "overall": 0.83,  # what a rename might look like
        "metrics": [
            {"name": "acc", "categories": [
                {"subsets": [{"name": "default", "score": 0.83, "num": 100}]}
            ]}
        ],
    }

    with pytest.raises(BenchmarkSchemaError) as excinfo:
        backend._parse_results(renamed, "gsm8k", [])

    assert "score" in str(excinfo.value)


def test_an_empty_report_still_parses_so_the_existing_backstop_handles_it():
    """The missing-file case must keep its current behaviour."""
    from surogate_eval.benchmarks.backends.evalscope_backend import EvalScopeBackend

    backend = EvalScopeBackend.__new__(EvalScopeBackend)

    parsed = backend._parse_results({}, "gsm8k", [])

    assert parsed["overall_score"] == 0.0
    assert parsed["task_results"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_evalscope_score_parsing.py -k top_level_score -v`
Expected: FAIL, `ImportError: cannot import name 'BenchmarkSchemaError'`

- [ ] **Step 3: Add the error type**

Append to `surogate_eval/errors.py`:

```python
class BenchmarkSchemaError(EvalError):
    """A backend's result payload did not have the shape we know how to read.

    Raised rather than defaulted past, because the alternative is publishing a
    number we did not measure. Not retryable: an upstream schema change will
    not resolve on the next attempt.
    """
```

- [ ] **Step 4: Raise it in the parser**

In `_parse_results`, replace:

```python
        task_results = {}
        overall_score = results.get('score', 0.0)
```

with:

```python
        task_results = {}

        # An empty payload means the report file was missing. That is already
        # handled downstream: no task parses, and BenchmarkResult.result_counts
        # charges one errored unit. A payload that loaded but has no 'score' is
        # different - it is what an upstream key rename looks like - and
        # defaulting it to 0.0 reports a model that scored zero.
        if results and 'score' not in results:
            raise BenchmarkSchemaError(
                f"evalscope report for {benchmark_name!r} has no 'score' key; "
                f"got keys: {sorted(results)}"
            )

        overall_score = results.get('score', 0.0)
```

Add the import beside the file's existing absolute imports (near line 64, alongside
`from surogate_eval.targets import BaseTarget`):

```python
from surogate_eval.errors import BenchmarkSchemaError
```

This file uses absolute imports throughout; do not use a relative one here.

- [ ] **Step 5: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_evalscope_score_parsing.py -v`
Expected: PASS

- [ ] **Step 6: Confirm the error is not retried**

Run: `./.venv/bin/python -c "from surogate_eval.benchmarks.backends.evalscope_backend import EvalScopeBackend; from surogate_eval.errors import BenchmarkSchemaError; b = EvalScopeBackend.__new__(EvalScopeBackend); print('retryable:', b._is_retryable_error(BenchmarkSchemaError('evalscope report has no score key')))"`
Expected: `retryable: False`

If this prints `True`, stop: the message overlaps a `RETRYABLE_ERRORS` substring and must be reworded.

- [ ] **Step 7: Run the full suite**

Run: `./.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add surogate_eval/errors.py surogate_eval/benchmarks/backends/evalscope_backend.py tests/test_evalscope_score_parsing.py
git commit -m "fix(benchmarks): raise on an evalscope report with no top-level score

results.get('score', 0.0) turned an upstream key rename into a benchmark that
scored zero: the subsets parse, scored_n is positive, and the run completes
reporting a real-looking number nobody measured. The evalscope floor is an
open upper bound, so this is a live upgrade hazard.

The missing-file case keeps its current behaviour, since an empty payload is
already charged as one errored unit downstream."
```

---

## Self-Review

**Spec coverage.** Defect 1 (absent vs zero) and defect 2 (averaging fallback) are Task 1. The
quieter benchmark-level variant is Task 3. The spec's testing list maps to Task 1's `CASES` and
`UNREADABLE` tables, except the "report missing its top-level score" case, which is Task 3. Task 2
is not in the spec: it is the same defect one line earlier, found while writing the plan, and it is
isolated so it can be dropped without touching the rest.

**Placeholders.** None. Every code step carries the code.

**Type consistency.** `_extract_sample_score` returns a 3-tuple in Task 1 and is called as a 3-tuple
in Tasks 1 and 2. `BenchmarkSchemaError` is defined in Task 3 Step 3 and used in Steps 1 and 4.
`score` is `Optional[float]` from Task 1 Step 3 onward, and every consumer added after that point
guards for `None`.

**Out of scope, unchanged by this plan:** `success = float(score) > 0` for graded metrics, and
feeding per-sample errors into the run-wide error rate.
