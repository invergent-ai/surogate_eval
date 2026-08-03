"""A sample we cannot read a score for is unmeasured, not scored zero.

The extraction used ``if score == 0.0`` as its "keep looking" signal, which
made a sample evalscope genuinely scored 0.0 indistinguishable from one whose
key was missing. A correctly-parsed wrong answer therefore fell through to a
fallback that averaged every number in the row, and came back out with a
fabricated non-zero score: an IFEval sample that failed its prompt-level
constraint was recorded at 0.375 and flagged as a pass.

No network: these are plain dicts shaped like evalscope review rows.
"""

import json

import pytest

from surogate_eval.benchmarks.backends.evalscope_backend import (
    EvalScopeBackend,
    _extract_sample_score,
)
from surogate_eval.errors import BenchmarkSchemaError

#: The model id every file-driven test writes under. Arbitrary, but the
#: reviews directory is keyed on it, so writer and reader must agree.
_MODEL_ID = "model-under-test"


def _load_rows(tmp_path, *rows, dataset="gsm8k"):
    """Write ``rows`` as an evalscope review file and read them back.

    Returns whatever the real ``_load_predictions`` produced, so a test can
    assert on the records the backend actually builds. The scaffolding is
    identical for every file-driven test here; only the rows differ, and
    keeping it in one place makes the row under test the visible part.
    """
    reviews = tmp_path / "reviews" / _MODEL_ID
    reviews.mkdir(parents=True, exist_ok=True)
    (reviews / f"{dataset}_default.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    backend = EvalScopeBackend.__new__(EvalScopeBackend)
    return backend._load_predictions(str(tmp_path), _MODEL_ID, dataset)

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


def test_a_row_with_no_score_object_is_unmeasured(tmp_path):
    """A row carrying no score at all did not score zero either.

    Task 1 fixed the case where the score object exists but cannot be read.
    A row with no ``sample_score`` took a different path and kept the old
    0.0 default, which reports it as an answer the model got wrong.

    A second, scored row is included alongside it so the same test drives
    both halves of the loop through ``_load_predictions``: the unmeasured
    branch (no ``sample_score``) and the scored branch (a realistic
    ``sample_score``), the latter of which regressed silently when a guard
    was removed and ``prediction``/``extracted`` stopped being read.

    Driven through the real review-file loop rather than the helper, because
    the defect is in the loop's initialiser, not in the extraction. No
    network: this reads one file from tmp_path.
    """

    unmeasured_row = {"input": "2+2?", "target": "4"}
    scored_row = {
        "input": "What is 6 times 7?",
        "target": "42",
        "sample_score": {
            "score": {
                "value": {"acc": 1.0},
                "main_score_name": "acc",
                "prediction": "Let me work through this. 6 times 7 is 42.",
                "extracted_prediction": "42",
            },
            "sample_metadata": {"subject": "arithmetic"},
        },
    }
    rows = _load_rows(tmp_path, unmeasured_row, scored_row)

    assert len(rows) == 2

    unmeasured, scored = rows

    assert unmeasured["score"] is None, "a row with no score object is not a zero"
    assert unmeasured["success"] is False
    assert unmeasured["status"] == "errored"
    assert unmeasured["reason"]

    assert scored["score"] == 1.0
    assert scored["success"] is True
    assert scored["status"] == "scored"
    assert scored["reason"] is None
    assert scored["output"] == "42", "extracted_prediction must win over the raw prediction"
    assert scored["raw_output"] == "Let me work through this. 6 times 7 is 42."
    assert scored["subset"] == "arithmetic"


def test_a_report_without_a_top_level_score_raises_rather_than_reporting_zero():
    """An evalscope key rename must not read as "the model scored zero".

    A missing report file is already handled: results is {}, no task parses,
    and BenchmarkResult.result_counts() charges one errored unit. The silent
    case is a report that loads and whose subsets parse, but that carries no
    top-level ``score`` key.
    """

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

    backend = EvalScopeBackend.__new__(EvalScopeBackend)

    parsed = backend._parse_results({}, "gsm8k", [])

    assert parsed["overall_score"] == 0.0
    assert parsed["task_results"] == {}


def test_a_healthy_report_parses_without_raising():
    """The allow direction for the new raise: every real evalscope run.

    The deny case (a report that loaded but has no top-level ``score``) and
    the empty-payload case are both tested above. Neither exercises the path
    every real run actually takes: a report that has a top-level ``score``
    plus ``metrics -> categories -> subsets`` entries. A realistic payload
    must parse cleanly and produce correct ``overall_score``, ``task_results``
    and ``num_samples``.
    """

    backend = EvalScopeBackend.__new__(EvalScopeBackend)
    healthy_report = {
        "score": 0.83,
        "dataset_name": "gsm8k",
        "model_name": "model-under-test",
        "metrics": [
            {
                "name": "AverageAccuracy",
                "categories": [
                    {
                        "subsets": [
                            {"name": "default", "score": 0.83, "num": 100},
                        ],
                    },
                ],
            },
        ],
    }

    parsed = backend._parse_results(healthy_report, "gsm8k", [])

    assert parsed["overall_score"] == 0.83
    assert parsed["task_results"] == {
        "default": {"score": 0.83, "accuracy": 0.83, "n_samples": 100},
    }
    assert parsed["num_samples"] == 100


def test_a_malformed_middle_row_does_not_drop_the_rest_of_the_file(tmp_path):
    """One bad row must not lose itself and every row after it.

    ``sample_score`` and its nested ``score`` are both supposed to be dicts.
    A malformed file can carry a non-dict in either spot (e.g. a list from a
    truncated write). Before this fix, ``.get(...)`` on that non-dict raised
    ``AttributeError``, which the per-file ``except Exception`` in
    ``_load_predictions`` caught - abandoning the whole file. A three-row
    file with one bad row in the middle came back with only 1 row instead
    of 3, and the loss was visible only as a ``logger.warning``.

    No network: this reads one file from tmp_path.
    """

    good_row_1 = {
        "input": "2+2?",
        "target": "4",
        "sample_score": {
            "score": {
                "value": {"acc": 1.0},
                "main_score_name": "acc",
                "prediction": "4",
                "extracted_prediction": "4",
            },
        },
    }
    malformed_row = {
        "input": "3+3?",
        "target": "6",
        # A truncated/corrupt write: the nested score is a list, not a dict.
        "sample_score": {"score": [1, 2, 3]},
    }
    good_row_2 = {
        "input": "5+5?",
        "target": "10",
        "sample_score": {
            "score": {
                "value": {"acc": 0.0},
                "main_score_name": "acc",
                "prediction": "11",
                "extracted_prediction": "11",
            },
        },
    }
    rows = _load_rows(tmp_path, good_row_1, malformed_row, good_row_2)

    assert len(rows) == 3, "a malformed middle row must not lose itself or later rows"

    first, middle, last = rows
    assert first["score"] == 1.0
    assert middle["score"] is None, "a non-dict score object cannot be given a number"
    assert middle["success"] is False
    assert middle["status"] == "errored"
    assert middle["reason"]
    assert last["score"] == 0.0


def test_no_sample_score_and_no_value_get_distinct_reasons():
    """Two different failure causes must not share one reason string.

    A row with no score object at all (``sample_score`` missing, or present
    without a nested ``score``) means the review process never produced a
    score for this sample. A row whose score object exists but carries no
    ``value`` field means scoring was attempted and something about the
    value's shape broke. A whole-file wall of unmeasured rows is triaged
    very differently depending on which of the two happened, so the two
    reasons must read differently.
    """
    _, _, no_score_object_reason = _extract_sample_score({})
    _, _, no_value_field_reason = _extract_sample_score(
        {"main_score_name": "acc", "prediction": "some model output"}
    )

    assert no_score_object_reason
    assert no_value_field_reason
    assert no_score_object_reason != no_value_field_reason


def test_a_malformed_sample_metadata_does_not_drop_the_rest_of_the_file(tmp_path):
    """The third instance of the same non-dict hazard, in ``sample_metadata``.

    ``sample_score`` and its nested ``score`` are both guarded. ``sample_metadata``
    is read the same way at two more places - once to build a readable
    ``expected``, once for the ``subset`` field - and both spell it
    ``(sample_score.get('sample_metadata') or {})``, which passes a truthy
    non-dict straight through to ``.get()``. So a list there raised
    ``AttributeError`` into the per-file handler and lost the rest of the
    file, exactly as a bad ``score`` used to.

    No network: this reads one file from tmp_path.
    """

    def row(idx, metadata):
        return {
            "input": f"q{idx}",
            "target": "",  # empty, so the metadata-driven `expected` path runs too
            "sample_score": {
                "score": {
                    "main_score_name": "acc",
                    "value": {"acc": 1.0},
                    "prediction": f"a{idx}",
                },
                "sample_metadata": metadata,
            },
        }

    rows = _load_rows(
        tmp_path,
        row(0, {"subject": "algebra"}),
        row(1, ["not", "a", "dict"]),  # the malformed one
        row(2, {"subject": "geometry"}),
    )

    assert len(rows) == 3, "a bad sample_metadata must not lose the rest of the file"
    assert rows[0]["subset"] == "algebra"
    assert rows[2]["subset"] == "geometry"
    # The malformed row still parses; only its metadata is unusable.
    assert rows[1]["score"] == 1.0
    assert rows[1]["subset"] == ""


def test_a_subset_without_a_score_raises_like_the_report_does():
    """The subset-level twin of the report-level guard.

    `_parse_results` raises when the report has no top-level `score`, on the
    grounds that defaulting it reports a model that scored zero. One loop
    down, each subset's score was still read with `.get('score', 0.0)`, which
    is the same fabrication at a level the report guard does not cover, and
    `result_counts()` then counts that subset as scored.
    """
    backend = EvalScopeBackend.__new__(EvalScopeBackend)
    renamed_subset = {
        "score": 0.83,
        "metrics": [
            {
                "name": "acc",
                "categories": [
                    {"subsets": [{"name": "default", "value": 0.83, "num": 100}]}
                ],
            }
        ],
    }

    with pytest.raises(BenchmarkSchemaError) as excinfo:
        backend._parse_results(renamed_subset, "gsm8k", [])

    assert "default" in str(excinfo.value)


def test_an_unparseable_line_does_not_lose_the_rest_of_the_file(tmp_path):
    """A truncated line is one unreadable row, not the end of the file."""
    reviews = tmp_path / "reviews" / _MODEL_ID
    reviews.mkdir(parents=True)
    good = {
        "input": "2+2?",
        "target": "4",
        "sample_score": {"score": {"value": {"acc": 1.0}, "main_score_name": "acc"}},
    }
    (reviews / "gsm8k_default.jsonl").write_text(
        json.dumps(good) + "\n" + '{"input": "3+3?", "sample_sc\n' + json.dumps(good) + "\n"
    )

    backend = EvalScopeBackend.__new__(EvalScopeBackend)
    rows = backend._load_predictions(str(tmp_path), _MODEL_ID, "gsm8k")

    assert len(rows) == 3, "a truncated line must cost one row, not the file"
    assert rows[0]["score"] == 1.0
    assert rows[1]["score"] is None
    assert rows[1]["status"] == "errored"
    assert rows[1]["reason"]
    assert rows[2]["score"] == 1.0


def test_an_unanticipated_field_failure_is_contained_to_its_row(tmp_path):
    """The class of hazard, not the three instances we happened to find.

    Field-level guards were added for `sample_score`, `score` and
    `sample_metadata`. They do not cover every field the row builder reads:
    `instruction_id_list` is fed straight to `enumerate()`, so a non-list
    there raises `TypeError` from a spot nobody guarded. Before per-row
    isolation that took the rest of the file with it. This test deliberately
    uses a field NO guard was written for, so it keeps proving the class is
    closed even if the specific guards change.
    """
    def row(idx, metadata):
        return {
            "input": f"q{idx}",
            "target": "",  # empty, so the metadata-driven `expected` path runs
            "sample_score": {
                "score": {"value": {"acc": 1.0}, "main_score_name": "acc"},
                "sample_metadata": metadata,
            },
        }

    rows = _load_rows(
        tmp_path,
        row(0, {"subject": "algebra"}),
        row(1, {"instruction_id_list": 42}),  # not a list: enumerate() raises
        row(2, {"subject": "geometry"}),
    )

    assert len(rows) == 3, "an unguarded field must not lose the rest of the file"
    assert rows[0]["subset"] == "algebra"
    assert rows[2]["subset"] == "geometry"
    assert rows[1]["status"] == "errored"
    assert rows[1]["score"] is None
    assert rows[1]["reason"]


@pytest.mark.parametrize(
    "label, payload",
    [
        (
            "report-level score is explicitly null",
            {"score": None, "metrics": []},
        ),
        (
            "subset-level score is explicitly null",
            {
                "score": 0.8,
                "metrics": [
                    {
                        "name": "acc",
                        "categories": [
                            {"subsets": [{"name": "default", "score": None, "num": 10}]}
                        ],
                    }
                ],
            },
        ),
    ],
    ids=["report", "subset"],
)
def test_an_explicitly_null_score_is_a_schema_problem_too(label, payload):
    """A present-but-null score is as unreadable as an absent one.

    Guarding on key absence alone lets `"score": null` through, and the
    fabrication it was meant to stop happens anyway one step later: the
    subset counts as scored, and the run fails much further downstream on a
    generic TypeError from formatting rather than saying what was wrong.
    """
    backend = EvalScopeBackend.__new__(EvalScopeBackend)

    with pytest.raises(BenchmarkSchemaError):
        backend._parse_results(payload, "gsm8k", [])
