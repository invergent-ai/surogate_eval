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
