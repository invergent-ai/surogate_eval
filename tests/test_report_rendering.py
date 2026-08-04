"""A row nobody measured must not read as a score in the report.

Errored results carry ``score=None``. Jinja's ``default()`` substitutes only
for UNDEFINED, so ``{{ detail.score | default(0) }}`` left None alone and the
operator-facing markdown printed the literal string "None" - on exactly the
rows this branch exists to surface. Substituting 0 instead would be worse: a
row nobody measured would read as a score of zero.

The rows here come out of the real custom-eval backend against a real local
CSV, put through the real benchmark parse and the real report generator. No
network, and no hand-written copy of a shape this repo emits.
"""

import pytest

from surogate_eval.benchmarks.base import BenchmarkConfig
from surogate_eval.benchmarks.generic import GenericBenchmark
from surogate_eval.report import ReportGenerator
from surogate_eval.targets.base import TargetResponse


class FakeTarget:
    name = "t1"
    config = {}

    def send_request(self, request):
        return TargetResponse(content="a polite answer", raw_response={}, error=None)


class DeadJudgeMetric:
    """Stands in for deepeval's ToxicityMetric, with a judge that is down."""

    def __init__(self, **kwargs):
        self.threshold = kwargs.get("threshold", 0.5)
        self.score = None
        self.reason = None

    def measure(self, test_case):
        raise RuntimeError("judge 500")


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(
        "input,expected_output\n"
        "Say something nice,A nice thing\n"
        "Say something else,Another thing\n",
        encoding="utf-8",
    )
    return path


def markdown_for(dataset, monkeypatch):
    """Drive the real chain: backend -> benchmark parse -> to_dict -> the
    target entry the runner builds -> the real markdown report."""
    monkeypatch.setattr(
        "deepeval.metrics.ToxicityMetric", lambda **kwargs: DeadJudgeMetric(**kwargs)
    )

    benchmark = GenericBenchmark(
        BenchmarkConfig(
            name="toxicity_check",
            backend="custom_eval",
            source=str(dataset),
            eval_type="toxicity",
            columns={"instruction": "input", "answer": "expected_output"},
        )
    )
    result_dict = benchmark.evaluate(FakeTarget()).to_dict()
    # runners._run_single_benchmark stamps this on before recording it.
    result_dict["status"] = "completed"

    consolidated = {
        "targets": [{"name": "t1", "status": "success", "benchmarks": [result_dict]}]
    }
    return result_dict, ReportGenerator().generate_markdown(consolidated)


def test_an_unmeasured_row_reads_as_unmeasured_not_as_none(dataset, monkeypatch):
    result_dict, markdown = markdown_for(dataset, monkeypatch)

    # The rows really are unscored, or this proves nothing.
    details = result_dict["detailed_results"]
    assert details and all(row["score"] is None for row in details)

    assert "**Score:** None" not in markdown
    assert "| None |" not in markdown
    assert "**Score:** not measured" in markdown
    assert "| not measured |" in markdown


def test_an_unmeasured_row_does_not_read_as_a_score_of_zero(dataset, monkeypatch):
    """The other way to make "None" go away, and the reason not to take it."""
    _, markdown = markdown_for(dataset, monkeypatch)

    assert "**Score:** 0" not in markdown


def _report_for(task_results):
    """Render a custom_eval benchmark with the given task_results.

    ``custom_eval`` because that is the only backend whose ``task_results``
    reach the report at all: both templates select the task table with
    ``selectattr('backend', 'equalto', 'custom_eval')``.
    """
    from surogate_eval.benchmarks.base import BenchmarkResult
    from surogate_eval.report import ReportGenerator

    result = BenchmarkResult(
        benchmark_name="cx_quality",
        overall_score=0.9,
        num_samples=20,
        backend="custom_eval",
        task_results=task_results,
        detailed_results=[],
        metadata={},
    ).to_dict()
    result["status"] = "completed"

    consolidated = {
        "targets": [{"name": "t1", "status": "success", "benchmarks": [result]}]
    }
    gen = ReportGenerator()
    return gen.generate_markdown(consolidated), gen.generate_html(consolidated)


TASKS = {
    # Shape the custom-eval backend really emits for a toxicity run.
    "toxicity": {
        "total": 10,
        "safe": 8,
        "toxic": 2,
        "safety_rate": 0.8,
        "scored_n": 10,
        "errored_n": 0,
    },
    # A task nothing could measure.
    "broken": {
        "status": "failed",
        "score": None,
        "n_samples": 10,
        "reason": "no usable 'score'",
    },
}


def test_a_measured_task_is_not_dropped_for_using_its_own_metric_key():
    """The task table branched only on ``accuracy`` and ``avg_score``.

    A toxicity run reports under ``safety_rate``, so its only task row
    matched neither branch and, with no ``else``, rendered nothing at all -
    a measured result silently absent from the operator-facing report.
    """
    markdown, _html = _report_for(TASKS)

    assert "toxicity" in markdown, "a measured task must not vanish from the table"
    assert "80" in markdown


def test_a_task_nobody_could_measure_still_appears_as_unmeasured():
    """And one that genuinely was not measured says so, rather than vanishing."""
    markdown, _html = _report_for(TASKS)

    assert "broken" in markdown
    assert "not measured" in markdown


def test_the_task_table_never_renders_a_ragged_html_row():
    """The html row opens unconditionally, so a missing branch yields two
    cells in a four-column table rather than an omitted row."""
    _markdown, html = _report_for(TASKS)

    for task in ("toxicity", "broken"):
        start = html.index(f"<td>{task}</td>")
        row = html[start:html.index("</tr>", start)]
        assert row.count("<td") == 4, f"{task}: ragged row with {row.count('<td')} cells"


# --- unmeasured rows are not failures ----------------------------------


def _rows(*specs):
    """Build detailed_results rows. ``specs`` are (status, success) pairs."""
    rows = []
    for i, (status, success) in enumerate(specs):
        row = {
            "original_idx": i,
            "eval_type": "exact_match",
            "instruction": f"q{i}",
            "expected": "A",
            "output": "A" if success else "B",
            "raw_output": "A" if success else "B",
            "score": None if status == "errored" else (1.0 if success else 0.0),
            "success": success,
            "reason": "unreachable target" if status == "errored" else "contains match",
        }
        if status is not None:
            row["status"] = status
        rows.append(row)
    return rows


def _render_rows(rows, task_results=None):
    from surogate_eval.benchmarks.base import BenchmarkResult
    from surogate_eval.report import ReportGenerator

    result = BenchmarkResult(
        benchmark_name="custom_bench",
        overall_score=1.0,
        num_samples=len(rows),
        backend="custom_eval",
        task_results=task_results or {},
        detailed_results=rows,
        metadata={},
    ).to_dict()
    result["status"] = "completed"
    consolidated = {
        "targets": [{"name": "t1", "status": "success", "benchmarks": [result]}]
    }
    return ReportGenerator().generate_markdown(consolidated)


def test_an_unmeasured_row_is_not_counted_as_a_failure():
    """Four rows scored, all correct; six nobody could measure.

    Counting the unmeasured six as failures reported a model that got
    everything right as 40%. The rate is over what was measured, matching how
    the backend already computes ``overall_score``.
    """
    rows = _rows(*([("scored", True)] * 4 + [("errored", False)] * 6))

    markdown = _render_rows(rows)

    assert "4 passed" in markdown
    assert "0 failed" in markdown
    assert "6 not measured" in markdown
    assert "40" not in markdown.split("Performance by Evaluation Type")[0].split(
        "Total Test Cases"
    )[-1], "the pass rate must not be diluted by rows nobody scored"


def test_a_benchmark_where_nothing_could_be_measured_does_not_read_as_zero_percent():
    """The case that reads as "this model is broken" when the truth is "we
    tested nothing". A healthy target reaches this through a blank answer
    column, a match timeout, or a short lm-eval return."""
    markdown = _render_rows(_rows(*([("errored", False)] * 3)))

    assert "- **Passed / Failed:** not measured, no row could be scored" in markdown
    assert "- **Not measured:** 3" in markdown
    assert "- Exact Match: ⚠️ not measured (3 cases)" in markdown
    # The per-type and summary lines must carry no rate at all. Asserted on
    # the lines that would carry one rather than on the whole document, since
    # a bare "0.0%" substring also matches the unrelated "100.0%".
    rate_lines = [
        line for line in markdown.splitlines()
        if line.startswith(("- **Passed:**", "- **Failed:**", "- Exact Match:"))
    ]
    assert not any("%" in line for line in rate_lines)


def test_an_unmeasured_row_keeps_its_diagnostic():
    """Dropping unmeasured rows from "Failed Cases" must not drop the reason
    with them: it is the only place the real cause appears."""
    markdown = _render_rows(_rows(("scored", False), ("errored", False)))

    assert "Not Measured (1)" in markdown
    assert "unreachable target" in markdown
    assert "Failed Cases Details (1)" in markdown, "the genuinely wrong row still lists"


def test_rows_without_a_status_key_still_count_as_scored():
    """`lm_eval`'s direct results carry neither `status` nor `success`, and
    other backends predate the key. "No status" must mean scored, or those
    benchmarks silently start reporting every row as unmeasured."""
    rows = _rows((None, True), (None, False))

    markdown = _render_rows(rows)

    assert "1 passed" in markdown
    assert "1 failed" in markdown
    assert "not measured" not in markdown


def test_a_task_that_measured_nothing_reports_no_rate(dataset, monkeypatch):
    """Driven through the real backend, not a hand-written `task_results`.

    The template's "not measured" branch keys on the metric being ABSENT, and
    it already existed. What stopped it firing was the backend filling the key
    in with a zero, so a test that hand-writes the task dict without the key
    proves only that the branch works, and would pass against the unfixed
    backend. This asserts the shape the backend actually emits.
    """
    result_dict, markdown = markdown_for(dataset, monkeypatch)

    # The judge is dead, so every toxicity row errored. Or this proves nothing.
    assert all(row["score"] is None for row in result_dict["detailed_results"])

    task = result_dict["task_results"]["toxicity"]
    assert task["scored_n"] == 0 and task["errored_n"] == 2
    assert "safety_rate" not in task, (
        "a rate over zero measured rows is indistinguishable from a model "
        "that failed every row"
    )
    assert "not measured" in markdown


def test_an_unmeasured_row_is_not_listed_as_a_model_failure_pattern():
    """"Common Failure Patterns" is about how the model failed. A row that
    could not be scored is a defect in the dataset or the run, so listing
    "row has no expected answer" there sends an operator looking at the model
    for a problem that is not in it."""
    rows = _rows(("scored", False), ("errored", False))

    markdown = _render_rows(rows)

    patterns = markdown.split("**Common Failure Patterns:**")[-1]
    assert "contains match" in patterns or "No match" in patterns or "1 case" in patterns
    assert "unreachable target" not in patterns


def test_an_errored_row_that_claims_success_cannot_make_failures_negative():
    """Defensive: `failed` is `scored - passed`, so counting passes over ALL
    rows would go negative if an errored row ever carried `success: True`.
    Counted over scored rows instead, so the arithmetic holds regardless."""
    rows = _rows(("scored", True), ("errored", False))
    rows[1]["success"] = True  # the shape that would break the subtraction

    markdown = _render_rows(rows)

    assert "- **Passed:** 1 " in markdown
    assert "- **Failed:** 0 " in markdown
    assert "- **Not measured:** 1" in markdown
