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
