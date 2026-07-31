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
