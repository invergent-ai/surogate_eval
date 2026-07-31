"""The lm-eval benchmark path, counted the way the run outcome counts it.

These cases cannot be driven end-to-end: lm-eval's ``simple_evaluate``
needs a served model, and it writes its task YAML into the installed
lm-eval package. So the harness output is the stub - that is lm-eval's
shape, not ours - and everything from ``_parse_results`` onwards is the
real code, right through to ``compute_outcome``. No fixture here is a
hand-written copy of a shape this repo emits.
"""

import pytest

from surogate_eval.benchmarks.base import BenchmarkConfig
from surogate_eval.benchmarks.generic import GenericBenchmark
from surogate_eval.outcome import compute_outcome, exit_code_for

pytest.importorskip("lm_eval")


def lm_eval_harness_output(**results):
    """The dict lm-evaluation-harness returns from simple_evaluate()."""
    return {"results": results, "samples": {}}


def run_outcome_for(harness_output):
    """Drive the real chain: lm-eval output -> backend parse -> BenchmarkResult
    -> to_dict -> the target entry the runner builds -> compute_outcome."""
    benchmark = GenericBenchmark(
        BenchmarkConfig(name="custom_task", backend="lm_eval")
    )
    parsed = benchmark.backend._parse_results(
        harness_output, "custom_task", "some/dataset"
    )
    result_dict = benchmark._parse_results(parsed).to_dict()
    # runners._run_single_benchmark stamps this on before recording it.
    result_dict["status"] = "completed"

    consolidated = {
        "targets": [{"name": "t1", "status": "success", "benchmarks": [result_dict]}]
    }
    return result_dict, compute_outcome(consolidated)


def test_scored_task_is_positive_evidence_of_measurement():
    result_dict, outcome = run_outcome_for(
        lm_eval_harness_output(custom_task={"exact_match,none": 0.42, "alias": "x"})
    )

    assert (result_dict["scored_n"], result_dict["errored_n"]) == (1, 0)
    assert outcome["status"] == "completed"
    assert exit_code_for(outcome) == 0


def test_task_without_a_numeric_metric_is_errored_not_dropped():
    result_dict, outcome = run_outcome_for(
        lm_eval_harness_output(
            good_task={"exact_match,none": 1.0},
            broken_task={"alias": "broken_task", "exact_match,none": "n/a"},
        )
    )

    assert (result_dict["scored_n"], result_dict["errored_n"]) == (1, 1)
    assert result_dict["task_results"]["broken_task"]["score"] is None
    assert outcome["error_rate"] == 0.5
    assert exit_code_for(outcome) == 1


def test_harness_output_with_no_tasks_at_all_fails_the_run():
    """lm-eval came back with nothing. The benchmark measured nothing, and
    saying so is the whole point of the exercise."""
    result_dict, outcome = run_outcome_for(lm_eval_harness_output())

    assert (result_dict["scored_n"], result_dict["errored_n"]) == (0, 1)
    assert outcome["status"] == "failed"
    assert exit_code_for(outcome) == 1
