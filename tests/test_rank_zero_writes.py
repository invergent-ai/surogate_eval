"""Only rank 0 writes the run's artifacts (E-RUN-6).

`cli/main.py` relaunches the whole pipeline under `torch.distributed.run`
with one process per GPU whenever the host has more than one CUDA device.
Every one of those processes then runs the same evaluation and writes the
same `eval_results/` files over each other: `bench_<name>.json`,
`progress.json`, `eval_<id>.json` and the summary report. Ops reads exactly
those files, so a multi-GPU pod publishes whichever process happened to
finish last, interleaved.

These tests pin the guard, not the launch heuristic. The heuristic (going
distributed at all for an eval that only calls an API) is a separate change;
the guard is correct either way and is what stops the corruption.

Both directions matter here. A guard that never writes is as broken as one
that always writes, and the single-process case is the only case that runs
today, so it is tested first.
"""

import json
from types import SimpleNamespace

import pytest

from surogate_eval import runners
from surogate_eval.eval import SurogateEval


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """The writers all resolve `eval_results` relative to the cwd.

    `_PROGRESS` is module state, so it is reset too: without that,
    `_flush_progress` serializes whatever counters an earlier test left
    behind and this file's assertions would depend on test order.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runners, "_PROGRESS", dict(runners._PROGRESS))
    return tmp_path / "eval_results"


@pytest.fixture(autouse=True)
def _no_inherited_rank(monkeypatch):
    """A stray RANK in the developer's shell must not decide these tests."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def _save_consolidated(results):
    """Drive eval.py's writer without building a whole SurogateEval."""
    stub = SimpleNamespace(
        consolidated_results=results,
        # The report is a second write on the same path and gets the same
        # guard for free; stubbed here so the test is about the JSON.
        _create_summary_report=lambda *a, **k: None,
    )
    SurogateEval._save_consolidated_results(stub)


# --- The single-process case still writes everything -----------------------


@pytest.mark.parametrize("rank", [None, "0"])
def test_the_writing_rank_writes_every_artifact(results_dir, monkeypatch, rank):
    """`None` is the single-process case, the only one that runs today."""
    if rank is not None:
        monkeypatch.setenv("RANK", rank)

    runners._write_bench_result({"benchmark_name": "gsm8k", "score": 1.0})
    runners._flush_progress()
    _save_consolidated({"targets": {}})

    assert json.loads((results_dir / "bench_gsm8k.json").read_text())["score"] == 1.0
    assert (results_dir / "progress.json").exists()
    assert list(results_dir.glob("eval_*.json"))


# --- Every other rank writes nothing ---------------------------------------


@pytest.mark.parametrize("rank", ["1", "3"])
def test_a_non_zero_rank_writes_no_bench_result(results_dir, monkeypatch, rank):
    monkeypatch.setenv("RANK", rank)

    runners._write_bench_result({"benchmark_name": "gsm8k", "score": 0.0})

    assert not results_dir.exists(), list(results_dir.iterdir())


def test_a_non_zero_rank_writes_no_progress(results_dir, monkeypatch):
    monkeypatch.setenv("RANK", "1")

    runners._flush_progress()

    assert not results_dir.exists()


def test_a_non_zero_rank_writes_no_consolidated_results(results_dir, monkeypatch):
    monkeypatch.setenv("RANK", "1")

    _save_consolidated({"targets": {}})

    assert not results_dir.exists()


def test_a_non_zero_rank_cannot_overwrite_rank_zeros_file(results_dir, monkeypatch):
    """The failure this exists to stop: rank 0 publishes its result and a
    later process replaces it with its own copy of the same benchmark."""
    runners._write_bench_result({"benchmark_name": "gsm8k", "score": 1.0})

    monkeypatch.setenv("RANK", "2")
    runners._write_bench_result({"benchmark_name": "gsm8k", "score": 0.0})

    assert json.loads((results_dir / "bench_gsm8k.json").read_text())["score"] == 1.0


def test_an_unparseable_rank_still_writes_every_artifact(results_dir, monkeypatch):
    """`is_master()` parses `RANK` with a bare `int()`, which raises on a
    value that is exported but not a number.

    Swallowing that inside each writer's own handler looked harmless and was
    not: `_save_consolidated_results` writes the run's *output*, after the
    outcome is computed, so a broken RANK produced a run that exits 0 with no
    results file at all -- the artifact ops reads. An unreadable rank writes
    instead of skipping, because two processes writing one file is
    recoverable and no process writing it is not.
    """
    monkeypatch.setenv("RANK", "not-a-rank")

    runners._flush_progress()
    runners._write_bench_result({"benchmark_name": "gsm8k", "score": 1.0})
    _save_consolidated({"targets": {}})

    assert (results_dir / "progress.json").exists()
    assert (results_dir / "bench_gsm8k.json").exists()
    assert list(results_dir.glob("eval_*.json")), "the run's own results were lost"
