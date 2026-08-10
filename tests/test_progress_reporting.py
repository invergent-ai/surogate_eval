import json
import threading
import time
from pathlib import Path

import pytest

from surogate_eval import runners


@pytest.fixture(autouse=True)
def in_tmp_cwd(tmp_path, monkeypatch):
    """Every writer call resolves `eval_results/` relative to cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runners, "_PROGRESS_CONTEXT",
        {"current_benchmark": "", "completed": 0, "total": 1},
    )
    monkeypatch.setattr(runners, "_PROGRESS_ROWS", {})
    yield


def _read():
    return json.loads(Path("eval_results/progress.json").read_text())


def test_benchmark_level_write_keeps_todays_shape():
    """An old ops reading this file must still find the three keys it knows."""
    runners._write_progress("gsm8k", 0, 3)

    data = _read()
    assert data["current_benchmark"] == "gsm8k"
    assert data["completed"] == 0
    assert data["total"] == 3


def test_row_progress_keeps_the_benchmark_context():
    """The row half must not drop the benchmark half. `eval.py` writes the
    benchmark keys before the backend starts, and a producer that wrote only
    row keys would erase the name the UI shows."""
    runners._write_progress("gsm8k", 1, 3)
    runners.report_rows(
        rows_done=7, rows_total=30, scored=6, errored=1, passed=4, score_sum=4.0,
    )

    data = _read()
    assert data["current_benchmark"] == "gsm8k"
    assert data["completed"] == 1
    assert data["total"] == 3
    assert data["rows_done"] == 7
    assert data["rows_total"] == 30
    assert data["scored"] == 6
    assert data["errored"] == 1
    assert data["passed"] == 4
    assert data["score_sum"] == 4.0


def test_a_new_benchmark_zeroes_the_previous_rows():
    """Otherwise benchmark 2 inherits benchmark 1's row counts and the UI
    shows 30/30 the moment the next benchmark starts.

    Zeroed, not omitted: ops treats an absent key as "this runner does not
    report rows" and leaves the database's previous rows_done/rows_total
    alone, so dropping the keys here left benchmark 2 showing benchmark 1's
    finished bar until its own first report. Explicit zeros are a real
    value ops ingests (`0 is not None`), and `rows_total: 0` already means
    "unknown" on both sides.
    """
    runners._write_progress("gsm8k", 0, 2)
    runners.report_rows(
        rows_done=30, rows_total=30, scored=30, errored=0, passed=20, score_sum=20.0,
    )

    runners._write_progress("mmlu", 1, 2)

    data = _read()
    assert data["current_benchmark"] == "mmlu"
    assert data["rows_done"] == 0
    assert data["rows_total"] == 0
    assert data["scored"] == 0
    assert data["errored"] == 0
    assert data["passed"] == 0
    assert data["score_sum"] == 0.0


def test_the_file_is_never_observed_half_written(monkeypatch):
    """Ops reads this on a 5s poll. A truncating in-place write can be caught
    mid-flight and parsed as invalid JSON; a temp-file rename cannot."""
    written = []
    real_replace = runners.os.replace

    def spy(src, dst):
        # At the moment of rename the destination must be either absent or
        # complete — never a partial document.
        if Path(dst).exists():
            json.loads(Path(dst).read_text())
        written.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(runners.os, "replace", spy)

    runners._write_progress("gsm8k", 0, 1)
    runners.report_rows(
        rows_done=1, rows_total=2, scored=1, errored=0, passed=1, score_sum=1.0,
    )

    assert len(written) == 2
    assert _read()["rows_done"] == 1


def test_a_write_failure_never_raises(monkeypatch):
    """Progress is best-effort. A full disk must not fail an eval that is
    otherwise working."""
    def boom(*a, **kw):
        raise OSError("no space left on device")

    monkeypatch.setattr(runners.os, "replace", boom)

    runners._write_progress("gsm8k", 0, 1)  # must not raise
    runners.report_rows(
        rows_done=1, rows_total=2, scored=1, errored=0, passed=1, score_sum=1.0,
    )


def test_the_file_is_readable_by_another_account():
    """`mkstemp` defaults to 0600. Ops reads this file from outside the
    process that writes it, and the whole path is best-effort, so a
    permission regression would fail silently rather than loudly."""
    import stat

    runners._write_progress("gsm8k", 0, 1)

    mode = Path("eval_results/progress.json").stat().st_mode
    assert mode & stat.S_IROTH, oct(mode)


def test_report_rows_cannot_interleave_with_a_benchmark_switch(monkeypatch):
    """`report_rows`'s "does for_benchmark match?" read and its write are two
    separate statements. Without a lock shared with `_write_progress`, the
    benchmark switch can run in the gap between them: `report_rows` reads
    the *old* benchmark name (check passes), the switch runs to completion
    in between (context now the new benchmark, row block zeroed), and then
    `report_rows` resumes and overwrites those zeros with the old
    benchmark's stale counts -- landing on the new benchmark's name.
    """
    class _PausingContext(dict):
        """Stands in for `_PROGRESS_CONTEXT`: the first read of
        `current_benchmark` blocks until released, simulating a benchmark
        switch that lands in the middle of `report_rows`'s check-then-write.
        """

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.entered = threading.Event()
            self.release = threading.Event()
            self._paused = False

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if key == "current_benchmark" and not self._paused:
                self._paused = True
                self.entered.set()
                assert self.release.wait(timeout=5), "test setup never released the pause"
            return value

    ctx = _PausingContext({"current_benchmark": "gsm8k", "completed": 0, "total": 2})
    monkeypatch.setattr(runners, "_PROGRESS_CONTEXT", ctx)
    monkeypatch.setattr(runners, "_PROGRESS_ROWS", {})

    reporter = threading.Thread(
        target=runners.report_rows,
        kwargs=dict(
            rows_done=30, rows_total=30, scored=30, errored=0, passed=20,
            score_sum=20.0, for_benchmark="gsm8k",
        ),
    )
    reporter.start()
    assert ctx.entered.wait(timeout=5), "report_rows never reached its context check"

    switcher = threading.Thread(target=runners._write_progress, args=("mmlu", 1, 2))
    switcher.start()
    time.sleep(0.2)  # an unlocked _write_progress has time to fully interleave here
    ctx.release.set()
    reporter.join(timeout=5)
    switcher.join(timeout=5)
    assert not reporter.is_alive() and not switcher.is_alive()

    data = _read()
    assert data["current_benchmark"] == "mmlu"
    assert data["rows_done"] == 0, (
        "a report_rows call that read the old benchmark name must not be able to "
        "land its stale counts once _write_progress has switched to a new one"
    )
