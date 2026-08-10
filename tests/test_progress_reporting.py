import json
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


def test_a_new_benchmark_clears_the_previous_rows():
    """Otherwise benchmark 2 inherits benchmark 1's row counts and the UI
    shows 30/30 the moment the next benchmark starts."""
    runners._write_progress("gsm8k", 0, 2)
    runners.report_rows(
        rows_done=30, rows_total=30, scored=30, errored=0, passed=20, score_sum=20.0,
    )

    runners._write_progress("mmlu", 1, 2)

    data = _read()
    assert data["current_benchmark"] == "mmlu"
    assert "rows_done" not in data
    assert "scored" not in data


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
