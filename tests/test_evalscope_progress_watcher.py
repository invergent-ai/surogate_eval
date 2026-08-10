import json
import time
from pathlib import Path

from surogate_eval import runners
from surogate_eval.benchmarks.backends._evalscope_progress import ReviewWatcher, count_reviews


def _read_progress():
    return json.loads(Path("eval_results/progress.json").read_text())


def _write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _review(score):
    """The nesting evalscope actually writes: sample_score.score.value.<metric>."""
    return {"sample_score": {"score": {"value": {"acc": score}}}}


def test_counts_lines_as_they_are_appended(tmp_path):
    """Evalscope appends one line per sample as it completes, so a count taken
    mid-run is a real row count."""
    f = tmp_path / "reviews" / "m" / "gsm8k_main.jsonl"
    _write_lines(f, [_review(1.0), _review(0.0)])

    assert count_reviews(tmp_path / "reviews" / "m", "gsm8k") == (2, 2, 1, 1.0)

    _write_lines(f, [_review(1.0)])

    assert count_reviews(tmp_path / "reviews" / "m", "gsm8k") == (3, 3, 2, 2.0)


def test_a_malformed_line_does_not_stop_the_count(tmp_path):
    """One truncated line, which is exactly what a mid-write read produces,
    must cost itself and not the rows around it."""
    f = tmp_path / "reviews" / "m" / "gsm8k_main.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(_review(1.0)) + "\n" + '{"sample_sc' + "\n")

    rows_done, scored, passed, score_sum = count_reviews(
        tmp_path / "reviews" / "m", "gsm8k",
    )

    assert rows_done == 2
    assert scored == 1
    assert passed == 1
    assert score_sum == 1.0


def test_a_missing_directory_is_no_progress_yet_not_an_error(tmp_path):
    """The reviews directory does not exist until the first sample lands."""
    assert count_reviews(tmp_path / "nope", "gsm8k") == (0, 0, 0, 0.0)


def test_the_subset_suffix_does_not_match_a_prefix_sibling(tmp_path):
    """`mmmu_*` must not pick up `mmmu_pro_*`, the same trap `_load_predictions`
    guards with its underscore glob."""
    base = tmp_path / "reviews" / "m"
    _write_lines(base / "mmmu_val.jsonl", [_review(1.0)])
    _write_lines(base / "mmmu_pro_val.jsonl", [_review(1.0), _review(1.0)])

    assert count_reviews(base, "mmmu")[0] == 1


def test_watcher_reports_on_a_tick_and_stop_joins_the_thread(tmp_path, monkeypatch):
    """The threading glue itself, not just count_reviews.

    The tests above only call ``count_reviews`` directly; none of them starts
    a watcher. This is the one check on ``start``/``_loop``/``stop`` wiring,
    so a broken cadence, a wrong argument order into ``report_rows``, or a
    ``stop()`` that fails to join cannot slip through unnoticed -- the exact
    "hangs or leaks a thread" failure mode the watcher exists to avoid.
    """
    calls = []
    monkeypatch.setattr(
        "surogate_eval.benchmarks.backends._evalscope_progress.runners.report_rows",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    _write_lines(tmp_path / "reviews" / "m" / "gsm8k_main.jsonl", [_review(1.0)])

    watcher = ReviewWatcher(
        tmp_path / "reviews" / "m", "gsm8k", benchmark_name="gsm8k",
        rows_total=10, interval=0.05,
    )
    watcher.start()
    for _ in range(100):  # up to ~2s of a 0.05s cadence
        if calls:
            break
        time.sleep(0.02)
    watcher.stop()

    assert calls, "the watcher never reported a tick"
    args, kwargs = calls[0]
    # (rows_done, rows_total, scored, errored, passed, score_sum)
    assert args == (1, 10, 1, 0, 1, 1.0)
    assert kwargs == {"for_benchmark": "gsm8k"}, "every write must be tagged with its own benchmark"
    assert not watcher._thread.is_alive(), "stop() must join the watcher thread"


def test_report_rows_is_a_no_op_once_the_context_has_moved_past_it(tmp_path, monkeypatch):
    """A write tagged for a benchmark that is no longer current must not
    apply. This is what turns a leaked ``ReviewWatcher``'s stale tick (its
    ``stop()`` join can time out while a tick is mid-flight, per the report
    that prompted this test) into a silent no-op instead of a progress bar
    that jumps backwards onto the next benchmark's name.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runners, "_PROGRESS_CONTEXT",
        {"current_benchmark": "", "completed": 0, "total": 1},
    )
    monkeypatch.setattr(runners, "_PROGRESS_ROWS", {})

    runners._write_progress("gsm8k", 0, 2)
    runners.report_rows(
        rows_done=5, rows_total=10, scored=5, errored=0, passed=3, score_sum=3.0,
        for_benchmark="gsm8k",
    )
    assert _read_progress()["rows_done"] == 5, "must write while for_benchmark matches the current context"

    runners._write_progress("mmlu", 1, 2)  # the next benchmark starts, and clears the row block
    runners.report_rows(
        rows_done=99, rows_total=10, scored=99, errored=0, passed=99, score_sum=99.0,
        for_benchmark="gsm8k",  # stale tag: the context has already moved on
    )

    after = _read_progress()
    assert after["current_benchmark"] == "mmlu"
    assert "rows_done" not in after, "a stale for_benchmark write must be a no-op, not resurrect the row block"


def test_a_stale_watcher_cannot_overwrite_the_next_benchmarks_row_block(tmp_path, monkeypatch):
    """The leaked-thread scenario end to end, through the real watcher.

    ``stop()``'s ``join(timeout=5)`` can return while a tick is still in
    flight; the thread stays alive and ticks again before it notices
    ``_stop`` is set. That next tick must not be able to stamp this
    benchmark's counts onto the row block once the next benchmark has
    already started.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runners, "_PROGRESS_CONTEXT",
        {"current_benchmark": "", "completed": 0, "total": 1},
    )
    monkeypatch.setattr(runners, "_PROGRESS_ROWS", {})

    runners._write_progress("gsm8k", 0, 2)
    _write_lines(tmp_path / "reviews" / "m" / "gsm8k_main.jsonl", [_review(1.0)])

    watcher = ReviewWatcher(
        tmp_path / "reviews" / "m", "gsm8k", benchmark_name="gsm8k",
        rows_total=10, interval=0.05,
    )
    watcher.start()
    for _ in range(100):  # wait for a real tick to land
        if _read_progress().get("rows_done"):
            break
        time.sleep(0.02)
    assert _read_progress()["rows_done"] == 1

    # The next benchmark starts while this watcher is still alive -- the
    # leaked-thread scenario a timed-out join allows.
    runners._write_progress("mmlu", 1, 2)
    time.sleep(0.15)  # a few more 0.05s ticks of the still-running watcher
    watcher.stop()

    after = _read_progress()
    assert after["current_benchmark"] == "mmlu"
    assert "rows_done" not in after, "a stale watcher must not resurrect the old benchmark's row block"
