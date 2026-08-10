import json
import time
from pathlib import Path

import pytest

from surogate_eval import runners
from surogate_eval.benchmarks.backends._evalscope_progress import ReviewWatcher, count_reviews
from surogate_eval.benchmarks.backends.evalscope_backend import EvalScopeBackend


def _read_progress():
    return json.loads(Path("eval_results/progress.json").read_text())


def _write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_evalscope_tracker(reviews_dir, processed_count, total_count):
    """Evalscope's own progress file: `<work_dir>/progress.json`, two levels
    above `reviews_dir` (`<work_dir>/reviews/<model_id>`). Not our
    `eval_results/progress.json` -- same filename, unrelated file.
    """
    path = reviews_dir.parent.parent / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": "running",
        "pipeline": "eval",
        "total_count": total_count,
        "processed_count": processed_count,
        "percent": round(processed_count / total_count * 100, 2) if total_count else None,
        "updated_at": "2026-08-10T00:00:00",
    }))


def _review(score):
    """The nesting evalscope actually writes: sample_score.score.value.<metric>."""
    return {"sample_score": {"score": {"value": {"acc": score}}}}


def _multi_key_review(**value):
    """A row with several numeric score keys, e.g. DROP's `em` + `f1` -- the
    shape `_review` cannot produce, since it only ever emits a single key."""
    return {"sample_score": {"score": {"value": value}}}


def _wait_until(condition, attempts=100, interval=0.02):
    """Poll ``condition`` up to ~2s (100 * 0.02s) of a 0.05s watcher cadence,
    for a background watcher tick to land."""
    for _ in range(attempts):
        if condition():
            return
        time.sleep(interval)


@pytest.fixture
def report_rows_calls(monkeypatch):
    """Captures every call ``_evalscope_progress`` makes to
    ``runners.report_rows``, in order, as ``(args, kwargs)`` pairs."""
    calls = []
    monkeypatch.setattr(
        "surogate_eval.benchmarks.backends._evalscope_progress.runners.report_rows",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def test_counts_lines_as_they_are_appended(tmp_path):
    """Evalscope appends one line per sample as it completes, so a count taken
    mid-run is a real row count."""
    f = tmp_path / "reviews" / "m" / "gsm8k_main.jsonl"
    _write_lines(f, [_review(1.0), _review(0.0)])

    assert count_reviews(tmp_path / "reviews" / "m", "gsm8k") == (2, 2, 1, 1.0)

    _write_lines(f, [_review(1.0)])

    assert count_reviews(tmp_path / "reviews" / "m", "gsm8k") == (3, 3, 2, 2.0)


def test_multi_key_score_picks_the_same_key_the_final_report_would(tmp_path):
    """`count_reviews` must read a multi-key row (DROP: `em` + `f1`) the same
    way `_extract_sample_score` does for the final report -- not just take
    the first key in dict order, which used to disagree with the report
    whenever the "first" key was not the one `main_score_name`/the fallback
    order would have picked.

    `f1` is inserted first here; the fallback order ranks `em` ahead of
    `f1`, so the correct read is `em`'s 0.3, not `f1`'s 0.7. 0.3 must also
    count as passed -- `score > 0`, the same rule `_review_row_to_record`
    uses -- not the old `score >= 0.5`, which 0.3 would fail.
    """
    f = tmp_path / "reviews" / "m" / "drop_main.jsonl"
    _write_lines(f, [_multi_key_review(f1=0.7, em=0.3)])

    rows_done, scored, passed, score_sum = count_reviews(
        tmp_path / "reviews" / "m", "drop",
    )

    assert rows_done == 1
    assert scored == 1
    assert score_sum == 0.3
    assert passed == 1, "0.3 > 0 must pass, matching _review_row_to_record's rule"


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


def test_watcher_reports_on_a_tick_and_stop_joins_the_thread(tmp_path, report_rows_calls):
    """The threading glue itself, not just count_reviews.

    The tests above only call ``count_reviews`` directly; none of them starts
    a watcher. This is the one check on ``start``/``_loop``/``stop`` wiring,
    so a broken cadence, a wrong argument order into ``report_rows``, or a
    ``stop()`` that fails to join cannot slip through unnoticed -- the exact
    "hangs or leaks a thread" failure mode the watcher exists to avoid.
    """
    _write_lines(tmp_path / "reviews" / "m" / "gsm8k_main.jsonl", [_review(1.0)])

    watcher = ReviewWatcher(
        lambda: tmp_path / "reviews" / "m", "gsm8k", benchmark_name="gsm8k",
        interval=0.05,
    )
    watcher.start()
    _wait_until(lambda: report_rows_calls)
    watcher.stop()

    assert report_rows_calls, "the watcher never reported a tick"
    args, kwargs = report_rows_calls[0]
    # (rows_done, rows_total, scored, passed, score_sum). No evalscope
    # tracker file here, so rows_total falls back to 0 (unknown).
    assert args == (1, 0, 1, 1, 1.0)
    assert kwargs == {"for_benchmark": "gsm8k"}, "every write must be tagged with its own benchmark"
    assert not watcher._thread.is_alive(), "stop() must join the watcher thread"


def test_watcher_follows_a_work_dir_that_moves_after_construction(tmp_path, report_rows_calls):
    """`run_task` appends a timestamp to `task_config.work_dir` *in place*
    after the watcher's reviews directory is computed (`setup_work_directory`,
    evalscope/run.py:56-58). A `reviews_dir` resolved once at construction
    time keeps pointing at the pre-timestamp directory forever, so the
    watcher must re-resolve it on every tick against whatever the live
    value is, not a snapshot taken up front. This is the exact shape of the
    real bug: the reviews directory only exists at its final location once
    the caller's mutable config object has already moved on.
    """
    class _MovingTaskConfig:
        """Stands in for evalscope's TaskConfig: work_dir mutates in place,
        exactly as setup_work_directory does when run_task starts."""

        def __init__(self, work_dir):
            self.work_dir = work_dir
            self.model_id = "m"

    task_config = _MovingTaskConfig(str(tmp_path / "pre_timestamp"))

    watcher = ReviewWatcher(
        reviews_dir=lambda: Path(task_config.work_dir) / "reviews" / task_config.model_id,
        dataset="gsm8k", benchmark_name="gsm8k", interval=0.05,
    )
    watcher.start()
    time.sleep(0.12)  # a couple of ticks against the pre-timestamp (nonexistent) dir

    # The equivalent of setup_work_directory renaming work_dir in place.
    task_config.work_dir = str(tmp_path / "20260810_132912")
    _write_lines(
        Path(task_config.work_dir) / "reviews" / "m" / "gsm8k_main.jsonl", [_review(1.0)],
    )

    _wait_until(lambda: report_rows_calls)
    watcher.stop()

    assert report_rows_calls, "the watcher never picked up the directory once work_dir moved"
    args, _ = report_rows_calls[0]
    assert args[0] == 1, "rows_done once the moved directory is followed"


def test_tracker_file_supplies_rows_done_and_total_overriding_review_count(tmp_path, report_rows_calls):
    """`enable_progress_tracker` (set in `_prepare_task_config`) makes evalscope
    write its own `<work_dir>/progress.json` with a real processed_count/
    total_count. Once it is there, it replaces the reviews line count for
    rows_done and the old `config['limit']` guess for rows_total -- scored/
    passed/score_sum still come from the reviews file, since the tracker
    carries no per-sample outcome at all.
    """
    reviews_dir = tmp_path / "reviews" / "m"
    # Two review lines: if rows_done still came from this count, it would
    # report 2. The tracker instead says 7 processed of 20 total.
    _write_lines(reviews_dir / "gsm8k_main.jsonl", [_review(1.0), _review(0.0)])
    _write_evalscope_tracker(reviews_dir, processed_count=7, total_count=20)

    watcher = ReviewWatcher(
        lambda: reviews_dir, "gsm8k", benchmark_name="gsm8k", interval=0.05,
    )
    watcher.start()
    _wait_until(lambda: report_rows_calls)
    watcher.stop()

    assert report_rows_calls, "the watcher never reported a tick"
    args, _ = report_rows_calls[0]
    # (rows_done, rows_total, scored, passed, score_sum)
    assert args == (7, 20, 2, 1, 1.0), "rows_done/rows_total from the tracker; scores still from the reviews file"


def test_tracker_file_absent_falls_back_to_review_count_with_total_zero(tmp_path, report_rows_calls):
    """No evalscope tracker file at all -- an older evalscope, or a future
    release that drops the flag. rows_done falls back to the reviews line
    count, same as before this change; rows_total falls back to 0 (unknown),
    since the old `config['limit']` guess it used to carry is gone for good.
    """
    reviews_dir = tmp_path / "reviews" / "m"
    _write_lines(reviews_dir / "gsm8k_main.jsonl", [_review(1.0)])
    # No progress.json anywhere under tmp_path.

    watcher = ReviewWatcher(
        lambda: reviews_dir, "gsm8k", benchmark_name="gsm8k", interval=0.05,
    )
    watcher.start()
    _wait_until(lambda: report_rows_calls)
    watcher.stop()

    assert report_rows_calls, "the watcher never reported a tick"
    args, _ = report_rows_calls[0]
    assert args == (1, 0, 1, 1, 1.0)


def test_malformed_tracker_file_falls_back_without_raising(tmp_path, report_rows_calls):
    """A tracker file caught mid-write must fall back exactly like an absent
    one -- never raise, never stall the watcher. Evalscope writes it via
    temp-file-then-``os.replace``, but a reader is not guaranteed to never
    see a half-written file (e.g. non-atomic renames on some filesystems),
    so this reader must survive one anyway.
    """
    reviews_dir = tmp_path / "reviews" / "m"
    _write_lines(reviews_dir / "gsm8k_main.jsonl", [_review(1.0)])
    tracker_path = reviews_dir.parent.parent / "progress.json"
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_path.write_text('{"status": "running", "processed_count": 5, "tot')  # truncated mid-write

    watcher = ReviewWatcher(
        lambda: reviews_dir, "gsm8k", benchmark_name="gsm8k", interval=0.05,
    )
    watcher.start()
    _wait_until(lambda: report_rows_calls)
    watcher.stop()

    assert report_rows_calls, "a malformed tracker file must not stop the watcher from reporting"
    args, _ = report_rows_calls[0]
    assert args == (1, 0, 1, 1, 1.0), "same fallback as an absent tracker file"


def test_tracker_file_is_read_from_the_post_mutation_work_dir(tmp_path, report_rows_calls):
    """Same relocation as `test_watcher_follows_a_work_dir_that_moves_after_
    construction`, but for evalscope's tracker file: it must be resolved
    from the same live `reviews_dir` the watcher already re-reads every
    tick, not a path fixed at construction time.
    """
    class _MovingTaskConfig:
        """Stands in for evalscope's TaskConfig: work_dir mutates in place,
        exactly as setup_work_directory does when run_task starts."""

        def __init__(self, work_dir):
            self.work_dir = work_dir
            self.model_id = "m"

    task_config = _MovingTaskConfig(str(tmp_path / "pre_timestamp"))

    watcher = ReviewWatcher(
        reviews_dir=lambda: Path(task_config.work_dir) / "reviews" / task_config.model_id,
        dataset="gsm8k", benchmark_name="gsm8k", interval=0.05,
    )
    watcher.start()
    time.sleep(0.12)  # a couple of ticks against the pre-timestamp (nonexistent) dir

    # The equivalent of setup_work_directory renaming work_dir in place.
    task_config.work_dir = str(tmp_path / "20260810_132912")
    moved_reviews_dir = Path(task_config.work_dir) / "reviews" / "m"
    _write_lines(moved_reviews_dir / "gsm8k_main.jsonl", [_review(1.0)])
    _write_evalscope_tracker(moved_reviews_dir, processed_count=9, total_count=42)

    _wait_until(lambda: report_rows_calls)
    watcher.stop()

    assert report_rows_calls, "the watcher never picked up the tracker file once work_dir moved"
    args, _ = report_rows_calls[0]
    assert args == (9, 42, 1, 1, 1.0), "rows_done/rows_total from the tracker at the moved work_dir"


def test_report_rows_is_a_no_op_once_the_context_has_moved_past_it(tmp_path, monkeypatch):
    """A write tagged for a benchmark that is no longer current must not
    apply. This is what turns a leaked ``ReviewWatcher``'s stale tick (its
    ``stop()`` join can time out while a tick is mid-flight, per the report
    that prompted this test) into a silent no-op instead of a progress bar
    that jumps backwards onto the next benchmark's name.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runners, "_PROGRESS",
        {
            "current_benchmark": "", "completed": 0, "total": 1,
            "rows_done": 0, "rows_total": 0, "scored": 0, "errored": 0,
            "passed": 0, "score_sum": 0.0,
        },
    )

    runners._write_progress("gsm8k", 0, 2)
    runners.report_rows(
        rows_done=5, rows_total=10, scored=5, passed=3, score_sum=3.0,
        for_benchmark="gsm8k",
    )
    assert _read_progress()["rows_done"] == 5, "must write while for_benchmark matches the current context"

    runners._write_progress("mmlu", 1, 2)  # the next benchmark starts, and zeroes the row block
    runners.report_rows(
        rows_done=99, rows_total=10, scored=99, passed=99, score_sum=99.0,
        for_benchmark="gsm8k",  # stale tag: the context has already moved on
    )

    after = _read_progress()
    assert after["current_benchmark"] == "mmlu"
    assert after["rows_done"] == 0, "a stale for_benchmark write must be a no-op, not resurrect the row block"


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
        runners, "_PROGRESS",
        {
            "current_benchmark": "", "completed": 0, "total": 1,
            "rows_done": 0, "rows_total": 0, "scored": 0, "errored": 0,
            "passed": 0, "score_sum": 0.0,
        },
    )

    runners._write_progress("gsm8k", 0, 2)
    _write_lines(tmp_path / "reviews" / "m" / "gsm8k_main.jsonl", [_review(1.0)])

    watcher = ReviewWatcher(
        lambda: tmp_path / "reviews" / "m", "gsm8k", benchmark_name="gsm8k",
        interval=0.05,
    )
    watcher.start()
    _wait_until(lambda: _read_progress().get("rows_done"))  # wait for a real tick to land
    assert _read_progress()["rows_done"] == 1

    # The next benchmark starts while this watcher is still alive -- the
    # leaked-thread scenario a timed-out join allows.
    runners._write_progress("mmlu", 1, 2)
    time.sleep(0.15)  # a few more 0.05s ticks of the still-running watcher
    watcher.stop()

    after = _read_progress()
    assert after["current_benchmark"] == "mmlu"
    assert after["rows_done"] == 0, "a stale watcher must not resurrect the old benchmark's row block"


def test_watcher_publishes_coherent_progress_when_the_tracker_lags_the_reviews_file(tmp_path, monkeypatch):
    """End-to-end regression for Finding 1, through the real (unmocked)
    ``report_rows``: evalscope's tracker file flushes at most once a second
    and so reliably lags the reviews JSONL, which is appended to before the
    tracker is updated. That lag must never surface as a negative errored
    count or a rows_done smaller than scored.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runners, "_PROGRESS",
        {
            "current_benchmark": "", "completed": 0, "total": 1,
            "rows_done": 0, "rows_total": 0, "scored": 0, "errored": 0,
            "passed": 0, "score_sum": 0.0,
        },
    )
    runners._write_progress("gsm8k", 0, 1)

    reviews_dir = tmp_path / "reviews" / "m"
    # 5 rows already scored, but the tracker (flushed at most once a second)
    # still reports only 3 processed -- the reviews file is reliably ahead
    # of the tracker on evalscope's own write order.
    _write_lines(reviews_dir / "gsm8k_main.jsonl", [_review(1.0)] * 5)
    _write_evalscope_tracker(reviews_dir, processed_count=3, total_count=20)

    watcher = ReviewWatcher(lambda: reviews_dir, "gsm8k", benchmark_name="gsm8k", interval=0.05)
    watcher.start()
    _wait_until(lambda: _read_progress().get("scored"))
    watcher.stop()

    data = _read_progress()
    assert data["scored"] == 5
    assert data["errored"] >= 0, "errored must never go negative"
    assert data["rows_done"] >= data["scored"], "rows_done must never show fewer rows than scored"


class _FakeTarget:
    name = "t1"
    config = {"model": "m", "base_url": "https://target.example", "api_key": "k"}


class _RaisesOnInit:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("boom: could not resolve reviews_dir")


class _RaisesOnStart:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        raise RuntimeError("can't start new thread")

    def stop(self):  # pragma: no cover - must never be reached
        raise AssertionError("stop() must not run on a watcher that never started")


@pytest.mark.parametrize("fake_watcher_cls", [_RaisesOnInit, _RaisesOnStart])
def test_a_broken_progress_watcher_does_not_fail_an_otherwise_healthy_benchmark(
    monkeypatch, fake_watcher_cls,
):
    """Row-progress reporting is a nice-to-have layered onto a benchmark run,
    never something that can take the run down with it. A bad limit, a
    missing work_dir, or a thread the OS refuses to start must all be
    swallowed, and a watcher that never started must never be stopped.
    """
    monkeypatch.setattr(
        "surogate_eval.benchmarks.backends.evalscope_backend.run_task",
        lambda task_cfg: {},
    )
    monkeypatch.setattr(
        "surogate_eval.benchmarks.backends._evalscope_progress.ReviewWatcher",
        fake_watcher_cls,
    )

    backend = EvalScopeBackend()
    result = backend.evaluate(_FakeTarget(), "gsm8k", {})

    assert result["metadata"]["backend"] == "evalscope"
