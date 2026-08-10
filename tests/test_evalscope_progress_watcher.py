import json
import time

from surogate_eval.benchmarks.backends._evalscope_progress import ReviewWatcher, count_reviews


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
        lambda *args: calls.append(args),
    )
    _write_lines(tmp_path / "reviews" / "m" / "gsm8k_main.jsonl", [_review(1.0)])

    watcher = ReviewWatcher(
        tmp_path / "reviews" / "m", "gsm8k", rows_total=10, interval=0.05,
    )
    watcher.start()
    for _ in range(100):  # up to ~2s of a 0.05s cadence
        if calls:
            break
        time.sleep(0.02)
    watcher.stop()

    assert calls, "the watcher never reported a tick"
    # (rows_done, rows_total, scored, errored, passed, score_sum)
    assert calls[0] == (1, 10, 1, 0, 1, 1.0)
    assert not watcher._thread.is_alive(), "stop() must join the watcher thread"
