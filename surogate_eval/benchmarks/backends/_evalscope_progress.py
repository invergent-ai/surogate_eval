"""Live row counts for evalscope-backed benchmarks.

`run_task()` blocks, so the backend cannot report from the call itself.
Evalscope appends one JSON line per sample to
``reviews/{model_id}/{benchmark}_{subset}.jsonl`` as each completes -- its
evaluator persists results through a per-item `on_result` callback that
appends with ``DumpMode.APPEND``. Counting those lines is therefore a real
row count, and ``count_reviews`` still uses it -- but only for scores now.

For rows_done/rows_total, ``read_evalscope_tracker`` instead reads
evalscope's *own* progress file, written by its ``ProgressTracker`` at
``<work_dir>/progress.json`` when ``TaskConfig.enable_progress_tracker`` is
on (see ``_prepare_task_config``). Do not confuse the two: that is a
different file from *our* ``eval_results/progress.json``
(``runners._flush_progress``) -- same filename, unrelated file, unrelated
directory.

That layout is a library detail rather than a documented contract, so every
read here is defensive: a missing directory or file is "no progress yet",
and a malformed line or a malformed tracker file costs itself and nothing
else.
"""

import json
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from surogate_eval import runners
from surogate_eval.utils.logger import get_logger
from .evalscope_backend import EvalScopeBackend, _extract_sample_score

logger = get_logger()


def _sibling_dataset_prefixes(dataset: str) -> frozenset:
    """Other datasets this backend can run whose evalscope id starts with
    ``dataset + '_'``.

    A plain ``{dataset}_*.jsonl`` glob is not enough to isolate one dataset's
    files: fnmatch's ``*`` does not stop at the next underscore, so it also
    matches a longer, independently-registered dataset that happens to start
    with the same prefix -- ``mmmu_*`` matches ``mmmu_pro_val.jsonl`` too,
    because "pro_val" satisfies ``*`` just as well as a real mmmu subset
    name would. Whether "pro" is a separate dataset or just an unusual
    subset name cannot be told from string shape alone.

    Built from ``EvalScopeBackend.BENCHMARK_MAP``'s values -- this
    codebase's own complete list of evalscope dataset ids it will ever run
    -- rather than evalscope's own benchmark registry: a registry import
    that failed used to fail open to "no known siblings", silently
    reintroducing the prefix-collision bug this function exists to prevent.
    ``BENCHMARK_MAP`` is a plain dict literal in this codebase, so reading
    it cannot fail the way an optional import can, and it is strictly more
    precise anyway -- it is exactly the set of datasets a match here could
    ever actually collide with.
    """
    prefix = f"{dataset}_"
    return frozenset(
        name for name in EvalScopeBackend.BENCHMARK_MAP.values() if name.startswith(prefix)
    )


def drop_sibling_matches(paths: List[Path], dataset: str) -> List[Path]:
    """Remove glob matches that actually belong to a longer sibling dataset.

    Shared by ``resolve_review_paths`` here and
    ``EvalScopeBackend._load_predictions``, which globs the same directory
    the same way and is exposed to the same trap.
    """
    siblings = _sibling_dataset_prefixes(dataset)
    if not siblings:
        return paths
    return [
        p for p in paths
        if not any(p.name == f"{sib}.jsonl" or p.name.startswith(f"{sib}_") for sib in siblings)
    ]


def resolve_review_paths(reviews_dir: Path, dataset: str) -> List[Path]:
    """The review file(s) for one dataset in ``reviews_dir``.

    Underscore-glob first (``{dataset}_*.jsonl``, siblings dropped), and the
    bare ``{dataset}.jsonl`` only as a fallback when that glob matched
    nothing. Shared by ``count_reviews`` here and
    ``EvalScopeBackend._load_predictions`` so the two cannot again resolve a
    dataset's reviews to different files -- they used to apply different
    merge rules (this one always unioned the bare file in;
    ``_load_predictions`` used it only as a fallback) and so could disagree
    on which rows count. ``_load_predictions``'s rule wins here because the
    final report is the authority.
    """
    matches = drop_sibling_matches(
        list(reviews_dir.glob(f"{dataset}_*.jsonl")), dataset,
    )
    if not matches:
        exact = reviews_dir / f"{dataset}.jsonl"
        if exact.exists():
            matches = [exact]
    return matches


def _read_line_score(line: str) -> Optional[float]:
    """The score on one reviews-file line, or ``None`` if it could not be
    read.

    Shared by ``count_reviews`` (a fresh full pass) and ``ReviewCounter``
    (incremental) so the two cannot drift on what counts as a readable row.
    Reads with ``evalscope_backend._extract_sample_score`` -- the same
    picker the final report uses in ``_review_row_to_record`` -- so a row
    with several numeric score keys (DROP carries both ``em`` and ``f1``)
    cannot read a different score, live versus in the final report.
    """
    try:
        row = json.loads(line)
        sample_score = row.get("sample_score")
        if not isinstance(sample_score, dict):
            sample_score = {}
        score, _, _ = _extract_sample_score(sample_score.get("score") or {})
        return score
    except Exception:
        return None


def count_reviews(reviews_dir: Path, dataset: str) -> Tuple[int, int, int, float]:
    """(rows_done, scored, passed, score_sum) from the reviews JSONL so far.

    A fresh, full pass over every matching file every call -- correct, and
    fine for a one-off count, but not what the watcher's own tick uses (see
    ``ReviewCounter``, which tracks incrementally instead of re-reading the
    whole file on the cadence this function would require).

    ``rows_done`` counts every line, including one we could not read a score
    from: the sample was processed either way, and a progress bar that stalls
    on an unreadable row is worse than one that advances. A row passes on
    ``score > 0``, the same rule ``_review_row_to_record`` applies.
    """
    rows_done = scored = passed = 0
    score_sum = 0.0
    try:
        if not reviews_dir.is_dir():
            return (0, 0, 0, 0.0)
        for path in resolve_review_paths(reviews_dir, dataset):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows_done += 1
                    score = _read_line_score(line)
                    if score is None:
                        continue
                    scored += 1
                    score_sum += score
                    if score > 0:
                        passed += 1
    except Exception:
        logger.debug("review count failed", exc_info=True)
    return (rows_done, scored, passed, score_sum)


class ReviewCounter:
    """Incremental, stateful counterpart to ``count_reviews`` for a watcher
    that ticks repeatedly against the same growing files.

    ``count_reviews`` re-reads and re-parses every matching file, in full,
    on every call. On a 14k-row benchmark that is roughly a 40MB re-parse
    every couple of seconds, in a GIL-holding background thread, for the
    life of the run -- expensive enough that it is what can make a tick
    outlive ``ReviewWatcher.stop()``'s 5s join. This class instead tracks a
    byte offset (and any trailing partial line) per file, so a tick after
    the first only reads and scores the bytes appended since the previous
    one, while returning the same running-total shape ``count_reviews``
    does.

    A trailing line with no ``\\n`` yet is a real possibility, not an edge
    case: the file is being appended to concurrently by evalscope while this
    reads it, so the last line on disk may be half-written. That line is
    held rather than counted or discarded, and is re-read -- prefixed onto
    whatever is read next -- until a tick sees it complete. One instance is
    scoped to one benchmark's watch (a fresh ``ReviewWatcher`` per
    benchmark constructs a fresh counter), so offsets never leak from one
    benchmark's files into the next.
    """

    def __init__(self) -> None:
        self._offsets: Dict[Path, int] = {}
        self._partial: Dict[Path, bytes] = {}
        self._rows_done = 0
        self._scored = 0
        self._passed = 0
        self._score_sum = 0.0

    def update(self, reviews_dir: Path, dataset: str) -> Tuple[int, int, int, float]:
        """Advance the running totals with whatever is newly available, and
        return them. Best-effort like ``count_reviews``: any failure costs
        only this call, never raises, and never rolls back what was already
        counted.
        """
        try:
            if reviews_dir.is_dir():
                for path in resolve_review_paths(reviews_dir, dataset):
                    self._consume(path)
        except Exception:
            logger.debug("incremental review count failed", exc_info=True)
        return (self._rows_done, self._scored, self._passed, self._score_sum)

    def _consume(self, path: Path) -> None:
        offset = self._offsets.get(path, 0)
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
        except Exception:
            return
        if not chunk:
            return
        buf = self._partial.pop(path, b"") + chunk
        lines = buf.split(b"\n")
        # The last element is either b"" (the chunk ended exactly on a
        # newline) or a line still being written -- either way, not ours to
        # count yet. Held for the next tick rather than counted or dropped.
        # The file offset still advances past it: those bytes have already
        # been read into `_partial` and must not be re-read from disk next
        # time, or they would be double-counted once the line completes.
        trailing = lines[-1]
        self._partial[path] = trailing
        self._offsets[path] = offset + len(chunk)
        for raw_line in lines[:-1]:
            self._count_line(raw_line)

    def _count_line(self, raw_line: bytes) -> None:
        stripped = raw_line.strip()
        if not stripped:
            return
        self._rows_done += 1
        try:
            text = stripped.decode("utf-8")
        except UnicodeDecodeError:
            return
        score = _read_line_score(text)
        if score is None:
            return
        self._scored += 1
        self._score_sum += score
        if score > 0:
            self._passed += 1


def read_evalscope_tracker(reviews_dir: Path) -> Optional[Tuple[int, int]]:
    """(rows_done, rows_total) from evalscope's own progress tracker, or
    ``None`` when it is not there to read.

    ``TaskConfig.enable_progress_tracker=True`` (set in
    ``_prepare_task_config``) makes evalscope write its *own*
    ``<work_dir>/progress.json`` -- not our ``eval_results/progress.json``
    -- with a real ``processed_count``/``total_count`` computed from the
    dataset and any configured limit. ``reviews_dir`` is
    ``<work_dir>/reviews/<model_id>``, so that file sits two levels up from
    it -- the same ``work_dir`` the reviews directory is already resolved
    from each tick, not a second independently-tracked path.

    Best-effort like ``count_reviews``: the tracker may not have written
    yet, a reader can catch its temp-then-``os.replace`` mid-flight, or a
    future evalscope release may drop the flag entirely -- any of that
    reads as "not available yet", not an error.
    """
    try:
        tracker_path = reviews_dir.parent.parent / "progress.json"
        data = json.loads(tracker_path.read_text())
        return (int(data["processed_count"]), int(data["total_count"]))
    except Exception:
        logger.debug("evalscope progress-tracker read failed", exc_info=True)
        return None


class ReviewWatcher:
    """Publish row progress for a running evalscope benchmark.

    Started when the benchmark begins and stopped before the next one is
    dispatched, so a late write can never stamp one benchmark's counts onto
    the next one's name.
    """

    def __init__(
        self,
        reviews_dir: Callable[[], Path],
        dataset: str,
        benchmark_name: str,
        interval: float = 2.0,
    ) -> None:
        # A callable, not a path: evalscope's TaskConfig.work_dir gets a
        # timestamp appended *in place* by setup_work_directory once
        # run_task starts, after the caller builds this watcher. Resolving
        # once at construction would capture the pre-timestamp directory
        # forever, so every tick calls this again for the live value --
        # read_evalscope_tracker rides along on that same resolved Path.
        self._reviews_dir = reviews_dir
        self._dataset = dataset
        self._benchmark_name = benchmark_name
        self._interval = interval
        # Incremental, not count_reviews: a full re-parse of a 14k-row
        # benchmark's reviews file every couple of seconds, in this
        # GIL-holding background thread, is expensive enough to be what
        # makes a tick outlive stop()'s join. One counter per watcher, so
        # its byte offsets are scoped to this benchmark's files only.
        self._counter = ReviewCounter()
        # Guards a _tick() call: stop() runs one more tick itself (see
        # below), on the caller's thread, which can otherwise overlap with
        # the background thread's own in-flight tick -- both would read and
        # mutate self._counter's offsets/totals at once. A tick is a couple
        # of small file reads every few seconds, so serializing them costs
        # nothing real.
        self._tick_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _tick(self) -> None:
        """One count-and-publish cycle: called on the background thread's
        own cadence, and once more by stop() to grab the final state before
        the thread is joined. Never raises -- stop() calls this directly, on
        whatever thread stop() itself runs on, and progress reporting must
        never be able to fail the run it is layered onto.
        """
        try:
            with self._tick_lock:
                reviews_dir = self._reviews_dir()
                rows_done, scored, passed, score_sum = self._counter.update(
                    reviews_dir, self._dataset,
                )
                # The reviews file still supplies scores -- evalscope's
                # tracker carries no per-sample outcome, only counts -- but
                # once it has written, its processed_count/total_count are a
                # real fact and replace the reviews line count (a proxy for
                # rows_done) and the old config['limit'] guess (gone; see
                # _prepare_task_config).
                tracker = read_evalscope_tracker(reviews_dir)
                if tracker is not None:
                    rows_done, rows_total = tracker
                else:
                    rows_total = 0  # unknown: no fact available yet, no guess left to fall back to
                if rows_done:
                    # for_benchmark: stop()'s join(timeout=5) can return
                    # while this tick is still running. Tagging the write
                    # means a tick that only finishes after the next
                    # benchmark's watcher has already started gets silently
                    # dropped by report_rows instead of stamping this
                    # benchmark's counts over it.
                    runners.report_rows(
                        rows_done, rows_total, scored, passed, score_sum,
                        for_benchmark=self._benchmark_name,
                    )
        except Exception:
            logger.debug("progress tick failed", exc_info=True)

    def _loop(self) -> None:
        # Promptly, not after a full interval: a benchmark that finishes
        # inside one interval would otherwise never get a single mid-run
        # tick before stop() runs, and stop()'s own final tick (below) is
        # what covers the rest of the gap.
        self._tick()
        while not self._stop.wait(self._interval):
            self._tick()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # A final count before joining: by the time evalscope_backend calls
        # stop(), run_task has already returned and the reviews file is at
        # its final state, so this is the truly last word -- not whatever
        # the last periodic tick happened to see. Without it, the last
        # published number is always a little short, and a benchmark that
        # finishes inside one interval publishes nothing at all before the
        # next benchmark's _write_progress zeroes the row block.
        self._tick()
        self._thread.join(timeout=5)
