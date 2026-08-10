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
from typing import Callable, List, Optional, Tuple

from surogate_eval import runners
from surogate_eval.utils.logger import get_logger
from .evalscope_backend import _extract_sample_score

logger = get_logger()


def _sibling_dataset_prefixes(dataset: str) -> frozenset:
    """Other registered evalscope dataset ids that start with ``dataset + '_'``.

    A plain ``{dataset}_*.jsonl`` glob is not enough to isolate one dataset's
    files: fnmatch's ``*`` does not stop at the next underscore, so it also
    matches a longer, independently-registered dataset that happens to start
    with the same prefix -- ``mmmu_*`` matches ``mmmu_pro_val.jsonl`` too,
    because "pro_val" satisfies ``*`` just as well as a real mmmu subset
    name would. Whether "pro" is a separate dataset or just an unusual
    subset name cannot be told from string shape alone; evalscope's own
    benchmark registry is the only place that distinction actually lives.
    Best-effort: if the registry cannot be read, fail open to "no known
    siblings" rather than raise.
    """
    try:
        from evalscope.api.registry import BENCHMARK_REGISTRY

        prefix = f"{dataset}_"
        return frozenset(name for name in BENCHMARK_REGISTRY if name.startswith(prefix))
    except Exception:
        return frozenset()


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


def count_reviews(reviews_dir: Path, dataset: str) -> Tuple[int, int, int, float]:
    """(rows_done, scored, passed, score_sum) from the reviews JSONL so far.

    ``rows_done`` counts every line, including one we could not read a score
    from: the sample was processed either way, and a progress bar that stalls
    on an unreadable row is worse than one that advances.

    The score itself is read with ``evalscope_backend._extract_sample_score``
    -- the same picker the final report uses in ``_review_row_to_record`` --
    and a row passes on ``score > 0``, the same rule ``_review_row_to_record``
    applies. A row with several numeric score keys (DROP carries both ``em``
    and ``f1``) used to be able to read a different score, and pass/fail
    differently, live versus in the final report; picking the same key by
    the same rule and passing by the same threshold makes that impossible.
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
                    try:
                        row = json.loads(line)
                        sample_score = row.get("sample_score")
                        if not isinstance(sample_score, dict):
                            sample_score = {}
                        score, _, _ = _extract_sample_score(sample_score.get("score") or {})
                    except Exception:
                        continue
                    if score is None:
                        continue
                    scored += 1
                    score_sum += score
                    if score > 0:
                        passed += 1
    except Exception:
        logger.debug("review count failed", exc_info=True)
    return (rows_done, scored, passed, score_sum)


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
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            reviews_dir = self._reviews_dir()
            rows_done, scored, passed, score_sum = count_reviews(
                reviews_dir, self._dataset,
            )
            # The reviews file still supplies scores -- evalscope's tracker
            # carries no per-sample outcome, only counts -- but once it has
            # written, its processed_count/total_count are a real fact and
            # replace the reviews line count (a proxy for rows_done) and the
            # old config['limit'] guess (gone; see _prepare_task_config).
            tracker = read_evalscope_tracker(reviews_dir)
            if tracker is not None:
                rows_done, rows_total = tracker
            else:
                rows_total = 0  # unknown: no fact available yet, no guess left to fall back to
            if rows_done:
                # for_benchmark: stop()'s join(timeout=5) can return while
                # this tick is still running. Tagging the write means a tick
                # that only finishes after the next benchmark's watcher has
                # already started gets silently dropped by report_rows
                # instead of stamping this benchmark's counts over it.
                runners.report_rows(
                    rows_done, rows_total, scored,
                    rows_done - scored, passed, score_sum,
                    for_benchmark=self._benchmark_name,
                )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
