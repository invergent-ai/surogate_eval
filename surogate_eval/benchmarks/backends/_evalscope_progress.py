"""Live row counts for evalscope-backed benchmarks.

`run_task()` blocks, so the backend cannot report from the call itself.
Evalscope appends one JSON line per sample to
``reviews/{model_id}/{benchmark}_{subset}.jsonl`` as each completes -- its
evaluator persists results through a per-item `on_result` callback that
appends with ``DumpMode.APPEND``. Counting those lines is therefore a real
row count.

That layout is a library detail rather than a documented contract, so every
read here is defensive: a missing directory is "no progress yet", and a
malformed line costs itself and nothing else.
"""

import json
import threading
from pathlib import Path
from typing import List, Tuple

from surogate_eval import runners
from surogate_eval.utils.logger import get_logger

logger = get_logger()


def _score_of(row: dict):
    """Pull the score out of evalscope's nesting, or None if unreadable."""
    value = (
        (row.get("sample_score") or {}).get("score") or {}
    ).get("value")
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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

    Shared by ``count_reviews`` here and ``EvalScopeBackend._load_predictions``,
    which globs the same directory the same way and is exposed to the same
    trap.
    """
    siblings = _sibling_dataset_prefixes(dataset)
    if not siblings:
        return paths
    return [
        p for p in paths
        if not any(p.name == f"{sib}.jsonl" or p.name.startswith(f"{sib}_") for sib in siblings)
    ]


def count_reviews(reviews_dir: Path, dataset: str) -> Tuple[int, int, int, float]:
    """(rows_done, scored, passed, score_sum) from the reviews JSONL so far.

    ``rows_done`` counts every line, including one we could not read a score
    from: the sample was processed either way, and a progress bar that stalls
    on an unreadable row is worse than one that advances.
    """
    rows_done = scored = passed = 0
    score_sum = 0.0
    try:
        if not reviews_dir.is_dir():
            return (0, 0, 0, 0.0)
        # Underscore glob so `mmmu` does not swallow `mmmu_pro`, matching
        # `_load_predictions` -- then drop any match the glob still let
        # through because it belongs to a longer sibling dataset.
        matches = drop_sibling_matches(
            list(reviews_dir.glob(f"{dataset}_*.jsonl")), dataset,
        )
        exact = reviews_dir / f"{dataset}.jsonl"
        if exact.exists():
            matches.append(exact)
        for path in matches:
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows_done += 1
                    try:
                        score = _score_of(json.loads(line))
                    except Exception:
                        continue
                    if score is None:
                        continue
                    scored += 1
                    score_sum += score
                    if score >= 0.5:
                        passed += 1
    except Exception:
        logger.debug("review count failed", exc_info=True)
    return (rows_done, scored, passed, score_sum)


class ReviewWatcher:
    """Publish row progress for a running evalscope benchmark.

    Started when the benchmark begins and stopped before the next one is
    dispatched, so a late write can never stamp one benchmark's counts onto
    the next one's name.
    """

    def __init__(
        self,
        reviews_dir: Path,
        dataset: str,
        benchmark_name: str,
        rows_total: int,
        interval: float = 2.0,
    ) -> None:
        self._reviews_dir = reviews_dir
        self._dataset = dataset
        self._benchmark_name = benchmark_name
        self._rows_total = rows_total
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            rows_done, scored, passed, score_sum = count_reviews(
                self._reviews_dir, self._dataset,
            )
            if rows_done:
                # for_benchmark: stop()'s join(timeout=5) can return while
                # this tick is still running. Tagging the write means a tick
                # that only finishes after the next benchmark's watcher has
                # already started gets silently dropped by report_rows
                # instead of stamping this benchmark's counts over it.
                runners.report_rows(
                    rows_done, self._rows_total, scored,
                    rows_done - scored, passed, score_sum,
                    for_benchmark=self._benchmark_name,
                )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
