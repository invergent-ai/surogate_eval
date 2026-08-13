# surogate/eval/benchmarks/backends/evalscope_backend.py
"""EvalScope backend for benchmark evaluation."""
import tempfile
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import requests

# ── tree-sitter compat patch ──────────────────────────────────────
# bfcl_eval calls Language(tree_sitter_java.language(), "java") (old
# 2-arg API) but tree-sitter >=0.22 only accepts 1 arg.  Patch
# Language.__init__ to silently drop the second arg so both APIs work.
try:
    import tree_sitter as _ts

    _orig_lang_init = _ts.Language.__init__

    def _compat_lang_init(self, *args, **kwargs):
        if len(args) == 2 and isinstance(args[1], str):
            # Old API: Language(ptr, "java") → drop the name
            return _orig_lang_init(self, args[0], **kwargs)
        return _orig_lang_init(self, *args, **kwargs)

    _ts.Language.__init__ = _compat_lang_init

    # Also patch Parser.set_language → Parser.language setter.
    # Old API: parser.set_language(lang)
    # New API: parser.language = lang
    if not hasattr(_ts.Parser, 'set_language'):
        def _set_language(self, lang):
            self.language = lang
        _ts.Parser.set_language = _set_language
except Exception:
    pass

_original_request = requests.Session.request


def _patched_request(self, method, url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    return _original_request(self, method, url, **kwargs)


requests.Session.request = _patched_request

# Patch datasets.load_dataset to silently drop trust_remote_code
# (removed in datasets v4 but evalscope adapters still pass it)
try:
    import datasets as _ds
    _orig_load_dataset = _ds.load_dataset
    import inspect as _inspect
    if 'trust_remote_code' not in _inspect.signature(_orig_load_dataset).parameters:
        def _patched_load_dataset(*args, **kwargs):
            kwargs.pop('trust_remote_code', None)
            return _orig_load_dataset(*args, **kwargs)
        _ds.load_dataset = _patched_load_dataset
except Exception:
    pass

from surogate_eval.errors import BenchmarkSchemaError
from surogate_eval.targets import BaseTarget
from surogate_eval.utils.logger import get_logger

try:
    from evalscope import run_task, TaskConfig
    from evalscope.constants import EvalType

    EVALSCOPE_AVAILABLE = True
except ImportError:
    EVALSCOPE_AVAILABLE = False
    TaskConfig = None
    EvalType = None

logger = get_logger()


#: Keys we know how to read a sample score from, tried in this order after
#: ``main_score_name`` and ``acc``. Each entry is a benchmark we have seen
#: write its score under that name.
_KNOWN_SCORE_KEYS = (
    'accuracy', 'correct', 'score',
    'resolved', 'passed', 'pass',
    'is_correct',           # simple_qa, simple_vqa
    'prompt_level_strict',  # IFEval/IFBench
    'em', 'exact_match',    # DROP, drivelology
    'f1',                   # DROP, tool_bench
    'winrate',              # alpaca_eval
    'overall_score',        # healthbench
    'eq_bench_score',       # eq_bench
    'total_score',          # mia_bench
)

#: The fallback order, built once. ``main_score_name`` is evalscope's own
#: per-sample signal and is tried ahead of these when the row carries one.
_SCORE_KEY_FALLBACKS = ('acc',) + _KNOWN_SCORE_KEYS


def _extract_sample_score(
        score_obj: Any
) -> Tuple[Optional[float], Dict[str, Any], Optional[str]]:
    """Read one sample's score out of an evalscope review row.

    Returns ``(score, score_details, reason)``. A ``score`` of ``None`` means
    the row could not be read, which is not the same thing as a row that
    scored zero, and the two must never be conflated: the previous version
    used ``score == 0.0`` as its "keep looking" signal, so a correctly-parsed
    wrong answer fell through to a fallback that averaged every number in the
    row and returned a fabricated non-zero score.

    ``reason`` is set exactly when ``score`` is ``None``. Two distinct causes
    are told apart deliberately, because they are triaged differently: no
    score object at all (the review process never produced one) versus a
    score object that exists but has no ``value`` field (scoring was
    attempted and something about the value's shape broke).
    """
    # A malformed review file can carry a non-dict here (e.g. a list from a
    # truncated write). That must be unmeasured, not an AttributeError that
    # takes the rest of the file down with it.
    if not isinstance(score_obj, dict):
        return None, {}, f"score object is not a dict (got {type(score_obj).__name__})"

    if not score_obj:
        return None, {}, "no score object for this sample (sample_score/score missing)"

    value = score_obj.get('value')

    # Some adapters write a bare number rather than a dict of named metrics.
    if isinstance(value, (int, float)):
        return float(value), {}, None

    if value is None:
        return None, {}, "score object has no 'value' field"

    if not isinstance(value, dict) or not value:
        return None, {}, f"score 'value' field is unusable (got {type(value).__name__})"

    main_name = score_obj.get('main_score_name')
    candidates = (
        (main_name, *_SCORE_KEY_FALLBACKS) if main_name else _SCORE_KEY_FALLBACKS
    )

    for key in candidates:
        if key not in value:
            continue
        try:
            return float(value[key]), value, None
        except (TypeError, ValueError):
            # A key we recognise carrying something we cannot read is a
            # failure to measure, not a reason to go looking for a number
            # elsewhere in the row.
            return None, value, f"score key {key!r} is not a number: {value[key]!r}"

    return None, value, f"unrecognised score schema, keys: {sorted(value)}"


def _unreadable_row_record(reason: str, lineno: int) -> Dict[str, Any]:
    """A review row we could not read at all, recorded rather than dropped.

    Dropping it would understate the sample count and hide the failure, so
    the row is kept as unmeasured, on the same rule the rest of this module
    follows: a failure to measure is never a score. It shows in the report
    and in the sample list; it does not reach the run-wide error rate, which
    this backend feeds at task granularity only (see
    ``BenchmarkResult.result_counts``).
    """
    return {
        'input': '',
        'expected': '',
        'output': '',
        'raw_output': '',
        'score': None,
        'score_details': {},
        'success': False,
        'status': 'errored',
        'reason': f'could not read review row {lineno}: {reason}',
        'subset': '',
        'metadata': {},
    }


class EvalScopeBackend:
    """Backend using EvalScope (formerly llmuses) for benchmark evaluation."""

    # Default retry configuration
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 5  # seconds

    # Map our benchmark names to EvalScope dataset names.
    # Verified against evalscope 1.7.0 benchmark registry.
    BENCHMARK_MAP = {
        # ── Standard knowledge & reasoning ────────────────────────
        'mmlu': 'mmlu',
        'mmlu_pro': 'mmlu_pro',
        'mmlu_redux': 'mmlu_redux',
        'mmmlu': 'mmmlu',
        'cmmlu': 'cmmlu',
        'c_eval': 'ceval',
        'gsm8k': 'gsm8k',
        'arc': 'arc',
        'arc_challenge': 'arc',
        'hellaswag': 'hellaswag',
        'truthfulqa': 'truthful_qa',
        'truthful_qa': 'truthful_qa',
        'winogrande': 'winogrande',
        'bbh': 'bbh',
        'gpqa': 'gpqa_diamond',
        'gpqa_diamond': 'gpqa_diamond',
        'super_gpqa': 'super_gpqa',
        'musr': 'musr',
        'hle': 'hle',
        'simple_qa': 'simple_qa',
        'zebralogic': 'zebralogicbench',

        # ── Math ──────────────────────────────────────────────────
        'math': 'competition_math',
        'math_500': 'math_500',
        'aime': 'aime24',
        'aime_2024': 'aime24',
        'aime_2025': 'aime25',
        'aime_2026': 'aime26',
        'hmmt25': 'hmmt25',
        'minerva_math': 'minerva_math',
        'poly_math': 'poly_math',

        # ── Coding ────────────────────────────────────────────────
        'humaneval': 'humaneval',
        'humaneval_plus': 'humaneval_plus',
        'mbpp': 'mbpp',
        'mbpp_plus': 'mbpp_plus',
        'live_code_bench': 'live_code_bench',
        'scicode': 'scicode',
        'multipl_e_humaneval': 'multiple_humaneval',
        'multipl_e_mbpp': 'multiple_mbpp',
        'swe_bench': 'swe_bench_verified',
        'swe_bench_verified': 'swe_bench_verified',
        'swe_bench_lite': 'swe_bench_lite',
        'swe_bench_mini': 'swe_bench_verified_mini',
        'swe_bench_pro': 'swe_bench_pro',
        'swe_bench_multilingual': 'swe_bench_multilingual',
        'terminal_bench': 'terminal_bench_v2',
        'terminal_bench_v2': 'terminal_bench_v2',

        # ── Agent & function calling ──────────────────────────────
        'bfcl': 'bfcl_v3',
        'bfcl_v3': 'bfcl_v3',
        'bfcl_v4': 'bfcl_v4',
        'tau_bench': 'tau_bench',
        'tau2_bench': 'tau2_bench',
        'toolbench': 'tool_bench',
        'tool_bench': 'tool_bench',
        # Custom adapters (surogate_eval.benchmarks.adapters)
        'deep_planning': 'deep_planning',
        'wide_search': 'wide_search',
        'mcp_atlas': 'mcp_atlas',
        'text_to_sql': 'text_to_sql',

        # ── Chat / multi-turn ─────────────────────────────────────
        'mt_bench': 'mt_bench',

        # ── Instruction following ─────────────────────────────────
        'ifeval': 'ifeval',
        'ifbench': 'ifbench',
        'alpaca_eval': 'alpaca_eval',
        'arena_hard': 'arena_hard',
        'multi_if': 'multi_if',

        # ── Long context ──────────────────────────────────────────
        'longbench': 'longbench_v2',
        'longbench_v2': 'longbench_v2',
        'needle_haystack': 'needle_haystack',
        'frames': 'frames',

        # ── Reasoning ─────────────────────────────────────────────
        'process_bench': 'process_bench',
        'coin_flip': 'coin_flip',
        'drop': 'drop',

        # ── QA ────────────────────────────────────────────────────
        'squad': 'squad',
        'boolq': 'boolq',
        'lambada': 'lambada',
        'logiqa': 'logi_qa',
        'mathqa': 'math_qa',
        'triviaqa': 'trivia_qa',
        'commonsenseqa': 'commonsense_qa',
        'piqa': 'piqa',
        'siqa': 'siqa',
        'race': 'race',
        'sciq': 'sciq',
        'pubmedqa': 'pubmedqa',
        'health_bench': 'health_bench',

        # ── Multimodal / Vision ───────────────────────────────────
        'mmmu': 'mmmu',
        'mmmu_pro': 'mmmu_pro',
        'mathvista': 'math_vista',
        'math_vista': 'math_vista',
        'math_verse': 'math_verse',
        'math_vision': 'math_vision',
        'chartqa': 'chartqa',
        'docvqa': 'docvqa',
        'infovqa': 'infovqa',
        'ai2d': 'ai2d',
        'ocr_bench': 'ocr_bench',
        'seed_bench': 'seed_bench_2_plus',
        'mm_bench': 'mm_bench',
        'mm_star': 'mm_star',
        'pope': 'pope',
        'real_world_qa': 'real_world_qa',
        'hallusion_bench': 'hallusion_bench',
        'simple_vqa': 'simple_vqa',
        'tir_bench': 'tir_bench',
        'zerobench': 'zerobench',
        'visulogic': 'visulogic',
        'blink': 'blink',
        'omni_bench': 'omni_bench',
    }

    # Dataset overrides: remap dataset IDs and optionally set a different
    # hub or ModelScope domain per benchmark.
    #
    # Keys: evalscope dataset name
    # Values can include:
    #   dataset_id  — override the dataset_id
    #   hub         — 'modelscope' or 'huggingface'
    #   domain      — 'www.modelscope.ai' or 'www.modelscope.cn'
    #
    # Default hub is modelscope with domain .ai (international).
    # Benchmarks only available on .cn fall back automatically.
    DATASET_OVERRIDES = {
        # ── Remap AI-ModelScope/* → modelscope/* for .ai ──────────
        'gsm8k': {'dataset_id': 'modelscope/gsm8k'},
        'humaneval': {'dataset_id': 'modelscope/humaneval'},
        'truthful_qa': {'dataset_id': 'modelscope/truthful_qa'},
        'winogrande': {'dataset_id': 'modelscope/winogrande_val'},
        'drop': {'dataset_id': 'modelscope/DROP'},
        'mmmu': {'dataset_id': 'modelscope/MMMU'},
        'mmmu_pro': {'dataset_id': 'modelscope/MMMU_Pro'},
        'alpaca_eval': {'dataset_id': 'modelscope/alpaca_eval'},
        'arena_hard': {'dataset_id': 'modelscope/arena-hard-auto-v0.1'},
        'science_qa': {'dataset_id': 'modelscope/ScienceQA'},
        'chinese_simple_qa': {'dataset_id': 'modelscope/Chinese-SimpleQA'},
        'math_500': {'dataset_id': 'modelscope/MATH-500'},
        'iquiz': {'dataset_id': 'modelscope/IQuiz'},
        'bfcl_v3': {'dataset_id': 'modelscope/bfcl_v3'},
        'olympiad_bench': {'dataset_id': 'modelscope/OlympiadBench'},
        'needle_haystack': {'dataset_id': 'modelscope/Needle-in-a-Haystack-Corpus'},
        # ── HuggingFace-only datasets ─────────────────────────────
        'mmlu_pro': {'hub': 'huggingface'},
        'drop': {'hub': 'huggingface'},
        'squad': {'hub': 'huggingface'},
        'triviaqa': {'hub': 'huggingface'},
        'boolq': {'hub': 'huggingface'},
        'lambada': {'hub': 'huggingface'},
        'commonsenseqa': {'hub': 'huggingface'},
        'piqa': {'hub': 'huggingface'},
        'race': {'hub': 'huggingface'},
        'sciq': {'hub': 'huggingface'},
        'pubmedqa': {'hub': 'huggingface'},
        'health_bench': {'hub': 'huggingface'},
        # ── Only on .cn — use .cn domain ──────────────────────────
        'ifeval': {'domain': 'www.modelscope.cn'},
        'gpqa_diamond': {'domain': 'www.modelscope.cn'},
        'mmlu_redux': {'domain': 'www.modelscope.cn'},
        'musr': {'domain': 'www.modelscope.cn'},
        'super_gpqa': {'domain': 'www.modelscope.cn'},
        'tool_bench': {'domain': 'www.modelscope.cn'},
        'live_code_bench': {'domain': 'www.modelscope.cn'},
        'swe_bench_verified': {'domain': 'www.modelscope.cn'},
        'swe_bench_lite': {'domain': 'www.modelscope.cn'},
        'swe_bench_verified_mini': {'domain': 'www.modelscope.cn'},
        'humaneval_plus': {'domain': 'www.modelscope.cn'},
        'arc': {'domain': 'www.modelscope.cn'},
        # Custom adapters — all on .cn
        'deep_planning': {'domain': 'www.modelscope.cn'},
        'wide_search': {'hub': 'huggingface'},
        'mcp_atlas': {'domain': 'www.modelscope.cn'},
        'swe_bench_pro': {'domain': 'www.modelscope.cn'},
        'swe_bench_multilingual': {'domain': 'www.modelscope.cn'},
        'mt_bench': {'hub': 'huggingface'},
        'text_to_sql': {'hub': 'huggingface'},
        # Vision — all on .cn
        'mmmu': {'domain': 'www.modelscope.cn'},
        'mmmu_pro': {'domain': 'www.modelscope.cn'},
        'math_vista': {'domain': 'www.modelscope.cn'},
        'chartqa': {'domain': 'www.modelscope.cn'},
        'docvqa': {'domain': 'www.modelscope.cn'},
        'ocr_bench': {'domain': 'www.modelscope.cn'},
        'hallusion_bench': {'domain': 'www.modelscope.cn'},
        'real_world_qa': {'domain': 'www.modelscope.cn'},
    }

    # Errors that indicate dataset download issues (retryable)
    RETRYABLE_ERRORS = [
        "An error occurred while generating the dataset",
        "Couldn't find file at",
        "Connection reset by peer",
        "Connection timed out",
        "Read timed out",
        "获取数据集文件列表失败",  # ModelScope Chinese error
    ]

    def __init__(self):
        """Initialize EvalScope backend."""
        if not EVALSCOPE_AVAILABLE:
            raise ImportError(
                "EvalScope is not installed. "
                "Install it with: pip install evalscope"
            )
        logger.debug("Initialized EvalScope benchmark backend")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if the error is retryable (e.g., dataset download issues)."""
        error_str = str(error)
        return any(retryable in error_str for retryable in self.RETRYABLE_ERRORS)

    def _clear_dataset_cache(self, dataset_name: str):
        """Clear cached dataset files to force re-download."""
        import shutil

        cache_dirs = [
            Path.home() / '.cache' / 'modelscope' / 'hub' / 'datasets',
            Path.home() / '.cache' / 'huggingface' / 'datasets',
            Path('/root/.cache/modelscope/hub/datasets'),
            Path('/root/.cache/huggingface/datasets'),
        ]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                # Look for dataset-specific cache directories
                for item in cache_dir.iterdir():
                    if dataset_name.lower() in item.name.lower():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                            logger.debug(f"Cleared cache: {item}")
                        except Exception as e:
                            logger.debug(f"Failed to clear cache {item}: {e}")

    def evaluate(
            self,
            target: BaseTarget,
            benchmark_name: str,
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate target using EvalScope benchmark with retry logic.
        """
        logger.info(f"Running EvalScope evaluation for benchmark: {benchmark_name}")

        # Map benchmark name to EvalScope dataset name
        evalscope_dataset = self.BENCHMARK_MAP.get(benchmark_name.lower())
        if not evalscope_dataset:
            raise ValueError(
                f"Benchmark '{benchmark_name}' not supported by EvalScope backend. "
                f"Supported: {list(self.BENCHMARK_MAP.keys())}"
            )

        # Get retry configuration
        max_retries = config.get('max_retries', self.DEFAULT_MAX_RETRIES)
        retry_delay = config.get('retry_delay', self.DEFAULT_RETRY_DELAY)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Prepare EvalScope task config
                task_config = self._prepare_task_config(target, evalscope_dataset, config)

                logger.info(f"Running EvalScope task for dataset: {evalscope_dataset}")

                # Pre-run: apply configurable max_num_steps for tau_bench.
                self._apply_tau_bench_max_turns(evalscope_dataset, config)

                # Patch MCQ answer extraction to handle models that
                # don't follow the exact ANSWER: format.
                self._patch_mcq_extract_answer(evalscope_dataset)

                # Run the task. run_task() blocks for the whole benchmark, so
                # a background watcher counts the reviews file evalscope
                # appends to as it scores and reports row progress; it must
                # be stopped before this method returns, or a late write
                # could stamp this benchmark's counts onto the next one's.
                from ._evalscope_progress import ReviewWatcher

                # Setup and start are guarded on their own: a bad limit, a
                # work_dir that isn't there yet, or the OS refusing to start
                # a new thread must not fail a benchmark that is otherwise
                # fine. Progress reporting is a nice-to-have layered on top
                # of the run, never something the run depends on.
                watcher = None
                try:
                    watcher = ReviewWatcher(
                        # A closure, not a value computed here: run_task
                        # mutates task_config.work_dir in place
                        # (setup_work_directory appends a timestamp) after
                        # this call, so the watcher must read work_dir fresh
                        # on every tick or it watches a directory that never
                        # receives anything.
                        reviews_dir=lambda: Path(task_config.work_dir) / 'reviews' / task_config.model_id,
                        dataset=evalscope_dataset,
                        # The name eval.py's progress context tracks (`name`
                        # in _run_single_benchmark), not evalscope_dataset --
                        # they differ for benchmarks like 'aime' -> 'aime24'.
                        # Every write is tagged with this so a stale tick
                        # from a watcher whose stop() timed out cannot land
                        # as this benchmark's counts once the context has
                        # moved on.
                        benchmark_name=benchmark_name,
                        # Off for a custom dataset path (see
                        # _prepare_task_config): no tracker is written, and
                        # a same-second work_dir collision with the previous
                        # benchmark would otherwise have us read its file.
                        read_tracker=bool(
                            getattr(task_config, 'enable_progress_tracker', True)
                        ),
                        # Used only when evalscope writes no tracker at all
                        # (mt_bench and any other benchmark with no bundled
                        # `_meta`), so the bar has a denominator instead of
                        # sitting on the "unknown" sentinel for the whole
                        # benchmark.
                        limit=config.get('limit'),
                    )
                    watcher.start()
                except Exception:
                    logger.warning(
                        "Could not start row-progress watcher; continuing "
                        "without row progress for this benchmark",
                        exc_info=True,
                    )
                    watcher = None

                try:
                    results = run_task(task_cfg=task_config)
                finally:
                    # Only stop a watcher that actually started: stop() on
                    # one whose start() never ran (or never happened) joins
                    # a thread that was never launched.
                    if watcher is not None:
                        watcher.stop()

                # EvalScope saves results to work_dir/reports/{model_id}/{dataset}.json
                import json
                work_dir = task_config.work_dir
                model_id = task_config.model_id
                results_file = Path(work_dir) / 'reports' / model_id / f'{evalscope_dataset}.json'

                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    logger.debug(f"Loaded results from: {results_file}")
                else:
                    logger.warning(f"Results file not found: {results_file}")
                    results = {}

                # Load detailed predictions
                detailed_results = self._load_predictions(work_dir, model_id, evalscope_dataset)

                # Parse results
                parsed_results = self._parse_results(results, benchmark_name, detailed_results)

                logger.success(f"EvalScope evaluation completed for: {benchmark_name}")
                return parsed_results

            except Exception as e:
                last_error = e

                if self._is_retryable_error(e) and attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed with retryable error: {e}. "
                        f"Retrying in {retry_delay} seconds..."
                    )

                    # Clear cache before retry to force fresh download
                    self._clear_dataset_cache(evalscope_dataset)

                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay = min(retry_delay * 2, 60)
                else:
                    # Non-retryable error or max retries reached
                    logger.error(f"EvalScope evaluation failed: {e}")
                    logger.debug(traceback.format_exc())
                    raise

        # If we get here, all retries failed
        logger.error(f"All {max_retries} attempts failed for benchmark: {benchmark_name}")
        raise last_error

    def _load_predictions(
            self,
            work_dir: str,
            model_id: str,
            dataset_name: str
    ) -> List[Dict[str, Any]]:
        """Load detailed predictions from EvalScope output."""
        import json

        detailed_results = []

        # EvalScope stores scores in reviews directory, not predictions
        reviews_dir = Path(work_dir) / 'reviews' / model_id
        if not reviews_dir.exists():
            logger.debug(f"Reviews directory not found: {reviews_dir}")
            # Fallback to predictions directory
            reviews_dir = Path(work_dir) / 'predictions' / model_id
            if not reviews_dir.exists():
                logger.debug(f"Predictions directory not found: {reviews_dir}")
                return detailed_results

        # Underscore glob first, but that alone still matches a longer
        # sibling dataset (e.g. mmmu_pro_val.jsonl when dataset_name is
        # 'mmmu': fnmatch's `*` does not stop at the next underscore).
        # resolve_review_paths cross-checks against the sibling-dataset list
        # to tell a subset name from a sibling dataset, and is shared with
        # the live ReviewCounter so the two cannot resolve a dataset's
        # reviews to different files.
        from ._evalscope_progress import resolve_review_paths

        matches = resolve_review_paths(reviews_dir, dataset_name)
        for review_file in matches:
            logger.debug(f"Loading reviews from: {review_file}")
            # Two nested handlers, doing different jobs. The inner one keeps a
            # bad row from costing more than itself: the handler this replaced
            # wrapped the whole file, so one malformed record - a truncated
            # line, or a field holding the wrong type - abandoned every
            # remaining row in that file, visible only as a warning. Guarding
            # individual fields cannot close that class of failure; isolating
            # the row can. The outer one still catches what is genuinely
            # file-level (open, permissions, a read failing mid-iteration),
            # where there is no row to attribute the failure to.
            try:
                with open(review_file, 'r') as f:
                    # Streamed rather than read into a list: a record's raw row
                    # is kept in ``metadata``, so buffering the file as well
                    # would hold every row twice at peak.
                    for lineno, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            detailed_results.append(
                                self._review_row_to_record(json.loads(line))
                            )
                        except Exception as e:
                            logger.error(
                                f"Review row {review_file.name}:{lineno} "
                                f"could not be read: {e}"
                            )
                            logger.debug(traceback.format_exc())
                            detailed_results.append(
                                _unreadable_row_record(f"{type(e).__name__}: {e}", lineno)
                            )
            except Exception as e:
                logger.warning(f"Failed to read reviews from {review_file}: {e}")
                logger.debug(traceback.format_exc())

        logger.info(f"Loaded {len(detailed_results)} detailed predictions")
        return detailed_results

    def _get_max_input_tokens(
            self,
            dataset_name: str,
            config: Dict[str, Any]
    ) -> int | None:
        """
        Sample dataset to find max input token count.
        """
        try:
            from datasets import load_dataset

            subset = config.get('subset')
            if isinstance(subset, list):
                subset = subset[0] if subset else None

            # Map to HuggingFace dataset ID
            hf_dataset_map = {
                'mmlu': 'cais/mmlu',
                'arc': 'allenai/ai2_arc',
                'arc_challenge': 'allenai/ai2_arc',
                'hellaswag': 'Rowan/hellaswag',
                'truthfulqa': 'truthfulqa/truthful_qa',
                'gsm8k': 'openai/gsm8k',
                'winogrande': 'allenai/winogrande',
            }

            dataset_id = hf_dataset_map.get(dataset_name.lower(), dataset_name)

            logger.debug(f"Loading dataset samples from: {dataset_id} (subset: {subset})")

            # Load samples
            try:
                if subset:
                    ds = load_dataset(dataset_id, subset, split='test', trust_remote_code=True)
                else:
                    ds = load_dataset(dataset_id, split='test', trust_remote_code=True)
            except:
                # Try without subset
                ds = load_dataset(dataset_id, split='test', trust_remote_code=True)

            # Sample up to limit or 100 samples for estimation
            limit = config.get('limit') or 100
            sample_count = min(limit, len(ds))
            samples = list(ds.select(range(sample_count)))

            if not samples:
                return None

            # Find max length (rough estimate: 4 chars ≈ 1 token)
            max_chars = 0
            for sample in samples:
                # Try common field names
                text = (
                        sample.get('question', '') or
                        sample.get('input', '') or
                        sample.get('prompt', '') or
                        sample.get('text', '') or
                        str(sample)
                )
                max_chars = max(max_chars, len(str(text)))

            max_tokens = max_chars // 4
            logger.debug(f"Dataset max input: ~{max_tokens} tokens ({max_chars} chars from {sample_count} samples)")

            return max_tokens

        except Exception as e:
            logger.debug(f"Could not sample dataset: {e}")
            return None


    def _review_row_to_record(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Build one detailed-result record from one evalscope review row.

        Split out of the review-file loop so that a row which cannot be
        read costs exactly one row: the caller wraps this call, and
        anything raised in here is contained to the record it was
        building rather than to the rest of the file.
        """

        # Extract score from EvalScope's nested structure
        # sample_score.score.value.<metric>. A row we
        # cannot read a score from is unmeasured, which
        # includes a row carrying no score object at all.
        # A malformed file can carry a non-dict in either
        # spot (e.g. a list from a truncated write); guard
        # sample_score itself here so a bad one cannot
        # raise before _extract_sample_score is reached -
        # a bad nested score_obj is handled there instead.
        sample_score = sample.get('sample_score')
        if not isinstance(sample_score, dict):
            sample_score = {}
        score_obj = sample_score.get('score') or {}
        score, score_details, score_reason = _extract_sample_score(score_obj)

        # Read once, guarded: a truthy non-dict here would otherwise reach
        # ``.get`` at both use sites below. The caller isolates the row, so
        # the cost of that would be this row rather than the file; the guard
        # is what keeps the rest of the row usable, with only its metadata
        # discarded, instead of losing the whole record to the handler.
        sample_meta = sample_score.get('sample_metadata')
        if not isinstance(sample_meta, dict):
            sample_meta = {}

        # Extract input, target, and prediction
        input_text = sample.get('input', '')
        expected = sample.get('target', '')

        # For IFEval-style benchmarks: extract prompt from
        # messages or metadata when top-level fields are empty
        if not input_text:
            msgs = sample.get('messages', [])
            for m in msgs:
                if isinstance(m, dict) and m.get('role') == 'user':
                    input_text = m.get('content', '')
                    break
        if not expected or len(expected) > 2000:
            # Build a human-readable expected from metadata
            meta = sample_meta

            # IFEval/IFBench: instruction constraints
            instructions = meta.get('instruction_id_list', [])
            kwargs_list = meta.get('kwargs', [])
            if instructions:
                parts = []
                for idx, inst in enumerate(instructions):
                    kw = kwargs_list[idx] if idx < len(kwargs_list) else {}
                    kw_str = ', '.join(
                        f'{k}={v}' for k, v in (kw or {}).items()
                        if v is not None
                    )
                    parts.append(f'{inst}({kw_str})' if kw_str else inst)
                expected = ' | '.join(parts)

            # SWE-bench: show instance_id + repo instead of full patch
            elif meta.get('instance_id'):
                instance_id = meta['instance_id']
                repo = meta.get('repo', '')
                difficulty = meta.get('difficulty', '')
                expected = f'{instance_id} ({repo})' + (
                    f' [{difficulty}]' if difficulty else ''
                )

        # Get the model's prediction/output. score_obj can
        # still be a non-dict here (e.g. a malformed row's
        # nested score is a list); _extract_sample_score
        # already turned that into an unmeasured score
        # above, so treat prediction/extracted as absent
        # here too rather than raising on .get().
        score_fields = score_obj if isinstance(score_obj, dict) else {}
        prediction = score_fields.get('prediction', '') or ''
        extracted = score_fields.get('extracted_prediction', '') or ''

        return {
            'input': input_text,
            'expected': expected,
            'output': extracted or prediction[:500],
            'raw_output': prediction,
            'score': score,
            'score_details': score_details,
            'success': score is not None and score > 0,
            'status': 'errored' if score is None else 'scored',
            'reason': score_reason,
            'subset': sample_meta.get('subject', ''),
            'metadata': sample
        }

    def _parse_results(
            self,
            results: Dict[str, Any],
            benchmark_name: str,
            detailed_results: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse EvalScope results into standardized format.
        """
        task_results = {}

        # An empty payload means the report file was missing. That is already
        # handled downstream: no task parses, and BenchmarkResult.result_counts
        # charges one errored unit. A payload that loaded but has no 'score' is
        # different - it is what an upstream key rename looks like - and
        # defaulting it to 0.0 reports a model that scored zero.
        if results and results.get('score') is None:
            raise BenchmarkSchemaError(
                f"evalscope report for {benchmark_name!r} has no usable 'score'; "
                f"got keys: {sorted(results)}"
            )

        overall_score = results.get('score', 0.0)

        # Extract subset scores from metrics
        metrics = results.get('metrics', [])
        total_samples = 0

        for metric in metrics:
            metric_name = metric.get('name', 'unknown')
            categories = metric.get('categories', [])

            for category in categories:
                subsets = category.get('subsets', [])

                for subset in subsets:
                    subset_name = subset.get('name')
                    if not subset_name:
                        continue

                    # A missing score is a schema we cannot read, not a model
                    # that scored zero: defaulting it would have
                    # ``result_counts()`` count this subset as scored on a
                    # fabricated number. Recorded as failed rather than
                    # raised, so it costs one subset and not the whole
                    # report - the same rule a bad row follows. The
                    # report-level guard above still raises, because a report
                    # with no score has nothing left to salvage.
                    if subset.get('score') is None:
                        logger.error(
                            f"evalscope subset {subset_name!r} in {benchmark_name!r} "
                            f"has no usable 'score'; got keys: {sorted(subset)}"
                        )
                        task_results[subset_name] = {
                            'status': 'failed',
                            'score': None,
                            'n_samples': subset.get('num', 0),
                            'reason': (
                                f"no usable 'score'; got keys: {sorted(subset)}"
                            ),
                        }
                        continue

                    subset_score = subset['score']
                    subset_num = subset.get('num', 0)

                    task_results[subset_name] = {
                        'score': subset_score,
                        'accuracy': subset_score,
                        'n_samples': subset_num,
                    }
                    total_samples += subset_num

        return {
            'overall_score': overall_score,
            'num_samples': total_samples,
            'task_results': task_results,
            'detailed_results': detailed_results or [],
            'metadata': {
                'backend': 'evalscope',
                'benchmark': benchmark_name,
                'num_datasets': len(task_results),
                'dataset_name': results.get('dataset_name', benchmark_name),
                'model_name': results.get('model_name', 'unknown'),
            }
        }

    _mbpp_patched = False

    @classmethod
    def _patch_mbpp_nosandbox(cls):
        """Patch MBPP adapter to use subprocess exec instead of raising
        RuntimeError when use_sandbox=false."""
        if cls._mbpp_patched:
            return
        try:
            from evalscope.benchmarks.mbpp.mbpp_adapter import MBPPAdapter
            import subprocess, tempfile

            def _match_score_nosandbox(self, original_prediction, filtered_prediction, reference, task_state):
                from evalscope.api.metric import Score
                score = Score(
                    extracted_prediction=filtered_prediction,
                    prediction=original_prediction,
                )
                problem = task_state.metadata
                completion = filtered_prediction
                for test in task_state.metadata.get('test_list', []):
                    completion += '\n' + test + '\n'

                passed = False
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(completion)
                        f.flush()
                        result = subprocess.run(
                            ['python3', f.name],
                            capture_output=True, timeout=30,
                        )
                        passed = result.returncode == 0
                except (subprocess.TimeoutExpired, Exception) as e:
                    logger.debug(f"MBPP exec failed: {e}")
                    passed = False

                score.value = {'acc': passed}
                score.metadata = {
                    'task_id': problem.get('task_id', ''),
                    'timeout': getattr(self, 'review_timeout', 30),
                }
                score.main_score_name = 'acc'
                return score

            MBPPAdapter.match_score = _match_score_nosandbox
            cls._mbpp_patched = True
            logger.info("Patched MBPP adapter for subprocess execution (no sandbox)")
        except ImportError:
            logger.debug("MBPP adapter not available for patching")

    # MCQ vision benchmarks that use letter-based answers (A/B/C/D).
    _MCQ_BENCHMARKS = {
        'real_world_qa', 'mmmu', 'mmmu_pro', 'mm_bench', 'mm_star',
        'science_qa', 'a_okvqa', 'ai2d', 'seed_bench_2_plus',
    }

    @staticmethod
    def _patch_mcq_extract_answer(dataset_name: str) -> None:
        """Patch MCQ benchmark adapters with a more robust answer extractor.

        Models often respond with explanations before the answer letter,
        or omit the ``ANSWER:`` prefix.  This wraps the original
        ``extract_answer`` to fall back to extracting a standalone
        letter (A-J) from the end of the response.
        """
        import re as _re
        if dataset_name not in EvalScopeBackend._MCQ_BENCHMARKS:
            return
        try:
            from evalscope.api.registry import BENCHMARK_REGISTRY
            meta = BENCHMARK_REGISTRY.get(dataset_name)
            if meta is None:
                return
            adapter_cls = meta.data_adapter
            if adapter_cls is None:
                return

            original = adapter_cls.extract_answer

            def _robust_extract(self, prediction: str, task_state=None) -> str:
                # Try original first
                result = original(self, prediction, task_state)
                if result:
                    return result
                # Fallback: find ANSWER: pattern (case-insensitive)
                matches = _re.findall(r'(?i)answer\s*[:=]\s*([A-Ja-j])', prediction)
                if matches:
                    return matches[-1].strip().lower()
                # Fallback: \boxed{X} (LaTeX)
                matches = _re.findall(r'\\boxed\{([A-Ja-j])\}', prediction)
                if matches:
                    return matches[-1].lower()
                # Fallback: last standalone letter on its own line
                lines = prediction.strip().splitlines()
                for line in reversed(lines):
                    line = line.strip().rstrip('.')
                    if _re.fullmatch(r'[A-Ja-j]', line):
                        return line.lower()
                # Fallback: last letter in the response
                matches = _re.findall(r'\b([A-Ja-j])\b', prediction)
                if matches:
                    return matches[-1].lower()
                return ''

            adapter_cls.extract_answer = _robust_extract
            logger.info(f"Patched extract_answer for MCQ benchmark: {dataset_name}")
        except Exception as e:
            logger.debug(f"Could not patch extract_answer for {dataset_name}: {e}")

    @staticmethod
    def _apply_tau_bench_max_turns(dataset_name: str, config: dict) -> None:
        """Patch tau_bench's agent solve loop with configurable max turns.

        The evalscope tau_bench adapter hardcodes ``max_num_steps=30`` in
        its monkey-patched ``ToolCallingAgent.solve``.  If the user passed
        ``backend_params.max_turns``, we re-patch the method to use that
        value instead.
        """
        if dataset_name not in ('tau_bench', 'tau2_bench'):
            return
        bp = config.get('backend_params', {}) or {}
        max_turns = bp.get('max_turns')
        if max_turns is None:
            return
        max_turns = int(max_turns)
        try:
            from tau_bench.agents.tool_calling_agent import ToolCallingAgent
            original_solve = ToolCallingAgent.solve

            def _solve_with_limit(self, env, task_index=None, max_num_steps=max_turns):
                return original_solve(self, env, task_index=task_index, max_num_steps=max_num_steps)

            ToolCallingAgent.solve = _solve_with_limit
            logger.info(f"Patched tau_bench max_num_steps to {max_turns}")
        except ImportError:
            logger.debug("tau_bench not installed, skipping max_turns patch")

    def _prepare_task_config(
            self,
            target: BaseTarget,
            dataset_name: str,
            config: Dict[str, Any]
    ) -> TaskConfig:
        """Prepare EvalScope TaskConfig."""

        # Code benchmarks that require sandbox execution
        CODE_BENCHMARKS = {'mbpp', 'humaneval', 'humaneval_plus', 'mbpp_plus', 'ds1000', 'leetcode'}

        eval_type = self._get_eval_type(target)
        force_default_subset = config.get('_force_default_subset', False)

        # Base config
        task_cfg_dict = {
            'datasets': [dataset_name],
            'work_dir': tempfile.gettempdir(),
            # Makes evalscope run its own ProgressTracker, which writes a
            # real processed_count/total_count to <work_dir>/progress.json
            # (a different file from our eval_results/progress.json) as the
            # run proceeds. ReviewWatcher (_evalscope_progress.py) reads it
            # for row counts instead of guessing the total from
            # config['limit'], which was 0 -- "unknown" -- for every
            # benchmark run without an explicit limit. Off by default in
            # evalscope.
            'enable_progress_tracker': True,
        }

        # Enable sandbox for code benchmarks (unless explicitly disabled
        # via backend_params.use_sandbox=false — needed when running
        # inside k8s pods where Docker is unavailable).
        bp = config.get('backend_params', {}) or {}
        use_sandbox = bp.get('use_sandbox', config.get('use_sandbox', True))
        if dataset_name.lower() in CODE_BENCHMARKS and use_sandbox:
            task_cfg_dict['use_sandbox'] = True
            task_cfg_dict['sandbox_type'] = 'docker'
            logger.info(f"Enabling sandbox for code benchmark: {dataset_name}")
        elif dataset_name.lower() in CODE_BENCHMARKS:
            logger.info(f"Sandbox disabled for code benchmark: {dataset_name} (use_sandbox=false)")
            # Patch MBPP to use subprocess exec instead of raising RuntimeError.
            self._patch_mbpp_nosandbox()

        # Add model configuration based on eval type
        if eval_type == EvalType.SERVICE:
            task_cfg_dict['eval_type'] = EvalType.SERVICE
            task_cfg_dict['model'] = target.config.get('model', 'model')

            base_url = target.config.get('base_url')
            if base_url:
                if not base_url.endswith('/v1'):
                    base_url = f"{base_url}/v1" if not base_url.endswith('/') else f"{base_url}v1"
                task_cfg_dict['api_url'] = base_url

            api_key = target.config.get('api_key', 'EMPTY')
            task_cfg_dict['api_key'] = api_key
        else:
            model_path = target.config.get('model_path') or target.config.get('model')
            task_cfg_dict['model'] = model_path

            model_args = {}
            if target.config.get('device'):
                model_args['device_map'] = target.config['device']
            if model_args:
                task_cfg_dict['model_args'] = model_args

        # Handle custom dataset path
        dataset_path = config.get('dataset_path') or config.get('path') or config.get('custom_dataset')
        dataset_hub = config.get('dataset_hub') or config.get('hub')
        available_splits = None

        if dataset_path:
            # Check if it's a LakeFS URL - need to download first
            if dataset_path.startswith('lakefs://'):
                local_dataset_path, available_splits = self._download_lakefs_dataset(dataset_path, dataset_name)
                logger.info(f"Downloaded LakeFS dataset to: {local_dataset_path}")
                dataset_path = local_dataset_path
                dataset_hub = 'local'

            if not task_cfg_dict.get('dataset_args'):
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}

            task_cfg_dict['dataset_args'][dataset_name]['dataset_id'] = dataset_path
            task_cfg_dict['dataset_args'][dataset_name]['default_subset'] = 'default'

            # The tracker's total would be a confident lie for this run.
            # evalscope's compute_eval_total_count reads per-subset
            # sample_count out of the *bundled* _meta files and consults
            # dataset_args['subset_list'] but never dataset_args
            # ['dataset_id'], so it reports the size of the upstream dataset
            # rather than the one actually being evaluated. Smaller custom
            # dataset: the bar stalls partway and never completes. Larger:
            # rows_done sails past rows_total. Off, so
            # read_evalscope_tracker finds no file and the watcher publishes
            # rows_total=0 -- the established "unknown" sentinel both repos
            # already fall back on (live-progress.ts treats <= 0 as
            # unknown). rows_done still comes from the reviews line count,
            # so progress is not lost, only the total nobody can know here.
            task_cfg_dict['enable_progress_tracker'] = False
            logger.info(
                "Custom dataset path set for %s; progress total unavailable "
                "(evalscope's estimate would describe the upstream dataset)",
                dataset_name,
            )

            # If local dataset has no train split, use test for few-shot examples
            if available_splits is not None and 'train' not in available_splits:
                if 'test' in available_splits or not available_splits:
                    task_cfg_dict['dataset_args'][dataset_name]['train_split'] = 'test'
                    logger.info(f"No train split found, using test split for few-shot examples")

            logger.info(f"Set custom dataset_id to: {dataset_path}")

        # Default: modelscope.cn — has all datasets. Some benchmarks
        # override to .ai where the dataset is available internationally.
        import os
        if 'MODELSCOPE_DOMAIN' not in os.environ:
            os.environ['MODELSCOPE_DOMAIN'] = 'www.modelscope.cn'

        resolved_hub = dataset_hub or 'modelscope'
        task_cfg_dict['dataset_hub'] = resolved_hub

        # Apply per-benchmark overrides (dataset_id, hub, domain)
        if dataset_name in self.DATASET_OVERRIDES:
            overrides = self.DATASET_OVERRIDES[dataset_name]

            if 'domain' in overrides:
                os.environ['MODELSCOPE_DOMAIN'] = overrides['domain']
                logger.info(f"Switched ModelScope domain to {overrides['domain']} for '{dataset_name}'")

            if 'hub' in overrides:
                task_cfg_dict['dataset_hub'] = overrides['hub']

            if 'dataset_id' in overrides:
                if not task_cfg_dict.get('dataset_args'):
                    task_cfg_dict['dataset_args'] = {}
                if dataset_name not in task_cfg_dict['dataset_args']:
                    task_cfg_dict['dataset_args'][dataset_name] = {}
                task_cfg_dict['dataset_args'][dataset_name]['dataset_id'] = overrides['dataset_id']

            logger.info(f"Applied dataset override for '{dataset_name}': {overrides}")

        # Add limit if specified
        if 'limit' in config and config['limit']:
            task_cfg_dict['limit'] = config['limit']

        # Add num_fewshot if specified
        if 'num_fewshot' in config and config['num_fewshot'] is not None:
            if not task_cfg_dict.get('dataset_args'):
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}

            task_cfg_dict['dataset_args'][dataset_name]['few_shot_num'] = config['num_fewshot']
            task_cfg_dict['dataset_args'][dataset_name]['few_shot_random'] = False

        # Add subset if specified
        subset = config.get('subset')

        if subset:
            # User explicitly specified a subset
            if not task_cfg_dict.get('dataset_args'):
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}

            if isinstance(subset, str):
                subset = [subset]
            task_cfg_dict['dataset_args'][dataset_name]['subset_list'] = subset
            logger.info(f"Using '{subset}' subset for dataset: {dataset_name}")

        elif force_default_subset:
            # Only force 'default' when retrying after subset error
            if not task_cfg_dict.get('dataset_args'):
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}
            task_cfg_dict['dataset_args'][dataset_name]['subset_list'] = ['default']
            logger.info(f"Retrying with 'default' subset for dataset: {dataset_name}")

        # Pass extra_params for benchmarks that support them
        # (terminal-bench, tau-bench, etc.)
        backend_params = config.get('backend_params', {})
        _EXTRA_PARAM_KEYS = ('timeout_multiplier', 'max_turns', 'environment_type', 'agent_name')
        extra_from_bp = {k: backend_params[k] for k in _EXTRA_PARAM_KEYS if k in backend_params}
        if extra_from_bp:
            if 'dataset_args' not in task_cfg_dict:
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}
            ds_args = task_cfg_dict['dataset_args'][dataset_name]
            if 'extra_params' not in ds_args:
                ds_args['extra_params'] = {}
            ds_args['extra_params'].update(extra_from_bp)
            logger.info(f"Passed extra_params for {dataset_name}: {extra_from_bp}")

        # Add backend params for concurrency and batching

        # Generation config - set defaults or use from backend_params
        if 'generation_config' not in task_cfg_dict:
            task_cfg_dict['generation_config'] = {
                'batch_size': backend_params.get('generation_batch_size', 1),
                'temperature': config.get('temperature', 0.0),
            }

        # Batch size for evaluation
        if 'batch_size' in backend_params:
            task_cfg_dict['eval_batch_size'] = backend_params['batch_size']
            logger.debug(f"Setting eval_batch_size to {backend_params['batch_size']}")

        # Worker threads for parallel evaluation
        max_workers = backend_params.get('max_workers')
        if not max_workers:
            infrastructure = target.config.get('infrastructure', {})
            max_workers = infrastructure.get('workers', 1)

        if max_workers and max_workers > 1:
            task_cfg_dict['judge_worker_num'] = max_workers
            logger.debug(f"Setting judge_worker_num to {max_workers}")

        # Inject judge model args for benchmarks that use LLM-as-judge.
        # The judge target is passed via backend_params['judge_target'] by
        # the runner when a judge_model config is present.
        judge_target = bp.get('judge_target') or config.get('judge_target')
        if judge_target and hasattr(judge_target, 'config'):
            jcfg = judge_target.config
            judge_url = jcfg.get('base_url', '')
            judge_model = jcfg.get('model', '')
            judge_key = jcfg.get('api_key', '')

            # Benchmarks with numeric scoring (1-10) vs binary A/B pattern
            _NUMERIC_JUDGE_BENCHMARKS = {
                'mt_bench', 'text_to_sql', 'wide_search', 'tool_decathlon',
                'deep_planning', 'hallusion_bench',
            }
            score_type = 'numeric' if dataset_name in _NUMERIC_JUDGE_BENCHMARKS else 'pattern'
            task_cfg_dict['judge_model_args'] = {
                'model_id': judge_model,
                'api_url': judge_url,
                'api_key': judge_key,
                'score_type': score_type,
            }
            task_cfg_dict['judge_strategy'] = 'auto'
            logger.info(f"Injected judge model args for {dataset_name}: model={judge_model}")

            # Benchmarks that need a user simulator (tau_bench, tau2_bench)
            # or other external model — override their extra_params with
            # the judge target credentials so they don't use hardcoded defaults.
            _USER_SIM_BENCHMARKS = {'tau_bench', 'tau2_bench'}
            if dataset_name in _USER_SIM_BENCHMARKS:
                if not task_cfg_dict.get('dataset_args'):
                    task_cfg_dict['dataset_args'] = {}
                if dataset_name not in task_cfg_dict['dataset_args']:
                    task_cfg_dict['dataset_args'][dataset_name] = {}
                ds_args = task_cfg_dict['dataset_args'][dataset_name]
                if 'extra_params' not in ds_args:
                    ds_args['extra_params'] = {}
                ds_args['extra_params']['user_model'] = judge_model
                ds_args['extra_params']['api_key'] = judge_key
                ds_args['extra_params']['api_base'] = judge_url
                # Pass configurable max turns and generation config.
                max_turns = backend_params.get('max_turns')
                if max_turns is not None:
                    ds_args['extra_params']['max_num_steps'] = int(max_turns)
                gen_cfg = backend_params.get('generation_config')
                if gen_cfg and isinstance(gen_cfg, dict):
                    ds_args['extra_params']['generation_config'] = gen_cfg
                logger.info(f"Injected user simulator config for {dataset_name}: model={judge_model}")

        return TaskConfig(**task_cfg_dict)

    def _download_lakefs_dataset(self, lakefs_url: str, dataset_name: str) -> tuple[str, set[str]]:
        """
        Download dataset from LakeFS to local directory.

        Args:
            lakefs_url: LakeFS URL (lakefs://repo/ref or lakefs://repo/ref/path)
            dataset_name: Name of the dataset (used for local directory naming)

        Returns:
            Tuple of (local_path, available_splits)
        """
        import subprocess

        # Create local directory for dataset
        local_dir = Path(tempfile.gettempdir()) / 'evalscope_datasets' / dataset_name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Ensure lakefs URL ends with / for recursive download of all files
        if not lakefs_url.endswith('/'):
            lakefs_url = lakefs_url + '/'

        logger.info(f"Downloading LakeFS dataset: {lakefs_url} -> {local_dir}")

        try:
            # Use lakectl to download the dataset
            result = subprocess.run(
                ['lakectl', 'fs', 'download', '--recursive', lakefs_url, str(local_dir) + '/'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"lakectl output: {result.stdout}")

            # Check what was downloaded and detect splits
            downloaded_files = list(local_dir.rglob('*'))
            logger.info(f"Downloaded {len(downloaded_files)} files to {local_dir}")

            # Detect available splits from directory names AND filenames
            available_splits = set()
            for f in downloaded_files:
                if f.is_dir():
                    name = f.name.lower()
                    if name in ('train', 'test', 'validation', 'val', 'dev'):
                        available_splits.add(name)
                elif f.is_file():
                    stem = f.stem.lower()
                    if stem in ('train', 'test', 'validation', 'val', 'dev'):
                        available_splits.add(stem)

            if not downloaded_files:
                raise RuntimeError(f"No files downloaded from {lakefs_url}")

            logger.info(f"Detected splits: {available_splits if available_splits else 'none (single file)'}")

            return str(local_dir), available_splits

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download from LakeFS: {e.stderr}")
            raise RuntimeError(f"LakeFS download failed: {e.stderr}")
        except FileNotFoundError:
            logger.error("lakectl command not found. Is LakeFS CLI installed?")
            raise RuntimeError("lakectl command not found")

    def _get_eval_type(self, target: BaseTarget) -> str:
        """
        Determine EvalScope eval type based on target.

        Args:
            target: Target instance

        Returns:
            EvalType constant
        """
        # Check if target has API endpoint
        if target.config.get('base_url') or target.config.get('api_url'):
            return EvalType.SERVICE

        # Check if it's a local model
        if target.config.get('model_path') or target.config.get('provider') == 'local':
            return EvalType.LOCAL

        # Default to service (API-based)
        return EvalType.SERVICE


    @staticmethod
    def list_available_benchmarks() -> List[str]:
        """List available EvalScope benchmarks."""
        if not EVALSCOPE_AVAILABLE:
            return []

        return list(EvalScopeBackend.BENCHMARK_MAP.keys())

    @staticmethod
    def get_supported_datasets() -> Dict[str, str]:
        """
        Get mapping of benchmark names to EvalScope dataset names.

        Returns:
            Dictionary mapping benchmark names to dataset names
        """
        return EvalScopeBackend.BENCHMARK_MAP.copy()


# Alias for backward compatibility
EvalScopeBackendWrapper = EvalScopeBackend
