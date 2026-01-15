# surogate/eval/benchmarks/backends/evalscope_backend.py
"""EvalScope backend for benchmark evaluation."""
import tempfile
import time
import random
from typing import Dict, Any, List
from pathlib import Path

# =============================================================================
# Patch User-Agent globally for ModelScope compatibility
# This fixes issues with ModelScope blocking default User-Agents from certain IPs
# =============================================================================

# Try to use fake-useragent, fallback to static list
try:
    from fake_useragent import UserAgent
    _ua = UserAgent()
    _USE_FAKE_UA = True
except ImportError:
    _USE_FAKE_UA = False
    _ua = None

# Fallback User-Agent list if fake-useragent is not installed
_FALLBACK_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
]


def _get_random_user_agent() -> str:
    """Get a random User-Agent string."""
    if _USE_FAKE_UA:
        try:
            return _ua.random
        except Exception:
            pass
    return random.choice(_FALLBACK_USER_AGENTS)


# Patch requests at multiple levels
import requests

# Patch Session.request (catches most session-based requests)
_original_session_request = requests.Session.request


def _patched_session_request(self, method, url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = _get_random_user_agent()
    return _original_session_request(self, method, url, **kwargs)


requests.Session.request = _patched_session_request

# Patch requests.get/post/head/put/delete/patch directly (for non-session calls)
_original_get = requests.get
_original_post = requests.post
_original_head = requests.head
_original_put = requests.put
_original_delete = requests.delete
_original_patch = requests.patch


def _patched_get(url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = _get_random_user_agent()
    return _original_get(url, **kwargs)


def _patched_post(url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = _get_random_user_agent()
    return _original_post(url, **kwargs)


def _patched_head(url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = _get_random_user_agent()
    return _original_head(url, **kwargs)


def _patched_put(url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = _get_random_user_agent()
    return _original_put(url, **kwargs)


def _patched_delete(url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = _get_random_user_agent()
    return _original_delete(url, **kwargs)


def _patched_patch(url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = _get_random_user_agent()
    return _original_patch(url, **kwargs)


requests.get = _patched_get
requests.post = _patched_post
requests.head = _patched_head
requests.put = _patched_put
requests.delete = _patched_delete
requests.patch = _patched_patch

# Patch urllib for lower-level requests (used by some libraries)
import urllib.request

_original_urlopen = urllib.request.urlopen


def _patched_urlopen(url, data=None, timeout=None, **kwargs):
    if isinstance(url, str):
        req = urllib.request.Request(url, data=data)
        req.add_header('User-Agent', _get_random_user_agent())
        return _original_urlopen(req, timeout=timeout)
    elif isinstance(url, urllib.request.Request):
        if not url.has_header('User-Agent') and not url.has_header('User-agent'):
            url.add_header('User-Agent', _get_random_user_agent())
        return _original_urlopen(url, data=data, timeout=timeout)
    return _original_urlopen(url, data=data, timeout=timeout, **kwargs)


urllib.request.urlopen = _patched_urlopen

# Set default opener with User-Agent
_opener = urllib.request.build_opener()
_opener.addheaders = [('User-Agent', _get_random_user_agent())]
urllib.request.install_opener(_opener)

# =============================================================================
# End User-Agent patch
# =============================================================================

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


class EvalScopeBackend:
    """Backend using EvalScope (formerly llmuses) for benchmark evaluation."""

    # Default retry configuration
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 5  # seconds

    # Map our benchmark names to EvalScope dataset names
    BENCHMARK_MAP = {
        # Standard benchmarks
        'mmlu': 'mmlu',
        'cmmlu': 'cmmlu',
        'c_eval': 'ceval',
        'gsm8k': 'gsm8k',
        'arc': 'arc',
        'arc_challenge': 'arc',
        'arc_easy': 'arc_easy',
        'hellaswag': 'hellaswag',
        'truthfulqa': 'truthful_qa',
        'winogrande': 'winogrande',
        'bbh': 'bbh',
        'humaneval': 'humaneval',
        'mbpp': 'mbpp',

        # EvalScope-specific benchmarks
        'bfcl': 'bfcl_v3',
        'bfcl_v3': 'bfcl_v3',
        'bfcl_v4': 'bfcl_v4',
        'tau_bench': 'tau_bench',
        'tau2_bench': 'tau2_bench',
        'longbench': 'longbench',
        'longbench_write': 'longbench_write',
        'toolbench': 'toolbench',

        # Mathematical benchmarks
        'math': 'math',
        'aime': 'aime_2024',
        'aime_2024': 'aime_2024',
        'aime_2025': 'aime_2025',

        # Multilingual
        'super_gpqa': 'super_gpqa',

        # Code
        'ifeval': 'ifeval',

        # QA
        'squad': 'squad',
        'drop': 'drop',
        'boolq': 'boolq',
        'lambada': 'lambada',
        'logiqa': 'logiqa',
        'mathqa': 'mathqa',

        # Multimodal/Vision benchmarks
        'mmmu': 'mmmu',
        'mmmu_pro': 'mmmu_pro',
        'mathvista': 'math_vista',
        'math_vista': 'math_vista',
        'chartqa': 'chartqa',
        'docvqa': 'docvqa',
        'infovqa': 'infovqa',
        'ai2d': 'ai2d',
        'seed_bench': 'seed_bench_2_plus',
        'mm_bench': 'mm_bench',
        'mm_star': 'mm_star',
        'pope': 'pope',
        'real_world_qa': 'real_world_qa',
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
            Path.home() / '.cache' / 'modelscope' / 'hub' / 'datasets' / 'downloads',
            Path('/root/.cache/modelscope/hub/datasets'),
            Path('/root/.cache/modelscope/hub/datasets/downloads'),
            Path.home() / '.cache' / 'huggingface' / 'datasets',
            Path('/root/.cache/huggingface/datasets'),
        ]

        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue

            try:
                # For downloads dir, clear everything (hash-based filenames)
                if 'downloads' in str(cache_dir):
                    for item in cache_dir.iterdir():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                            logger.debug(f"Cleared download cache: {item}")
                        except Exception as e:
                            logger.debug(f"Failed to clear {item}: {e}")
                else:
                    # For dataset dirs, look for dataset name or cais/mmlu pattern
                    for item in cache_dir.iterdir():
                        item_name = item.name.lower()
                        if dataset_name.lower() in item_name or 'cais' in item_name:
                            try:
                                if item.is_dir():
                                    shutil.rmtree(item)
                                else:
                                    item.unlink()
                                logger.debug(f"Cleared cache: {item}")
                            except Exception as e:
                                logger.debug(f"Failed to clear cache {item}: {e}")
            except Exception as e:
                logger.debug(f"Failed to process cache dir {cache_dir}: {e}")

    def evaluate(
            self,
            target: BaseTarget,
            benchmark_name: str,
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate target using EvalScope benchmark with retry logic.

        Args:
            target: Target to evaluate
            benchmark_name: Name of benchmark
            config: Evaluation configuration

        Returns:
            Evaluation results
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
                # Force redownload on retry attempts
                retry_config = config.copy()
                if attempt > 1:
                    retry_config['force_redownload'] = True

                # Prepare EvalScope task config
                task_config = self._prepare_task_config(target, evalscope_dataset, retry_config)

                logger.info(f"Running EvalScope task for dataset: {evalscope_dataset}")

                # Run the task
                results = run_task(task_cfg=task_config)

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

                # Parse results
                parsed_results = self._parse_results(results, benchmark_name)

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
                    import traceback
                    logger.debug(traceback.format_exc())
                    raise

        # If we get here, all retries failed
        logger.error(f"All {max_retries} attempts failed for benchmark: {benchmark_name}")
        raise last_error

    def _prepare_task_config(
            self,
            target: BaseTarget,
            dataset_name: str,
            config: Dict[str, Any]
    ) -> TaskConfig:
        """Prepare EvalScope TaskConfig."""
        eval_type = self._get_eval_type(target)

        # Base config
        task_cfg_dict = {
            'datasets': [dataset_name],
            'work_dir': tempfile.gettempdir(),
        }

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

        # Dataset path for custom datasets
        dataset_path = config.get('dataset_path') or config.get('path') or config.get('custom_dataset')
        if dataset_path:
            if not task_cfg_dict.get('dataset_args'):
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}

            task_cfg_dict['dataset_args'][dataset_name]['dataset_id'] = dataset_path
            logger.info(f"Set custom dataset_id to: {dataset_path}")

        # Dataset hub configuration
        dataset_hub = config.get('dataset_hub') or config.get('hub')
        if dataset_hub:
            task_cfg_dict['dataset_hub'] = dataset_hub

        # For local datasets, override the dataset_id in dataset_args
        if dataset_hub == 'local' and dataset_path:
            if not task_cfg_dict.get('dataset_args'):
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}

            task_cfg_dict['dataset_args'][dataset_name]['dataset_id'] = dataset_path
            logger.info(f"Set local dataset_id to: {dataset_path}")

        # Force redownload if specified (usually on retry)
        if config.get('force_redownload'):
            if not task_cfg_dict.get('dataset_args'):
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}
            task_cfg_dict['dataset_args'][dataset_name]['force_redownload'] = True
            logger.info("Forcing dataset redownload")

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
            if not task_cfg_dict.get('dataset_args'):
                task_cfg_dict['dataset_args'] = {}
            if dataset_name not in task_cfg_dict['dataset_args']:
                task_cfg_dict['dataset_args'][dataset_name] = {}

            if isinstance(subset, str):
                subset = [subset]
            task_cfg_dict['dataset_args'][dataset_name]['subset_list'] = subset

        # Add backend params for concurrency and batching
        backend_params = config.get('backend_params', {})

        # Generation config - set defaults or use from backend_params
        if 'generation_config' not in task_cfg_dict:
            task_cfg_dict['generation_config'] = {
                'batch_size': backend_params.get('generation_batch_size', 1),
                'max_tokens': backend_params.get('max_tokens', 512),
                'temperature': backend_params.get('temperature', 0.0)
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

        return TaskConfig(**task_cfg_dict)

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

    def _parse_results(
            self,
            results: Dict[str, Any],
            benchmark_name: str
    ) -> Dict[str, Any]:
        """
        Parse EvalScope results into standardized format.

        Args:
            results: Raw EvalScope results from JSON file
            benchmark_name: Benchmark name

        Returns:
            Parsed results dictionary
        """
        task_results = {}
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
                    subset_score = subset.get('score', 0.0)
                    subset_num = subset.get('num', 0)

                    if subset_name:
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
            'metadata': {
                'backend': 'evalscope',
                'benchmark': benchmark_name,
                'num_datasets': len(task_results),
                'dataset_name': results.get('dataset_name', benchmark_name),
                'model_name': results.get('model_name', 'unknown'),
            }
        }

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
