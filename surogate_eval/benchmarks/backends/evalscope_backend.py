# surogate/eval/benchmarks/backends/evalscope_backend.py
"""EvalScope backend for benchmark evaluation."""
import tempfile
import time
from typing import Dict, Any, List
from pathlib import Path

import requests

_original_request = requests.Session.request


def _patched_request(self, method, url, **kwargs):
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'User-Agent' not in kwargs['headers'] and 'user-agent' not in kwargs['headers']:
        kwargs['headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    return _original_request(self, method, url, **kwargs)


requests.Session.request = _patched_request

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
        'humaneval_plus': 'humaneval_plus',
        'mbpp': 'mbpp',
        'mbpp_plus': 'mbpp_plus',
        'ds1000': 'ds1000',
        'leetcode': 'leetcode',

        # EvalScope-specific benchmarks
        'bfcl': 'bfcl_v3',
        'bfcl_v3': 'bfcl_v3',
        'bfcl_v4': 'bfcl_v4',
        'tau_bench': 'tau_bench',
        'tau2_bench': 'tau2_bench',
        'longbench': 'longbench',
        'longbench_write': 'longbench_write',
        'toolbench': 'tool_bench',

        # Mathematical benchmarks
        'math': 'competition_math',
        'aime': 'aime24',
        'aime_2024': 'aime24',
        'aime_2025': 'aime25',

        # Multilingual
        'super_gpqa': 'super_gpqa',

        # Code
        'ifeval': 'ifeval',

        # QA
        'squad': 'squad',
        'drop': 'drop',
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
                    import traceback
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

        for review_file in reviews_dir.glob(f'{dataset_name}*.jsonl'):
            logger.debug(f"Loading reviews from: {review_file}")
            try:
                with open(review_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)

                            # Extract score from EvalScope's nested structure
                            # sample_score.score.value.acc
                            score = 0.0
                            sample_score = sample.get('sample_score', {})
                            if sample_score:
                                score_obj = sample_score.get('score', {})
                                if score_obj:
                                    value = score_obj.get('value', {})
                                    if isinstance(value, dict):
                                        # Get 'acc' or first available metric
                                        score = value.get('acc', 0.0)
                                        if score == 0.0:
                                            # Try other common metric names
                                            for key in ['accuracy', 'correct', 'score']:
                                                if key in value:
                                                    score = value[key]
                                                    break
                                    elif isinstance(value, (int, float)):
                                        score = float(value)

                            # Extract input, target, and prediction
                            input_text = sample.get('input', '')
                            expected = sample.get('target', '')

                            # Get the model's prediction/output
                            prediction = ''
                            extracted = ''
                            if sample_score:
                                score_obj = sample_score.get('score', {})
                                prediction = score_obj.get('prediction', '')
                                extracted = score_obj.get('extracted_prediction', '')

                            detailed_results.append({
                                'input': input_text,
                                'expected': expected,
                                'output': extracted or prediction[:500],  # Use extracted answer or truncated prediction
                                'raw_output': prediction,
                                'score': float(score),
                                'success': float(score) > 0,
                                'subset': sample.get('sample_score', {}).get('sample_metadata', {}).get('subject', ''),
                                'metadata': sample
                            })

            except Exception as e:
                logger.warning(f"Failed to load reviews from {review_file}: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        logger.info(f"Loaded {len(detailed_results)} detailed predictions")
        return detailed_results

    def _is_reasoning_model(self, target: BaseTarget) -> bool:
        """
        Detect if model uses reasoning/thinking by probing it.
        Caches result to avoid repeated probes.
        """
        cache_key = f"_is_reasoning_{target.name}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        logger.info(f"Probing model to detect reasoning mode...")

        try:
            # Use httpx directly with the correct URL
            import httpx

            base_url = target.config.get('base_url', '')
            if not base_url.endswith('/v1'):
                api_url = f"{base_url}/v1" if not base_url.endswith('/') else f"{base_url}v1"
            else:
                api_url = base_url

            model = target.config.get('model', '')
            api_key = target.config.get('api_key', 'EMPTY')

            with httpx.Client(verify=False, timeout=30) as client:
                response = client.post(
                    f"{api_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
                        "max_tokens": 150,
                    }
                )
                response.raise_for_status()
                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            logger.debug(f"Probe response: {content[:200]}...")

            is_reasoning = (
                    '<think>' in content or
                    '</think>' in content or
                    '<reasoning>' in content
            )

            setattr(self, cache_key, is_reasoning)

            if is_reasoning:
                logger.info(f"Detected reasoning model (uses <think> tags)")
            else:
                logger.info(f"Standard model detected (no <think> tags)")

            return is_reasoning

        except Exception as e:
            logger.warning(f"Could not probe model for reasoning detection: {e}")
            return False

    def _estimate_max_tokens(
            self,
            target: BaseTarget,
            dataset_name: str,
            config: Dict[str, Any]
    ) -> int:
        """
        Estimate max_tokens needed based on model behavior and dataset.
        User-specified max_tokens acts as a cap, not an override.
        """
        logger.info(f"Auto-detecting max_tokens for {dataset_name}...")

        # Detect if reasoning model by probing
        is_reasoning = self._is_reasoning_model(target)

        num_fewshot = config.get('num_fewshot', 5)

        # Base tokens: ~300 per few-shot example + 512 for answer
        base = 300 * max(1, num_fewshot) + 512

        if is_reasoning:
            # Reasoning models need 4-6x for thinking + answer
            estimated = min(base * 5, 8192)
            logger.info(f"Reasoning model - estimated max_tokens: {estimated}")
        else:
            estimated = min(base, 2048)
            logger.info(f"Standard model - estimated max_tokens: {estimated}")

        # User-specified max_tokens acts as a CAP (upper limit)
        user_max = config.get('max_tokens')
        if user_max:
            if estimated > user_max:
                logger.warning(
                    f"Estimated max_tokens ({estimated}) exceeds user cap ({user_max}). "
                    f"Using cap - model output may be truncated."
                )
                return user_max
            else:
                logger.debug(f"Estimated max_tokens ({estimated}) within user cap ({user_max})")

        return estimated

    def _estimate_from_samples(
            self,
            dataset_name: str,
            config: Dict[str, Any],
            tokenizer_name: str = None,
            is_reasoning: bool = False
    ) -> int | None:
        """
        Estimate tokens by sampling actual dataset prompts.
        Returns None if sampling fails.
        """
        try:
            from datasets import load_dataset

            # Load a few samples
            subset = config.get('subset')
            if isinstance(subset, list):
                subset = subset[0] if subset else None

            dataset_id = self.BENCHMARK_MAP.get(dataset_name, dataset_name)

            # Try loading from HuggingFace
            ds = load_dataset(
                f"cais/{dataset_id}" if dataset_id == 'mmlu' else dataset_id,
                subset,
                split='test',
                trust_remote_code=True
            )

            samples = list(ds.select(range(min(10, len(ds)))))

            if not samples:
                return None

            # Get tokenizer
            tokenizer = None
            if tokenizer_name:
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                except Exception:
                    pass

            # Calculate max input tokens across samples
            max_input_tokens = 0
            for sample in samples:
                # Build approximate prompt
                text = sample.get('question', '') or sample.get('input', '') or str(sample)

                if tokenizer:
                    tokens = len(tokenizer.encode(text))
                else:
                    tokens = len(text) // 4  # Rough estimate

                max_input_tokens = max(max_input_tokens, tokens)

            # Account for few-shot examples (each ~same size as sample)
            num_fewshot = config.get('num_fewshot', 5)
            total_input = max_input_tokens * (num_fewshot + 1)

            # Output estimation
            if is_reasoning:
                # Thinking: 3x input, Answer: 1x input
                output_tokens = max_input_tokens * 4
            else:
                output_tokens = max(256, max_input_tokens)

            # Add 20% buffer
            result = int(output_tokens * 1.2)

            # Clamp to reasonable range
            return max(512, min(result, 8192))

        except Exception as e:
            logger.debug(f"Sample-based estimation failed: {e}")
            return None


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
            'detailed_results': detailed_results or [],
            'metadata': {
                'backend': 'evalscope',
                'benchmark': benchmark_name,
                'num_datasets': len(task_results),
                'dataset_name': results.get('dataset_name', benchmark_name),
                'model_name': results.get('model_name', 'unknown'),
            }
        }

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
        }

        # Enable sandbox for code benchmarks
        if dataset_name.lower() in CODE_BENCHMARKS:
            task_cfg_dict['use_sandbox'] = True
            task_cfg_dict['sandbox_type'] = 'docker'
            logger.info(f"Enabling sandbox for code benchmark: {dataset_name}")

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

            # If local dataset has no train split, use test for few-shot examples
            if available_splits is not None and 'train' not in available_splits:
                if 'test' in available_splits or not available_splits:
                    task_cfg_dict['dataset_args'][dataset_name]['train_split'] = 'test'
                    logger.info(f"No train split found, using test split for few-shot examples")

            logger.info(f"Set custom dataset_id to: {dataset_path}")

        if dataset_hub:
            task_cfg_dict['dataset_hub'] = dataset_hub

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

        # Add backend params for concurrency and batching
        backend_params = config.get('backend_params', {})
        estimated_tokens = self._estimate_max_tokens(target, dataset_name, config)

        # Generation config - set defaults or use from backend_params
        if 'generation_config' not in task_cfg_dict:
            task_cfg_dict['generation_config'] = {
                'batch_size': backend_params.get('generation_batch_size', 1),
                'max_tokens': estimated_tokens,
                'temperature': config.get('temperature', 0.0),
            }
        logger.info(f"Using max_tokens: {estimated_tokens}")

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

            # Detect available splits from filenames
            available_splits = set()
            for f in downloaded_files:
                if f.is_file():
                    stem = f.stem.lower()
                    if stem in ('train', 'test', 'validation', 'val', 'dev'):
                        available_splits.add(stem)
                    logger.debug(f"  - {f}")

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
