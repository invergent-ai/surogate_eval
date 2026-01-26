# surogate/eval/benchmarks/backends/lm_eval_backend.py
"""LM Evaluation Harness backend for custom benchmark evaluation."""
import os
import ssl
from pathlib import Path

import yaml

# Must be set BEFORE any huggingface imports
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = '1'
os.environ['HF_ALLOW_CODE_EVAL'] = '1'
os.environ['TRUST_REMOTE_CODE'] = '1'

# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''

from typing import Dict, Any, List, Optional

from surogate_eval.targets import BaseTarget
from surogate_eval.utils.logger import get_logger

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests to disable SSL verification
import requests
from requests.adapters import HTTPAdapter

class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = ssl.create_default_context()
        kwargs['ssl_context'].check_hostname = False
        kwargs['ssl_context'].verify_mode = ssl.CERT_NONE
        return super().init_poolmanager(*args, **kwargs)

    def send(self, request, *args, **kwargs):
        kwargs['verify'] = False
        return super().send(request, *args, **kwargs)

# Monkey-patch requests.Session to always use our adapter
_original_session_init = requests.Session.__init__

def _patched_session_init(self, *args, **kwargs):
    _original_session_init(self, *args, **kwargs)
    self.verify = False
    adapter = SSLAdapter()
    self.mount('https://', adapter)
    self.mount('http://', adapter)

requests.Session.__init__ = _patched_session_init

# Also patch the default session
requests.packages.urllib3.disable_warnings()

try:
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    import lm_eval.tasks

    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    TaskManager = None

logger = get_logger()


class LMEvalBackend:
    """Backend using lm-evaluation-harness for custom benchmark evaluation."""

    TASK_TYPE_MAP = {
        'multiple_choice': 'multiple_choice',
        'generation': 'generate_until',
    }

    def __init__(self):
        if not LM_EVAL_AVAILABLE:
            raise ImportError(
                "lm-evaluation-harness is not installed. "
                "Install with: pip install lm-eval"
            )
        self._copied_files = []
        logger.debug("Initialized LM Evaluation Harness backend")

    def _is_hf_dataset(self, source: str) -> bool:
        """Check if source is a HuggingFace dataset path."""
        if not source:
            return False
        if source.startswith(('.', '/', '~')) or Path(source).suffix:
            return False
        return '/' in source or (not os.path.exists(source))

    def _get_lm_eval_tasks_dir(self) -> Path:
        """Get the lm-eval tasks directory."""
        return Path(lm_eval.tasks.__file__).parent

    def _generate_task_yaml(self, config: Dict[str, Any]) -> str:
        """Generate lm-eval task YAML and copy it to lm-eval tasks directory."""
        source = config.get('source')
        task_type = config.get('task_type', 'generation')
        columns = config.get('columns', {})
        choices_columns = config.get('choices_columns')
        choices_labels = config.get('choices_labels')
        split = config.get('split', 'test')
        subset = config.get('subset')
        prompt_template = config.get('prompt_template')
        system_prompt = config.get('system_prompt')
        num_fewshot = config.get('num_fewshot', 0)
        task_name = config.get('name', 'custom_task')
        max_tokens = config.get('max_tokens', 20)

        is_hf = self._is_hf_dataset(source)

        task_config = {
            'task': task_name,
            'dataset_path': source if is_hf else 'json',
            'test_split': split,
            'num_fewshot': num_fewshot,
        }

        # Only add fewshot_split if using HF dataset AND num_fewshot > 0
        if is_hf and num_fewshot > 0:
            task_config['fewshot_split'] = 'validation'

        if not is_hf:
            source_path = Path(source).absolute()
            target_split = split

            if source_path.suffix in ['.jsonl', '.json']:
                task_config['dataset_path'] = 'json'
                task_config['dataset_kwargs'] = {'data_files': {target_split: str(source_path)}}
            elif source_path.suffix == '.csv':
                task_config['dataset_path'] = 'csv'
                task_config['dataset_kwargs'] = {'data_files': {target_split: str(source_path)}}
            elif source_path.suffix == '.parquet':
                task_config['dataset_path'] = 'parquet'
                task_config['dataset_kwargs'] = {'data_files': {target_split: str(source_path)}}
            else:
                raise ValueError(f"Unsupported file format: {source_path.suffix}")

            task_config['test_split'] = target_split

        if is_hf and subset:
            task_config['dataset_name'] = subset

        question_col = columns.get('question', 'question')
        answer_col = columns.get('answer', 'answer')

        # Build doc_to_text
        if prompt_template:
            task_config['doc_to_text'] = prompt_template
        elif choices_columns and choices_labels:
            choices_text = "\\n".join([
                f"{label}. {{{{{col}}}}}"
                for label, col in zip(choices_labels, choices_columns)
            ])
            base_prompt = f"{{{{{question_col}}}}}\\n{choices_text}\\nAnswer:"

            if system_prompt:
                task_config['doc_to_text'] = f"{system_prompt}\\n\\n{base_prompt}"
            else:
                task_config['doc_to_text'] = base_prompt
        elif choices_columns:
            default_labels = [chr(65 + i) for i in range(len(choices_columns))]
            choices_text = "\\n".join([
                f"{label}. {{{{{col}}}}}"
                for label, col in zip(default_labels, choices_columns)
            ])
            base_prompt = f"{{{{{question_col}}}}}\\n{choices_text}\\nAnswer:"

            if system_prompt:
                task_config['doc_to_text'] = f"{system_prompt}\\n\\n{base_prompt}"
            else:
                task_config['doc_to_text'] = base_prompt
        else:
            base_prompt = f"{{{{{question_col}}}}}\\nAnswer:"

            if system_prompt:
                task_config['doc_to_text'] = f"{system_prompt}\\n\\n{base_prompt}"
            else:
                task_config['doc_to_text'] = base_prompt

        # Generation config
        task_config['output_type'] = 'generate_until'
        task_config['generation_kwargs'] = {
            'until': config.get('stop_sequences') or ['\n'],
            'max_gen_toks': max_tokens,
        }
        task_config['doc_to_target'] = "{{" + answer_col + "}}"

        # Metrics - use regexes_to_ignore to strip \boxed{} wrapper
        task_config['metric_list'] = [
            {
                'metric': 'exact_match',
                'aggregation': 'mean',
                'higher_is_better': True,
                'ignore_case': True,
                'ignore_punctuation': True,
                'regexes_to_ignore': [
                    r'^\s+',  # Leading whitespace
                    r'\s+$',  # Trailing whitespace
                    r'\\n.*',  # Everything after \n (escaped)
                    r'\n.*',  # Everything after newline
                    r'.*\\boxed\{',  # Before \boxed{
                    r'\}.*',  # After }
                ],
            }
        ]

        lm_eval_tasks_dir = self._get_lm_eval_tasks_dir()
        yaml_path = lm_eval_tasks_dir / f"{task_name}.yaml"
        logger.info(f"=== TASK CONFIG DEBUG ===")
        logger.info(f"Task name: {task_name}")
        logger.info(f"choices_columns: {choices_columns}")
        logger.info(f"choices_labels: {choices_labels}")
        logger.info(f"doc_to_text: {task_config.get('doc_to_text')}")
        logger.info(f"=========================")
        with open(yaml_path, 'w') as f:
            yaml.dump(task_config, f, default_flow_style=False, allow_unicode=True)

        self._copied_files.append(yaml_path)

        logger.info(f"Generated task config:\n{yaml.dump(task_config, default_flow_style=False)}")

        return task_name

    # In evaluate() method, around line 280, replace the model selection logic:

    def evaluate(
            self,
            target: BaseTarget,
            benchmark_name: str,
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate target using lm-evaluation-harness with custom dataset."""
        logger.info(f"Running lm-eval for benchmark: {benchmark_name}")

        # Resolve LakeFS tokenizer path
        tokenizer = config.get('tokenizer') or target.config.get('tokenizer')
        if tokenizer and tokenizer.startswith('lakefs://'):
            from surogate_eval.datasets import DatasetLoader
            loader = DatasetLoader()
            tokenizer = loader._download_from_lakefs(tokenizer)
            logger.info(f"Resolved LakeFS tokenizer to: {tokenizer}")
            config['tokenizer'] = tokenizer

        source = config.get('source')
        if not source:
            raise ValueError("'source' is required for lm-eval backend")

        task_name = self._generate_task_yaml({
            'name': benchmark_name,
            'source': source,
            'task_type': config.get('task_type', 'multiple_choice'),
            'columns': config.get('columns', {}),
            'choices_columns': config.get('choices_columns'),
            'choices_labels': config.get('choices_labels'),
            'split': config.get('split', 'test'),
            'subset': config.get('subset'),
            'prompt_template': config.get('prompt_template'),
            'num_fewshot': config.get('num_fewshot', 0),
            'stop_sequences': config.get('stop_sequences'),
        })

        base_url = target.config.get('base_url', '')
        api_key = target.config.get('api_key', '')
        model_name = target.config.get('model')

        # Detect if this is OpenAI API
        is_openai = 'api.openai.com' in base_url or (
                target.config.get('provider') == 'openai' and not base_url
        )

        if is_openai:
            # Use openai-completions for OpenAI API
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key

            model_args_str = f"model={model_name}"

            eval_args = {
                'model': 'openai-completions',
                'model_args': model_args_str,
                'tasks': [task_name],
                'batch_size': config.get('batch_size') or 1,
                'log_samples': config.get('log_samples', True),
                'confirm_run_unsafe_code': True,
            }
        else:
            # Use local-completions for vLLM/local endpoints
            base_url = base_url.rstrip('/')
            if not base_url.endswith('/v1/completions'):
                if base_url.endswith('/v1'):
                    base_url = base_url + '/completions'
                else:
                    base_url = base_url + '/v1/completions'

            model_args_dict = {
                'model': model_name,
                'base_url': base_url,
                'tokenizer_backend': 'huggingface',
            }

            tokenizer = config.get('tokenizer') or target.config.get('tokenizer')
            logger.info(
                f"DEBUG: tokenizer from config={config.get('tokenizer')}, from target={target.config.get('tokenizer')}, resolved={tokenizer}")
            if tokenizer:
                model_args_dict['tokenizer'] = tokenizer

            if api_key and api_key != 'EMPTY':
                model_args_dict['api_key'] = api_key

            num_concurrent = config.get('num_concurrent') or config.get('backend_params', {}).get('num_concurrent', 1)
            if num_concurrent and num_concurrent > 1:
                model_args_dict['num_concurrent'] = num_concurrent

            model_args_str = ','.join(f'{k}={v}' for k, v in model_args_dict.items())

            eval_args = {
                'model': 'local-completions',
                'model_args': model_args_str,
                'tasks': [task_name],
                'batch_size': config.get('batch_size') or 1,
                'log_samples': config.get('log_samples', True),
                'confirm_run_unsafe_code': True,
            }

        if config.get('limit'):
            eval_args['limit'] = config['limit']

        if config.get('num_fewshot') is not None:
            eval_args['num_fewshot'] = config['num_fewshot']

        gen_kwargs = {}
        if config.get('max_tokens'):
            gen_kwargs['max_gen_toks'] = config['max_tokens']
        if config.get('temperature') is not None:
            gen_kwargs['temperature'] = config['temperature']
        if gen_kwargs:
            eval_args['gen_kwargs'] = ','.join(f'{k}={v}' for k, v in gen_kwargs.items())

        logger.info(f"lm-eval args: {eval_args}")
        logger.info(f"Source: {source}")

        try:
            results = evaluator.simple_evaluate(**eval_args)
        except Exception as e:
            logger.error(f"lm-eval simple_evaluate failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

            return {
                'overall_score': 0.0,
                'num_samples': 0,
                'task_results': {},
                'metadata': {
                    'backend': 'lm_eval',
                    'benchmark': benchmark_name,
                    'source': source,
                    'status': 'failed',
                    'error': str(e),
                }
            }
        finally:
            self.cleanup()

        return self._parse_results(results, benchmark_name, source)

    def _extract_answer(self, output: str) -> str:
        """Extract answer from model output."""
        import re

        if not output:
            return ""

        output = output.strip()

        # Get only the first line (handle both escaped and actual newlines)
        first_line = re.split(r'\\n|\n', output)[0].strip()

        # Remove trailing quotes/commas that lm-eval sometimes adds
        first_line = re.sub(r'["\',]+$', '', first_line)

        # Extract from \boxed{X}
        boxed_match = re.search(r'\\boxed\{([^\}]+)\}', first_line)
        if boxed_match:
            return boxed_match.group(1).strip()

        # For MCQ: if starts with letter A-D followed by delimiter, extract just the letter
        letter_match = re.match(r'^([A-Da-d])(?:[\.\,\)\s]|$)', first_line)
        if letter_match:
            return letter_match.group(1).upper()

        return first_line

    def _parse_results(
            self,
            results: Dict[str, Any],
            benchmark_name: str,
            source: str
    ) -> Dict[str, Any]:
        """Parse lm-eval results into standardized format."""
        task_results = {}
        total_score = 0.0
        total_samples = 0
        num_tasks = 0

        for task, metrics in results.get('results', {}).items():
            score = None

            # Dynamic detection: get first numeric non-stderr metric
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and 'stderr' not in k and k != 'alias':
                    score = v
                    break

            if score is not None:
                n_samples = metrics.get('samples', 0)
                task_results[task] = {
                    'score': score,
                    'accuracy': score,
                    'n_samples': n_samples,
                    'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                }
                total_score += score
                total_samples += n_samples
                num_tasks += 1

        overall_score = total_score / num_tasks if num_tasks > 0 else 0.0

        detailed_results = []
        for task, samples in results.get('samples', {}).items():
            for sample in samples:
                # Extract raw output - handle nested structures
                filtered_resps = sample.get('filtered_resps', [])
                if filtered_resps:
                    raw_output = filtered_resps[0]
                    # Handle tuple inside list: [('output',)]
                    if isinstance(raw_output, (list, tuple)):
                        raw_output = raw_output[0] if raw_output else ''
                else:
                    raw_output = ''

                raw_output = str(raw_output) if raw_output else ''
                extracted = self._extract_answer(raw_output)

                detailed_results.append({
                    'input': sample.get('doc', {}),
                    'expected': sample.get('target', ''),
                    'output': extracted,
                    'raw_output': raw_output,
                    'metrics': {k: v for k, v in sample.items() if k in ['exact_match', 'acc']}
                })

        return {
            'overall_score': overall_score,
            'num_samples': total_samples,
            'task_results': task_results,
            'detailed_results': detailed_results,
            'metadata': {
                'backend': 'lm_eval',
                'benchmark': benchmark_name,
                'source': source,
                'num_tasks': num_tasks,
                'status': 'completed',
            }
        }

    def cleanup(self):
        """Clean up copied task files."""
        for filepath in self._copied_files:
            try:
                if filepath.exists():
                    filepath.unlink()
                    logger.debug(f"Cleaned up: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {filepath}: {e}")
        self._copied_files = []

    def __del__(self):
        self.cleanup()


LMEvalBackendWrapper = LMEvalBackend