# surogate/eval/benchmarks/backends/custom_eval_backend.py
"""Custom evaluation backend supporting mixed exact_match and judge evaluation types."""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from surogate_eval.targets import BaseTarget
from surogate_eval.utils.logger import get_logger

logger = get_logger()

try:
    from datasets import load_dataset, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False


class CustomEvalBackend:
    """
    Backend for custom evaluation with mixed eval types.

    Dataset schema (simple):
    - instruction (required): The full prompt including any choices
    - answer (required): Expected answer
    - eval_type (optional): 'judge' or 'exact_match' (default: exact_match)
    - judge_criteria (optional): Per-row criteria for judge eval
    """

    def __init__(self):
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        logger.debug("Initialized CustomEvalBackend")

    def _load_dataset(
            self,
            source: str,
            split: str = 'test',
            limit: Optional[int] = None
    ) -> Dataset:
        """Load dataset from HuggingFace, LakeFS, or local file."""
        logger.info(f"Loading dataset from: {source}")

        # Handle LakeFS URLs
        if source.startswith('lakefs://'):
            from surogate_eval.datasets import DatasetLoader
            loader = DatasetLoader()
            local_path = loader._download_from_lakefs(source)
            logger.info(f"Downloaded LakeFS dataset to: {local_path}")
            source = local_path

        # Check if HuggingFace dataset
        source_path = Path(source)
        if not source_path.exists() and '/' in source:
            # HuggingFace dataset
            dataset = load_dataset(source, split=split, trust_remote_code=True)
            logger.info(f"Loaded HF dataset '{source}' split '{split}'")
        else:
            # Local file
            if not source_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {source}")

            suffix = source_path.suffix.lower()
            if suffix in ['.jsonl', '.json']:
                dataset = load_dataset('json', data_files=str(source_path), split='train')
            elif suffix == '.csv':
                dataset = load_dataset('csv', data_files=str(source_path), split='train')
            elif suffix == '.parquet':
                dataset = load_dataset('parquet', data_files=str(source_path), split='train')
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            logger.info(f"Loaded local dataset: {source_path}")

        if limit and limit < len(dataset):
            dataset = dataset.select(range(limit))
            logger.info(f"Limited dataset to {limit} rows")

        return dataset

    def _get_column_value(self, row: Dict[str, Any], columns: Dict[str, str], key: str, default: Any = None) -> Any:
        """Get column value using column mapping."""
        column_name = columns.get(key, key)
        value = row.get(column_name)
        if value is None or (isinstance(value, str) and value.lower() == 'null'):
            return default
        return value

    def _extract_answer(self, output: str) -> str:
        """Extract answer from model output."""
        if not output:
            return ""

        output = output.strip()

        # Get first line
        first_line = re.split(r'\\n|\n', output)[0].strip()

        # Extract from \boxed{X}
        boxed_match = re.search(r'\\boxed\{([^\}]+)\}', first_line)
        if boxed_match:
            return boxed_match.group(1).strip()

        # For MCQ: extract letter A-D
        letter_match = re.match(r'^([A-Da-d])(?:[\.\,\)\s]|$)', first_line)
        if letter_match:
            return letter_match.group(1).upper()

        return first_line

    def _check_exact_match(self, output: str, expected: str) -> bool:
        """Check if output matches expected answer."""
        output = self._extract_answer(output).lower().strip()
        expected = expected.lower().strip()
        return output == expected

    def _evaluate_exact_match(
            self,
            row: Dict[str, Any],
            target: BaseTarget,
            columns: Dict[str, str],
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single row with exact match."""
        instruction = self._get_column_value(row, columns, 'instruction', '')
        expected = self._get_column_value(row, columns, 'answer', '')

        # Get model output
        try:
            from surogate_eval.targets.base import TargetRequest
            request = TargetRequest(prompt=instruction)
            response = target.send_request(request)
            raw_output = response.content
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'instruction': instruction,
                'expected': expected,
                'output': '',
                'raw_output': '',
                'score': 0.0,
                'success': False,
                'reason': f'Inference error: {str(e)}',
            }

        output = self._extract_answer(raw_output)
        success = self._check_exact_match(raw_output, expected)

        return {
            'instruction': instruction,
            'expected': expected,
            'output': output,
            'raw_output': raw_output,
            'score': 1.0 if success else 0.0,
            'success': success,
            'reason': 'Exact match' if success else 'No match',
        }

    def _evaluate_judge(
            self,
            row: Dict[str, Any],
            target: BaseTarget,
            columns: Dict[str, str],
            config: Dict[str, Any],
            judge_target: Optional[BaseTarget]
    ) -> Dict[str, Any]:
        """Evaluate a single row with LLM judge."""
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("deepeval is required for judge evaluation")

        instruction = self._get_column_value(row, columns, 'instruction', '')
        expected = self._get_column_value(row, columns, 'answer', '')
        criteria = self._get_column_value(row, columns, 'judge_criteria') or config.get(
            'judge_criteria',
            'Evaluate if the response correctly answers the question based on the expected answer.'
        )

        # Get model output
        try:
            from surogate_eval.targets.base import TargetRequest
            request = TargetRequest(prompt=instruction)
            response = target.send_request(request)
            actual_output = response.content
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'instruction': instruction,
                'expected': expected,
                'output': '',
                'score': 0.0,
                'success': False,
                'reason': f'Inference error: {str(e)}',
                'criteria': criteria,
            }

        # Run G-Eval
        try:
            judge_model = None
            if judge_target:
                from surogate_eval.models.deepeval_wrapper import DeepEvalTargetWrapper
                judge_model = DeepEvalTargetWrapper(judge_target)

            metric = GEval(
                name="judge",
                criteria=criteria,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                model=judge_model,
            )

            test_case = LLMTestCase(
                input=instruction,
                actual_output=actual_output,
                expected_output=expected,
            )

            metric.measure(test_case, _show_indicator=False)

            return {
                'instruction': instruction,
                'expected': expected,
                'output': actual_output,
                'score': metric.score,
                'success': metric.score >= 0.5,
                'reason': getattr(metric, 'reason', None),
                'criteria': criteria,
            }

        except Exception as e:
            logger.error(f"G-Eval failed: {e}")
            return {
                'instruction': instruction,
                'expected': expected,
                'output': actual_output,
                'score': 0.0,
                'success': False,
                'reason': f'Judge error: {str(e)}',
                'criteria': criteria,
            }

    def evaluate(
            self,
            target: BaseTarget,
            benchmark_name: str,
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate target using custom evaluation.

        Config:
            - source: Dataset path (HF, LakeFS, or local file)
            - columns: Column mappings (instruction, answer, eval_type, judge_criteria)
            - split: Dataset split
            - limit: Max rows
            - judge_criteria: Default criteria for judge rows
        """
        logger.info(f"Running custom evaluation: {benchmark_name}")

        source = config.get('source')
        if not source:
            raise ValueError("'source' is required")

        columns = config.get('columns', {})
        split = config.get('split', 'test')
        limit = config.get('limit')

        # Load dataset
        dataset = self._load_dataset(source, split, limit)

        # Validate columns
        instruction_col = columns.get('instruction', 'instruction')
        answer_col = columns.get('answer', 'answer')

        if instruction_col not in dataset.column_names:
            raise ValueError(f"Column '{instruction_col}' not found in dataset")
        if answer_col not in dataset.column_names:
            raise ValueError(f"Column '{answer_col}' not found in dataset")

        # Check if eval_type column exists, default all to exact_match if not
        eval_type_col = columns.get('eval_type', 'eval_type')
        has_eval_type = eval_type_col in dataset.column_names

        judge_target = config.get('backend_params', {}).get('judge_target')

        # Evaluate each row
        results = []
        exact_match_results = []
        judge_results = []

        for idx, row in enumerate(dataset):
            eval_type = 'exact_match'
            if has_eval_type:
                eval_type = self._get_column_value(row, columns, 'eval_type', 'exact_match')

            if eval_type == 'judge':
                result = self._evaluate_judge(row, target, columns, config, judge_target)
                result['eval_type'] = 'judge'
                judge_results.append(result)
            else:
                result = self._evaluate_exact_match(row, target, columns, config)
                result['eval_type'] = 'exact_match'
                exact_match_results.append(result)

            result['original_idx'] = idx
            results.append(result)

            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(dataset)}")

        # Calculate metrics
        total = len(results)
        em_total = len(exact_match_results)
        judge_total = len(judge_results)

        em_correct = sum(1 for r in exact_match_results if r['success'])
        judge_avg = sum(r['score'] for r in judge_results) / judge_total if judge_total else 0.0

        overall_score = 0.0
        if total > 0:
            overall_score = (
                (em_correct / em_total if em_total else 0.0) * em_total +
                judge_avg * judge_total
            ) / total

        logger.info(f"Completed: {em_correct}/{em_total} exact_match, {judge_avg:.2f} avg judge score")

        return {
            'overall_score': overall_score,
            'num_samples': total,
            'task_results': {
                'exact_match': {
                    'total': em_total,
                    'correct': em_correct,
                    'accuracy': em_correct / em_total if em_total else 0.0,
                },
                'judge': {
                    'total': judge_total,
                    'avg_score': judge_avg,
                    'success_rate': sum(1 for r in judge_results if r['success']) / judge_total if judge_total else 0.0,
                },
            },
            'detailed_results': results,
            'metadata': {
                'backend': 'custom_eval',
                'benchmark': benchmark_name,
                'source': source,
                'split': split,
                'num_exact_match': em_total,
                'num_judge': judge_total,
                'status': 'completed',
            }
        }


CustomEvalBackendWrapper = CustomEvalBackend