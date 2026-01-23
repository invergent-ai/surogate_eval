# surogate/eval/benchmarks/backends/custom_eval_backend.py
"""Custom evaluation backend supporting mixed exact_match and judge evaluation types."""

import os
import json
import tempfile
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

    Dataset schema:
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
            dataset = load_dataset(source, split=split, trust_remote_code=True)
            logger.info(f"Loaded HF dataset '{source}' split '{split}'")
        else:
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

    def _split_by_eval_type(
            self,
            dataset: Dataset,
            columns: Dict[str, str]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset rows by eval_type."""
        exact_match_rows = []
        judge_rows = []

        eval_type_col = columns.get('eval_type', 'eval_type')
        has_eval_type = eval_type_col in dataset.column_names

        for idx, row in enumerate(dataset):
            row_dict = dict(row)
            row_dict['_original_idx'] = idx

            if has_eval_type:
                eval_type = self._get_column_value(row, columns, 'eval_type', 'exact_match')
            else:
                eval_type = 'exact_match'

            if eval_type == 'judge':
                judge_rows.append(row_dict)
            else:
                exact_match_rows.append(row_dict)

        logger.info(f"Split dataset: {len(exact_match_rows)} exact_match, {len(judge_rows)} judge")
        return exact_match_rows, judge_rows

    def _evaluate_exact_match_rows(
            self,
            rows: List[Dict[str, Any]],
            target: BaseTarget,
            config: Dict[str, Any],
            columns: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Evaluate exact_match rows using LM-Eval backend."""
        if not rows:
            return []

        logger.info(f"Evaluating {len(rows)} exact_match rows with lm-eval")

        from .lm_eval_backend import LMEvalBackend

        # Prepare rows for lm-eval format
        lm_eval_rows = []
        for row in rows:
            lm_row = {
                'instruction': self._get_column_value(row, columns, 'instruction', ''),
                'answer': self._get_column_value(row, columns, 'answer', ''),
                '_original_idx': row['_original_idx']
            }
            lm_eval_rows.append(lm_row)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for row in lm_eval_rows:
                f.write(json.dumps(row) + '\n')
            temp_path = f.name

        try:
            # Configure lm-eval backend
            lm_config = {
                'source': temp_path,
                'columns': {
                    'question': 'instruction',
                    'answer': 'answer',
                },
                'split': 'test',
                'num_fewshot': 0,
                'max_tokens': config.get('max_tokens', 256),
                'tokenizer': config.get('tokenizer') or target.config.get('tokenizer'),
                'batch_size': config.get('batch_size', 1),
                'stop_sequences': config.get('stop_sequences'),
                'system_prompt': config.get('system_prompt'),
            }

            # Run lm-eval
            backend = LMEvalBackend()
            benchmark_name = f"{config.get('name', 'custom')}_exact_match"
            lm_results = backend.evaluate(target, benchmark_name, lm_config)

            # Map results back to original indices
            detailed_results = lm_results.get('detailed_results', [])
            results = []

            for i, row in enumerate(lm_eval_rows):
                if i < len(detailed_results):
                    detail = detailed_results[i]
                    score = 1.0 if detail.get('metrics', {}).get('exact_match', 0) else 0.0
                    success = bool(detail.get('metrics', {}).get('exact_match', 0))
                    output = detail.get('output', '')
                    raw_output = detail.get('raw_output', '')
                    reason = 'Exact match' if success else 'No match'
                else:
                    score = 0.0
                    success = False
                    output = ''
                    raw_output = ''
                    reason = 'No result'

                result = {
                    'original_idx': row['_original_idx'],
                    'eval_type': 'exact_match',
                    'instruction': row['instruction'],
                    'expected': row['answer'],
                    'output': output,
                    'raw_output': raw_output,
                    'score': score,
                    'success': success,
                    'reason': reason,
                }
                results.append(result)

            logger.info(f"Completed exact_match: {sum(r['success'] for r in results)}/{len(results)} correct")
            return results

        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _evaluate_judge_rows(
            self,
            rows: List[Dict[str, Any]],
            target: BaseTarget,
            config: Dict[str, Any],
            columns: Dict[str, str],
            judge_target: Optional[BaseTarget] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate judge rows using G-Eval."""
        if not rows:
            return []

        if not DEEPEVAL_AVAILABLE:
            raise ImportError("deepeval is required for judge evaluation")

        logger.info(f"Evaluating {len(rows)} judge rows with G-Eval")

        judge_model = None
        if judge_target:
            from surogate_eval.models.deepeval_wrapper import DeepEvalTargetWrapper
            judge_model = DeepEvalTargetWrapper(judge_target)
            logger.info(f"Using judge target: {judge_target.name}")

        default_criteria = config.get(
            'judge_criteria',
            'Evaluate if the response correctly answers the question based on the expected answer.'
        )

        results = []

        for row in rows:
            original_idx = row['_original_idx']
            instruction = self._get_column_value(row, columns, 'instruction', '')
            expected = self._get_column_value(row, columns, 'answer', '')
            row_criteria = self._get_column_value(row, columns, 'judge_criteria') or default_criteria

            # Get model output
            try:
                from surogate_eval.targets.base import TargetRequest
                request = TargetRequest(prompt=instruction)
                response = target.send_request(request)
                actual_output = response.content
            except Exception as e:
                logger.error(f"Inference error for row {original_idx}: {e}")
                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected,
                    'output': '',
                    'score': 0.0,
                    'success': False,
                    'reason': f'Inference error: {str(e)}',
                    'criteria': row_criteria,
                })
                continue

            # Run G-Eval
            try:
                metric = GEval(
                    name=f"judge_{original_idx}",
                    criteria=row_criteria,
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

                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected,
                    'output': actual_output,
                    'score': metric.score,
                    'success': metric.score >= 0.5,
                    'reason': getattr(metric, 'reason', None),
                    'criteria': row_criteria,
                })

                logger.debug(f"Row {original_idx} judge score: {metric.score:.3f}")

            except Exception as e:
                logger.error(f"G-Eval failed for row {original_idx}: {e}")
                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected,
                    'output': actual_output,
                    'score': 0.0,
                    'success': False,
                    'reason': f'Judge error: {str(e)}',
                    'criteria': row_criteria,
                })

        avg_score = sum(r['score'] for r in results) / len(results) if results else 0.0
        logger.info(f"Completed judge evaluation: avg score {avg_score:.3f}")

        return results

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
            - tokenizer: Tokenizer for lm-eval
            - max_tokens: Max generation tokens
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

        # Split by eval_type
        exact_match_rows, judge_rows = self._split_by_eval_type(dataset, columns)

        # Get judge target if configured
        judge_target = config.get('backend_params', {}).get('judge_target')

        # Evaluate each type
        exact_match_results = self._evaluate_exact_match_rows(
            exact_match_rows, target, config, columns
        )

        judge_results = self._evaluate_judge_rows(
            judge_rows, target, config, columns, judge_target
        )

        # Merge results
        all_results = exact_match_results + judge_results
        all_results.sort(key=lambda x: x['original_idx'])

        # Calculate metrics
        total = len(all_results)
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
            'detailed_results': all_results,
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