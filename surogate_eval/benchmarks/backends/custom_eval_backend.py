# surogate/eval/benchmarks/backends/custom_eval_backend.py
"""Custom evaluation backend supporting mixed exact_match and judge evaluation types."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

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

    Supports datasets with per-row eval_type:
    - exact_match: Uses LM-Eval for exact matching
    - judge: Uses G-Eval (LLM-as-judge) for scoring

    Dataset schema:
    - instruction (required): The prompt/question
    - answer (required): Expected answer
    - eval_type (required): 'judge' or 'exact_match'
    - system_prompt (optional): Per-row system prompt
    - judge_criteria (optional): Per-row criteria for judge eval
    - option_a, option_b, ... (optional): Multiple choice options
    """

    def __init__(self):
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library is required. Install with: pip install datasets"
            )
        self._judge_target: Optional[BaseTarget] = None
        self._judge_wrapper = None
        logger.debug("Initialized CustomEvalBackend")

    def _is_hf_dataset(self, source: str) -> bool:
        """Check if source is a HuggingFace dataset path."""
        if not source:
            return False
        if source.startswith(('.', '/', '~')) or Path(source).suffix:
            return False
        return '/' in source or (not os.path.exists(source))

    def _load_dataset(
            self,
            source: str,
            split: str = 'test',
            limit: Optional[int] = None
    ) -> Dataset:
        """Load dataset from HuggingFace, LakeFS, or local file."""
        logger.info(f"Loading dataset from: {source}")

        # Handle LakeFS URLs - use DatasetLoader
        if source.startswith('lakefs://'):
            from surogate_eval.datasets import DatasetLoader
            loader = DatasetLoader()
            # Download and get local path
            local_path = loader._download_from_lakefs(source)
            # Now load as local file
            source = local_path

        if self._is_hf_dataset(source):
            # HuggingFace dataset
            dataset = load_dataset(source, split=split, trust_remote_code=True)
            logger.info(f"Loaded HF dataset '{source}' split '{split}'")
        else:
            # Local file
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {source}")

            if source_path.suffix == '.jsonl':
                dataset = load_dataset('json', data_files=str(source_path), split='train')
            elif source_path.suffix == '.json':
                dataset = load_dataset('json', data_files=str(source_path), split='train')
            elif source_path.suffix == '.csv':
                dataset = load_dataset('csv', data_files=str(source_path), split='train')
            elif source_path.suffix == '.parquet':
                dataset = load_dataset('parquet', data_files=str(source_path), split='train')
            else:
                raise ValueError(f"Unsupported file format: {source_path.suffix}")

            logger.info(f"Loaded local dataset: {source_path}")

        if limit and limit < len(dataset):
            dataset = dataset.select(range(limit))
            logger.info(f"Limited dataset to {limit} rows")

        return dataset

    def _get_column_value(
            self,
            row: Dict[str, Any],
            columns: Dict[str, str],
            key: str,
            default: Any = None
    ) -> Any:
        """Get column value using column mapping."""
        column_name = columns.get(key, key)
        value = row.get(column_name)
        # Handle null/None values
        if value is None or (isinstance(value, str) and value.lower() == 'null'):
            return default
        return value

    def _has_options(
            self,
            row: Dict[str, Any],
            choices_columns: Optional[List[str]]
    ) -> bool:
        """Check if row has multiple choice options."""
        if not choices_columns:
            return False
        for col in choices_columns:
            value = row.get(col)
            if value is not None and value != '' and str(value).lower() != 'null':
                return True
        return False

    def _build_prompt(
            self,
            row: Dict[str, Any],
            columns: Dict[str, str],
            choices_columns: Optional[List[str]],
            choices_labels: Optional[List[str]],
            default_system_prompt: Optional[str] = None
    ) -> str:
        """Build the full prompt for a row."""
        instruction = self._get_column_value(row, columns, 'instruction', '')
        system_prompt = self._get_column_value(row, columns, 'system_prompt', default_system_prompt)

        # Build choices text if applicable
        choices_text = ""
        if self._has_options(row, choices_columns):
            labels = choices_labels or [chr(65 + i) for i in range(len(choices_columns))]
            choices_parts = []
            for label, col in zip(labels, choices_columns):
                value = row.get(col)
                if value is not None and value != '' and str(value).lower() != 'null':
                    choices_parts.append(f"{label}. {value}")
            if choices_parts:
                choices_text = "\n" + "\n".join(choices_parts)

        # Combine parts
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        prompt_parts.append(instruction + choices_text)

        return "\n\n".join(prompt_parts)

    def _split_by_eval_type(
            self,
            dataset: Dataset,
            columns: Dict[str, str]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset rows by eval_type."""
        exact_match_rows = []
        judge_rows = []

        for idx, row in enumerate(dataset):
            eval_type = self._get_column_value(row, columns, 'eval_type', 'exact_match')
            row_with_idx = dict(row)
            row_with_idx['_original_idx'] = idx

            if eval_type == 'judge':
                judge_rows.append(row_with_idx)
            else:
                exact_match_rows.append(row_with_idx)

        logger.info(f"Split dataset: {len(exact_match_rows)} exact_match, {len(judge_rows)} judge")
        return exact_match_rows, judge_rows

    def _evaluate_exact_match_rows(
            self,
            rows: List[Dict[str, Any]],
            target: BaseTarget,
            config: Dict[str, Any],
            columns: Dict[str, str],
            choices_columns: Optional[List[str]],
            choices_labels: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Evaluate exact_match rows using LM-Eval backend."""
        if not rows:
            return []

        logger.info(f"Evaluating {len(rows)} exact_match rows")

        # Split rows by whether they have MCQ options
        mcq_rows = []
        simple_rows = []

        for row in rows:
            has_choices = False
            if choices_columns:
                for col in choices_columns:
                    val = row.get(col)
                    if val is not None and str(val).strip() and str(val).lower() != 'null':
                        has_choices = True
                        break

            if has_choices:
                mcq_rows.append(row)
            else:
                simple_rows.append(row)

        logger.info(f"Split: {len(mcq_rows)} MCQ rows, {len(simple_rows)} simple rows")

        all_results = []

        # Evaluate MCQ rows
        if mcq_rows:
            mcq_results = self._run_exact_match_batch(
                rows=mcq_rows,
                target=target,
                config=config,
                columns=columns,
                choices_columns=choices_columns,
                choices_labels=choices_labels,
                is_mcq=True
            )
            all_results.extend(mcq_results)

        # Evaluate simple (non-MCQ) rows
        if simple_rows:
            simple_results = self._run_exact_match_batch(
                rows=simple_rows,
                target=target,
                config=config,
                columns=columns,
                choices_columns=None,  # No choices for simple rows
                choices_labels=None,
                is_mcq=False
            )
            all_results.extend(simple_results)

        # Sort by original index
        all_results.sort(key=lambda x: x['original_idx'])

        logger.info(
            f"Completed exact_match evaluation: {sum(r['success'] for r in all_results)}/{len(all_results)} correct")

        return all_results

    def _run_exact_match_batch(
            self,
            rows: List[Dict[str, Any]],
            target: BaseTarget,
            config: Dict[str, Any],
            columns: Dict[str, str],
            choices_columns: Optional[List[str]],
            choices_labels: Optional[List[str]],
            is_mcq: bool
    ) -> List[Dict[str, Any]]:
        """Run exact match evaluation on a batch of rows (either MCQ or simple)."""
        if not rows:
            return []

        logger.info(f"=== _run_exact_match_batch DEBUG ===")
        logger.info(f"is_mcq: {is_mcq}")
        logger.info(f"choices_columns passed: {choices_columns}")
        logger.info(f"choices_labels passed: {choices_labels}")
        logger.info(f"num rows: {len(rows)}")
        logger.info(f"first row sample: {rows[0] if rows else 'none'}")
        logger.info(f"====================================")

        from .lm_eval_backend import LMEvalBackend
        import tempfile
        import json

        # Prepare rows for lm-eval format
        lm_eval_rows = []
        for row in rows:
            lm_row = {
                'instruction': self._get_column_value(row, columns, 'instruction', ''),
                'answer': self._get_column_value(row, columns, 'answer', ''),
                '_original_idx': row['_original_idx']
            }

            # Add system prompt if present
            system_prompt = self._get_column_value(row, columns, 'system_prompt')
            if system_prompt:
                lm_row['system_prompt'] = system_prompt

            # Add options if MCQ
            if is_mcq and choices_columns:
                for col in choices_columns:
                    value = row.get(col)
                    if value is not None and str(value).lower() != 'null':
                        lm_row[col] = value

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
                'task_type': 'generation',
                'columns': {
                    'question': 'instruction',
                    'answer': 'answer',
                },
                'split': 'test',
                'num_fewshot': 0,
                'max_tokens': config.get('max_tokens', 256),
                'tokenizer': config.get('tokenizer'),
                'batch_size': config.get('batch_size', 1),
                'stop_sequences': config.get('stop_sequences'),
                'system_prompt': config.get('system_prompt'),
            }

            # Add choices configuration only for MCQ batch
            if is_mcq and choices_columns:
                lm_config['choices_columns'] = choices_columns
                lm_config['choices_labels'] = choices_labels or [chr(65 + i) for i in range(len(choices_columns))]

            # Run lm-eval
            backend = LMEvalBackend()
            batch_type = 'mcq' if is_mcq else 'simple'
            benchmark_name = f"{config.get('name', 'custom')}_{batch_type}_exact_match"
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
                    'format': 'mcq' if is_mcq else 'simple',
                }
                results.append(result)

            return results

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _split_by_format(
            self,
            rows: List[Dict[str, Any]],
            choices_columns: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split rows into MCQ and non-MCQ."""
        mcq_rows = []
        simple_rows = []

        for row in rows:
            has_choices = False
            if choices_columns:
                for col in choices_columns:
                    val = row.get(col)
                    if val is not None and str(val).strip():
                        has_choices = True
                        break

            if has_choices:
                mcq_rows.append(row)
            else:
                simple_rows.append(row)

        return mcq_rows, simple_rows

    def _any_row_has_options(
            self,
            rows: List[Dict[str, Any]],
            choices_columns: Optional[List[str]]
    ) -> bool:
        """Check if any row has options."""
        if not choices_columns:
            return False
        for row in rows:
            if self._has_options(row, choices_columns):
                return True
        return False

    def _evaluate_judge_rows(
            self,
            rows: List[Dict[str, Any]],
            target: BaseTarget,
            config: Dict[str, Any],  # Changed from Dict[str, str]
            columns: Dict[str, str],
            choices_columns: Optional[List[str]],
            choices_labels: Optional[List[str]],
            judge_target: Optional[BaseTarget] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate judge rows using G-Eval."""
        if not rows:
            return []

        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "deepeval is required for judge evaluation. Install with: pip install deepeval"
            )

        logger.info(f"Evaluating {len(rows)} judge rows")

        # Get judge model wrapper
        judge_model = None
        if judge_target:
            from surogate_eval.models.deepeval_wrapper import DeepEvalTargetWrapper
            judge_model = DeepEvalTargetWrapper(judge_target)
            logger.info(f"Using judge target: {judge_target.name}")

        # Default criteria - MUST have a valid fallback
        default_criteria = config.get(
            'judge_criteria',
            'Evaluate if the response correctly answers the question based on the expected answer.'
        )

        results = []

        for row in rows:
            original_idx = row['_original_idx']
            instruction = self._get_column_value(row, columns, 'instruction', '')
            expected_answer = self._get_column_value(row, columns, 'answer', '')
            system_prompt = self._get_column_value(row, columns, 'system_prompt')

            # Get row criteria, fall back to default if None/empty
            row_criteria = self._get_column_value(row, columns, 'judge_criteria')
            if not row_criteria:
                row_criteria = default_criteria

            # Build prompt
            prompt = self._build_prompt(
                row, columns, choices_columns, choices_labels, system_prompt
            )

            # Get model output
            try:
                from surogate_eval.targets.base import TargetRequest
                request = TargetRequest(prompt=prompt)
                response = target.send_request(request)
                actual_output = response.content
            except Exception as e:
                logger.error(f"Failed to get output for row {original_idx}: {e}")
                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected_answer,
                    'output': '',
                    'score': 0.0,
                    'success': False,
                    'reason': f'Inference error: {str(e)}',
                    'criteria': row_criteria,
                })
                continue

            # Run G-Eval
            try:
                # Create G-Eval metric with row-specific criteria
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

                # Create test case
                test_case = LLMTestCase(
                    input=instruction,
                    actual_output=actual_output,
                    expected_output=expected_answer,
                )

                # Evaluate
                metric.measure(test_case, _show_indicator=False)

                score = metric.score
                reason = metric.reason if hasattr(metric, 'reason') else None

                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected_answer,
                    'output': actual_output,
                    'score': score,
                    'success': score >= 0.5,
                    'reason': reason,
                    'criteria': row_criteria,
                })

                logger.debug(f"Row {original_idx} judge score: {score:.3f}")

            except Exception as e:
                logger.error(f"G-Eval failed for row {original_idx}: {e}")
                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected_answer,
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
        Evaluate target using custom mixed evaluation.

        Args:
            target: Target model to evaluate
            benchmark_name: Name of the benchmark
            config: Configuration dict containing:
                - source: HF dataset path or local file
                - columns: Column name mappings
                - choices_columns: Optional list of choice columns
                - choices_labels: Optional list of choice labels
                - split: Dataset split (for HF datasets)
                - limit: Max rows to evaluate
                - judge_model: Judge model config for judge rows
                - judge_criteria: Default criteria for judge rows
                - tokenizer: Tokenizer for exact_match rows
                - max_tokens: Max tokens for generation

        Returns:
            Evaluation results dict
        """
        logger.info(f"Running custom evaluation: {benchmark_name}")

        source = config.get('source')
        if not source:
            raise ValueError("'source' is required for custom_eval backend")

        columns = config.get('columns', {})
        choices_columns = config.get('choices_columns')
        choices_labels = config.get('choices_labels')
        split = config.get('split', 'test')
        limit = config.get('limit')

        # Load dataset
        dataset = self._load_dataset(source, split, limit)

        # Validate required columns
        required_columns = ['instruction', 'answer', 'eval_type']
        for req_col in required_columns:
            mapped_col = columns.get(req_col, req_col)
            if mapped_col not in dataset.column_names:
                raise ValueError(f"Required column '{mapped_col}' (mapped from '{req_col}') not found in dataset")

        # Split by eval_type
        exact_match_rows, judge_rows = self._split_by_eval_type(dataset, columns)

        # Get judge target if configured
        judge_target = config.get('backend_params', {}).get('judge_target')

        # Evaluate each type
        exact_match_results = self._evaluate_exact_match_rows(
            exact_match_rows, target, config, columns, choices_columns, choices_labels
        )

        judge_results = self._evaluate_judge_rows(
            judge_rows, target, config, columns, choices_columns, choices_labels, judge_target
        )

        # Merge results
        all_results = exact_match_results + judge_results
        all_results.sort(key=lambda x: x['original_idx'])

        # Calculate metrics
        total = len(all_results)
        exact_match_total = len(exact_match_results)
        judge_total = len(judge_results)

        exact_match_correct = sum(1 for r in exact_match_results if r['success'])
        judge_avg_score = sum(r['score'] for r in judge_results) / judge_total if judge_total else 0.0

        overall_score = (
                                (
                                    exact_match_correct / exact_match_total if exact_match_total else 0.0) * exact_match_total +
                                judge_avg_score * judge_total
                        ) / total if total else 0.0

        return {
            'overall_score': overall_score,
            'num_samples': total,
            'task_results': {
                'exact_match': {
                    'total': exact_match_total,
                    'correct': exact_match_correct,
                    'accuracy': exact_match_correct / exact_match_total if exact_match_total else 0.0,
                },
                'judge': {
                    'total': judge_total,
                    'avg_score': judge_avg_score,
                    'success_rate': sum(1 for r in judge_results if r['success']) / judge_total if judge_total else 0.0,
                },
            },
            'detailed_results': all_results,
            'metadata': {
                'backend': 'custom_eval',
                'benchmark': benchmark_name,
                'source': source,
                'split': split,
                'num_exact_match': exact_match_total,
                'num_judge': judge_total,
                'status': 'completed',
            }
        }


# Alias for consistency
CustomEvalBackendWrapper = CustomEvalBackend