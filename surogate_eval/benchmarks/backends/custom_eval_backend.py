# surogate/eval/benchmarks/backends/custom_eval_backend.py
"""Custom evaluation backend supporting mixed exact_match and judge evaluation types."""

import os
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from surogate_eval.benchmarks.matching import Matcher, build_matcher, clean_formatting
from surogate_eval.targets import BaseTarget
from surogate_eval.utils.logger import get_logger
from surogate_eval.utils.text import blank_as_none
from surogate_eval import runners

logger = get_logger()

#: Seconds between row-level progress writes. Ops polls at 5s, so a tighter
#: cadence buys nothing and a per-row write costs one file write per sample on
#: a 1319-row benchmark.
_PROGRESS_INTERVAL_SECONDS = 2.0


def _row_counts(results: list) -> tuple:
    """(scored, errored, passed, score_sum) over the rows measured so far.

    Derived from the results list both loops already build, rather than
    counters threaded through every append site: there are five of those
    between the two loops and each one is a place to forget.
    """
    scored = sum(1 for r in results if r.get("status") == "scored")
    errored = sum(1 for r in results if r.get("status") == "errored")
    passed = sum(1 for r in results if r.get("success"))
    score_sum = sum(
        r["score"] for r in results if r.get("score") is not None
    )
    return scored, errored, passed, float(score_sum)

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


#: Files a dataset repo carries alongside its data. Every one of these has a
#: data-ish extension, so a "first file with a known suffix" rule picks the
#: wrong one -- `dataset_dict.json` sorts before `train/dataset.parquet` and
#: holds `{"splits": ["train"]}`, which loads as a dataset with no columns and
#: reports the mapped column missing.
_METADATA_FILENAMES = frozenset({
    'dataset_dict.json',
    'dataset_info.json',
    'state.json',
})

#: Best first: parquet is the canonical form a Hub dataset is published in,
#: and the others are what an unconverted upload might still be sitting as.
_DATA_EXTENSIONS = ('.parquet', '.jsonl', '.json', '.csv')


def _pick_dataset_file(paths: List[str], split: str = '') -> Optional[str]:
    """Choose the data file to download from a repo listing.

    Three rules, in order:

    1. Metadata files are never data, whatever they are called.
    2. A file under ``{split}/`` wins, because a repo holding several splits
       otherwise loads whichever sorts first and scores the wrong rows
       silently.
    3. Failing that, prefer by extension rather than by listing order.

    Returns ``None`` when nothing qualifies, so the caller can name what it
    did find.
    """
    candidates = [
        p for p in paths
        if p and Path(p).name not in _METADATA_FILENAMES
        and Path(p).suffix.lower() in _DATA_EXTENSIONS
    ]
    if not candidates:
        return None

    def rank(path: str) -> tuple:
        in_split = bool(split) and Path(path).parent.name == split
        try:
            ext_rank = _DATA_EXTENSIONS.index(Path(path).suffix.lower())
        except ValueError:  # pragma: no cover - filtered above
            ext_rank = len(_DATA_EXTENSIONS)
        # Sorted ascending, so negate the flag we want to win.
        return (not in_split, ext_rank, path)

    return sorted(candidates, key=rank)[0]


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
        """Load dataset from HuggingFace, Hub (surogate-hub), LakeFS, or local file."""
        logger.info(f"Loading dataset from: {source}")

        # Handle Hub URLs — hub://{user}/{repo}/{ref}
        # Downloads all dataset files from the Hub REST API.
        if source.startswith('hub://'):
            local_path = self._download_from_hub(source, split)
            logger.info(f"Downloaded Hub dataset to: {local_path}")
            source = local_path

        # Handle LakeFS URLs
        elif source.startswith('lakefs://'):
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

    def _download_from_hub(self, hub_uri: str, split: str = '') -> str:
        """Download a dataset from the surogate-hub REST API.

        URI format: hub://{user}/{repo}/{ref}
        Env vars: HUBCTL_SERVER_ENDPOINT_URL, HUBCTL_CREDENTIALS_ACCESS_KEY_ID,
                  HUBCTL_CREDENTIALS_SECRET_ACCESS_KEY
        """
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        parts = hub_uri.replace('hub://', '').split('/', 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid Hub URI: {hub_uri}. Expected hub://user/repo/ref")

        user, repo, ref = parts[0], parts[1], parts[2]

        endpoint = os.environ.get('HUBCTL_SERVER_ENDPOINT_URL')
        access_key = os.environ.get('HUBCTL_CREDENTIALS_ACCESS_KEY_ID')
        secret_key = os.environ.get('HUBCTL_CREDENTIALS_SECRET_ACCESS_KEY')

        if not all([endpoint, access_key, secret_key]):
            raise ValueError(
                "Hub credentials not configured. Set environment variables: "
                "HUBCTL_SERVER_ENDPOINT_URL, HUBCTL_CREDENTIALS_ACCESS_KEY_ID, "
                "HUBCTL_CREDENTIALS_SECRET_ACCESS_KEY"
            )

        base_url = endpoint.rstrip('/')
        if '/api/v1' not in base_url:
            base_url += '/api/v1'

        auth = (access_key, secret_key)

        # List objects to find dataset files
        list_url = f"{base_url}/repositories/{user}/{repo}/refs/{ref}/objects/ls"
        resp = requests.get(list_url, auth=auth, verify=False, timeout=30)
        resp.raise_for_status()
        objects = resp.json().get('results', [])

        dataset_file = _pick_dataset_file(
            [o.get('path', '') for o in objects], split,
        )
        if not dataset_file:
            raise FileNotFoundError(
                f"No dataset file found in hub://{user}/{repo}/{ref}. "
                f"Found: {[o.get('path') for o in objects]}"
            )

        logger.info(f"Downloading from Hub: {user}/{repo}@{ref} -> {dataset_file}")

        # Download the file
        get_url = f"{base_url}/repositories/{user}/{repo}/refs/{ref}/objects"
        resp = requests.get(
            get_url, auth=auth, verify=False, timeout=120,
            params={'path': dataset_file},
        )
        resp.raise_for_status()

        suffix = Path(dataset_file).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(resp.content)
        temp_file.close()

        logger.info(f"Downloaded {len(resp.content)} bytes to {temp_file.name}")
        return temp_file.name

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
            columns: Dict[str, str],
            config: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset rows by eval_type based on config."""
        exact_match_rows = []
        judge_rows = []

        # Get eval_type from config (sent by frontend)
        mode = config.get('eval_type', 'exact_match')

        logger.info(f"Evaluation mode: {mode}")

        for idx, row in enumerate(dataset):
            row_dict = dict(row)
            row_dict['_original_idx'] = idx

            if mode == 'hybrid':
                # Per-row eval type from column
                row_eval_type = self._get_column_value(row, columns, 'eval_type', 'exact_match')
                if row_eval_type == 'judge':
                    judge_rows.append(row_dict)
                else:
                    exact_match_rows.append(row_dict)
            elif mode == 'judge':
                judge_rows.append(row_dict)
            else:  # exact_match
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
        """Evaluate exact_match rows.

        Uses lm-eval when a tokenizer is available (local models), otherwise
        falls back to direct inference + string comparison (API-only models).
        """
        if not rows:
            return []

        # Built here, above the path choice, for two reasons. A bad pattern or
        # an unknown mode would fail every row, so it is a config error rather
        # than a per-row failure. And building it inside the lm-eval path put
        # it under the blanket `except` below, which reported a matcher typo as
        # `lm-eval exact_match failed, falling back to...` before the direct
        # path rebuilt it and raised the same error - pointing the first
        # diagnostic at lm-eval, which was not what was broken.
        matcher = build_matcher(config.get('matcher'))

        # Use lm-eval only when a tokenizer is explicitly set (local models).
        # For API-only models, direct inference is more reliable (lm-eval
        # uses /completions which many providers don't support).
        tokenizer = config.get('tokenizer') or target.config.get('tokenizer')
        if tokenizer:
            try:
                return self._evaluate_exact_match_lm_eval(
                    rows, target, config, columns, tokenizer, matcher
                )
            except Exception as e:
                logger.warning(f"lm-eval exact_match failed, falling back to direct inference: {e}")

        return self._evaluate_exact_match_direct(rows, target, config, columns, matcher)

    def _evaluate_exact_match_lm_eval(
            self,
            rows: List[Dict[str, Any]],
            target: BaseTarget,
            config: Dict[str, Any],
            columns: Dict[str, str],
            tokenizer: str,
            matcher: Matcher,
    ) -> List[Dict[str, Any]]:
        """Evaluate exact_match rows using LM-Eval backend."""
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
                'tokenizer': tokenizer,
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
            # `LMEvalBackend.evaluate` does not raise when `simple_evaluate`
            # fails: it returns a payload with no `detailed_results` and the
            # real cause in `metadata['error']` (lm_eval_backend.py:350-366).
            # So the most likely real failure - an unreachable target, a bad
            # tokenizer - arrives here as "zero rows returned", every row
            # takes the unmeasured branch below, and the actual diagnostic is
            # dropped unless it is carried into the reason. Same principle as
            # building the matcher above the path fork: an error should name
            # what actually broke.
            lm_error = (lm_results.get('metadata') or {}).get('error')
            no_result_reason = (
                f'lm-eval returned no result for this row: {lm_error}'
                if lm_error else 'lm-eval returned no result for this row'
            )
            results = []

            for i, row in enumerate(lm_eval_rows):
                if i >= len(detailed_results):
                    # lm-eval returned fewer rows than we sent. This row was
                    # never scored, which is not the same as scoring it wrong.
                    results.append({
                        'original_idx': row['_original_idx'],
                        'eval_type': 'exact_match',
                        'instruction': row['instruction'],
                        'expected': row['answer'],
                        'output': '',
                        'raw_output': '',
                        'status': 'errored',
                        'score': None,
                        'success': False,
                        'reason': no_result_reason,
                    })
                    continue

                detail = detailed_results[i]
                # lm-eval generated; the comparison is ours. Its own
                # exact_match metric is deliberately not read, and neither is
                # its `output`, which is its own extraction heuristic - a
                # third extractor beside the matcher.
                raw_output = detail.get('raw_output', '') or ''
                status = 'scored'
                reason = None
                try:
                    # Pairing detail[i] with lm_eval_rows[i] is now score-
                    # bearing, not just cosmetic: before this change the score
                    # came from lm-eval's own per-sample metric, so a reordered
                    # sample list only mislabeled the displayed columns. Now
                    # one row's generation would be compared against another
                    # row's answer key, silently. lm-eval hands each sample its
                    # own `target` back, so a disagreement is detectable - and
                    # a row we cannot pair confidently is unmeasured, not wrong.
                    detail_expected = detail.get('expected')
                    if (
                        detail_expected is not None
                        and str(detail_expected).strip() != str(row['answer']).strip()
                    ):
                        raise ValueError(
                            f"lm-eval returned sample {i} with target "
                            f"{str(detail_expected)[:80]!r}, expected "
                            f"{str(row['answer'])[:80]!r}; results are not in the "
                            f"order they were sent"
                        )
                    success, output = matcher.compare(raw_output, row['answer'])
                    score = 1.0 if success else 0.0
                    # Named after the configured mode, not hardcoded as
                    # "exact match": the branch's whole point is that the
                    # rule need not be exact match, and the direct path
                    # reports success the same way (see there for why).
                    reason = f'{matcher.mode} match' if success else 'No match'
                except Exception as e:
                    status = 'errored'
                    score = None
                    success = False
                    output = ''
                    reason = f'Comparison error: {e}'

                result = {
                    'original_idx': row['_original_idx'],
                    'eval_type': 'exact_match',
                    'instruction': row['instruction'],
                    'expected': row['answer'],
                    'output': output,
                    'raw_output': raw_output,
                    'status': status,
                    'score': score,
                    'success': success,
                    'reason': reason,
                }
                results.append(result)

            errored_n = sum(1 for r in results if r['status'] == 'errored')
            scored_n = len(results) - errored_n
            logger.info(
                f"Completed exact_match (lm-eval): "
                f"{sum(r['success'] for r in results)}/{scored_n} correct "
                f"({errored_n} not measured)"
            )
            return results

        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _evaluate_exact_match_direct(
            self,
            rows: List[Dict[str, Any]],
            target: BaseTarget,
            config: Dict[str, Any],
            columns: Dict[str, str],
            matcher: Matcher,
    ) -> List[Dict[str, Any]]:
        """Evaluate exact_match rows via direct inference + string comparison."""
        logger.info(f"Evaluating {len(rows)} exact_match rows (direct inference)")

        from surogate_eval.targets.base import TargetRequest

        system_prompt = config.get('system_prompt')
        results = []
        last_report = 0.0

        for row in rows:
            original_idx = row['_original_idx']
            instruction = self._get_column_value(row, columns, 'instruction', '')
            expected = self._get_column_value(row, columns, 'answer', '')

            # A request that failed is a failure to measure, not a target
            # that answered wrongly. Same rule as _evaluate_toxicity_rows:
            # a response carrying an error is errored; an empty completion
            # with no error is a real (bad) answer and is still scored.
            request_error = None
            raw_output = ''
            try:
                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": instruction},
                    ]
                    request = TargetRequest(messages=messages)
                else:
                    request = TargetRequest(prompt=instruction)
                response = target.send_request(request)
                if response.error:
                    request_error = response.error
                else:
                    raw_output = response.content or ''
            except Exception as e:
                request_error = str(e)

            row_error = (
                f'Inference error: {request_error}' if request_error is not None else None
            )
            normalized_output = ''
            success = False

            if row_error is None:
                # Comparing can fail on its own, and a failure here is still
                # a failure to measure, not a wrong answer. ``matcher.compare``
                # coerces a non-string ``expected`` (a numeric ``answer``
                # column is inferred as int64) with ``str()``, so that alone
                # no longer raises - but a ``regex`` pattern that
                # catastrophically backtracks on one row's output still
                # raises ``MatchTimeout``. Outside the protected region that
                # would escape the row loop and the function, and
                # runners.py's top-level handler would turn the whole
                # benchmark into a status-only failure - discarding every
                # row already measured, for a perfectly healthy target. Same
                # rule as the request above, and the same shape as
                # _evaluate_judge_rows: one row we could not compare is one
                # errored row.
                try:
                    success, normalized_output = matcher.compare(raw_output, expected)
                except Exception as e:
                    row_error = f'Comparison error: {e}'

            if row_error is not None:
                logger.error(f"Row {original_idx} could not be measured: {row_error}")
                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'exact_match',
                    'instruction': instruction,
                    'expected': expected,
                    'output': '',
                    'raw_output': raw_output,
                    'status': 'errored',
                    'score': None,
                    'success': False,
                    'reason': row_error,
                })
                continue

            results.append({
                'original_idx': original_idx,
                'eval_type': 'exact_match',
                'instruction': instruction,
                'expected': expected,
                'output': normalized_output,
                'raw_output': raw_output,
                'status': 'scored',
                'score': 1.0 if success else 0.0,
                'success': success,
                # Named after the configured mode, not hardcoded as "exact
                # match": that word is only true under `mode: exact`, and
                # was misleading under `contains` (the default) and `regex`.
                # Matches the lm-eval path's wording so the two paths agree
                # in user-visible report text, same as they agree on the
                # score itself.
                'reason': f'{matcher.mode} match' if success else 'No match',
            })

            now = time.monotonic()
            if now - last_report >= _PROGRESS_INTERVAL_SECONDS:
                last_report = now
                runners.report_rows(
                    len(results), len(rows), *_row_counts(results),
                )

        runners.report_rows(len(results), len(rows), *_row_counts(results))

        errored_n = sum(1 for r in results if r['status'] == 'errored')
        scored_n = len(results) - errored_n
        logger.info(
            f"Completed exact_match (direct): {sum(r['success'] for r in results)}/{scored_n} correct "
            f"({errored_n} not measured)"
        )
        return results

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

        from surogate_eval.models.deepeval_wrapper import DeepEvalTargetWrapper
        if judge_target:
            judge_model = DeepEvalTargetWrapper(judge_target)
            logger.info(f"Using judge target: {judge_target.name}")
        else:
            # Fall back to the eval target itself as judge
            judge_model = DeepEvalTargetWrapper(target)
            logger.info(f"No judge configured, using target as judge: {target.name}")

        # ``blank_as_none`` and ``or``, not ``get``'s default: a criteria that
        # is empty OR only whitespace is as fatal to GEval as an absent one,
        # and a form posting a blank field produces all three. ``generic.py``
        # drops the unset case for every key; blankness is per-site because
        # only text fields have the notion.
        default_criteria = blank_as_none(config.get('judge_criteria')) or (
            'Evaluate if the response correctly answers the question '
            'based on the expected answer.'
        )

        prompt_template = config.get('prompt_template')

        results = []
        last_report = 0.0

        for row in rows:
            original_idx = row['_original_idx']
            instruction = self._get_column_value(row, columns, 'instruction', '')
            expected = self._get_column_value(row, columns, 'answer', '')
            # The more exposed of the two: this one comes from a dataset cell,
            # where a lone space is easy to produce by accident.
            row_criteria = blank_as_none(
                self._get_column_value(row, columns, 'judge_criteria')
            ) or default_criteria

            # Apply prompt template if provided (e.g. wrap raw text in a
            # classification prompt instead of sending it verbatim).
            prompt = instruction
            if prompt_template:
                prompt = prompt_template.replace('{instruction}', instruction)
                prompt = prompt.replace('{expected}', expected)

            # Get model output.
            #
            # A request that failed is a failure to measure, not a target
            # that answered wrongly. Same rule as _evaluate_toxicity_rows:
            # a response carrying an error is errored; an empty completion
            # with no error is a real (bad) answer and is still judged.
            request_error = None
            raw_output = ''
            normalized_output = ''
            try:
                from surogate_eval.targets.base import TargetRequest
                system_prompt = config.get('system_prompt')
                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    request = TargetRequest(messages=messages)
                else:
                    request = TargetRequest(prompt=prompt)
                response = target.send_request(request)
                if response.error:
                    request_error = response.error
                else:
                    raw_output = response.content or ''
                    # Formatting cleanup only. The judge path is not part of
                    # the matcher wiring, but it did call `_normalize_output`,
                    # so retiring that method changed what the judge sees: the
                    # email/percentage/date/yes-no heuristics used to hand
                    # G-Eval a *fragment* of the answer when the expected value
                    # had one of those shapes, and now it gets the whole
                    # cleaned response. That is the better input - a judge
                    # scoring an answer should see the answer, not a regex's
                    # guess at it - but it is a real change, not a no-op.
                    normalized_output = clean_formatting(raw_output)
                    logger.debug(f"Raw: {raw_output[:100]}... -> Normalized: {normalized_output[:100]}...")
            except Exception as e:
                request_error = str(e)

            if request_error is not None:
                logger.error(f"Inference error for row {original_idx}: {request_error}")
                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected,
                    'output': '',
                    'raw_output': '',
                    'status': 'errored',
                    'score': None,
                    'success': False,
                    'reason': f'Inference error: {request_error}',
                    'criteria': row_criteria,
                })
                continue

            # Run G-Eval with normalized output
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
                    actual_output=normalized_output,  # Use normalized
                    expected_output=expected,
                )

                metric.measure(test_case, _show_indicator=False)

                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected,
                    'output': normalized_output,  # Store normalized
                    'raw_output': raw_output,  # Store raw for reference
                    'status': 'scored',
                    'score': metric.score,
                    'success': metric.score >= 0.5,
                    'reason': getattr(metric, 'reason', None),
                    'criteria': row_criteria,
                })

                logger.debug(f"Row {original_idx} judge score: {metric.score:.3f}")

            except Exception as e:
                # A judge that breaks is a failure to measure, not a target
                # that answered badly. Recorded as a scored 0.0 it left
                # ``errored_n`` at zero while dragging ``avg_score``,
                # ``accuracy`` and ``overall_score`` down with fake zeroes -
                # the same confusion ``_evaluate_toxicity_rows`` handles as
                # errored. The target's answer is kept: what failed is the
                # judgement of it.
                logger.error(f"G-Eval failed for row {original_idx}: {e}")
                results.append({
                    'original_idx': original_idx,
                    'eval_type': 'judge',
                    'instruction': instruction,
                    'expected': expected,
                    'output': normalized_output,
                    'raw_output': raw_output,
                    'status': 'errored',
                    'score': None,
                    'success': False,
                    'reason': f'Judge error: {str(e)}',
                    'criteria': row_criteria,
                })

            now = time.monotonic()
            if now - last_report >= _PROGRESS_INTERVAL_SECONDS:
                last_report = now
                runners.report_rows(
                    len(results), len(rows), *_row_counts(results),
                )

        runners.report_rows(len(results), len(rows), *_row_counts(results))

        errored_n = sum(1 for r in results if r['status'] == 'errored')
        scored_n = len(results) - errored_n
        avg_score = (
            sum(r['score'] for r in results if r['status'] != 'errored') / scored_n
        ) if scored_n else 0.0
        logger.info(
            f"Completed judge evaluation: avg score {avg_score:.3f} over {scored_n} scored "
            f"({errored_n} not measured)"
        )

        return results

    def _evaluate_toxicity_rows(
            self,
            rows: List[Dict[str, Any]],
            target: BaseTarget,
            config: Dict[str, Any],
            columns: Dict[str, str],
            judge_target: Optional[BaseTarget] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate rows using DeepEval's ToxicityMetric.

        Sends each prompt to the target, then scores the response for
        toxicity.  Score 1.0 = safe, 0.0 = toxic.
        """
        from deepeval.metrics import ToxicityMetric as DeepEvalToxicity
        from deepeval.test_case import LLMTestCase
        from surogate_eval.models import DeepEvalTargetWrapper

        # Use judge target for the toxicity evaluator if available,
        # otherwise fall back to the target being tested.
        evaluator_model = None
        if judge_target:
            evaluator_model = DeepEvalTargetWrapper(judge_target)

        metric = DeepEvalToxicity(
            threshold=0.5,
            model=evaluator_model,
            include_reason=True,
            async_mode=False,
        )

        prompt_template = config.get('prompt_template')
        results = []

        for idx, row in enumerate(rows):
            instruction = self._get_column_value(row, columns, 'instruction', '')
            expected = self._get_column_value(row, columns, 'answer', '')

            prompt = instruction
            if prompt_template:
                prompt = prompt_template.replace('{instruction}', instruction)
                prompt = prompt.replace('{expected}', expected)

            # Get model response.
            #
            # A request that failed is a failure to measure, not a target
            # that said something inoffensive. Both halves of this used to
            # be swallowed into output='' and handed to the judge, which
            # duly found the empty string non-toxic and recorded the row as
            # scored and successful for a target that never answered.
            #
            # Same rule as LLMJudgeMetric._no_output_result: a response
            # carrying an error is errored; an empty completion with no
            # error is a real (bad) answer and is still judged.
            request_error = None
            output = ''
            try:
                from surogate_eval.targets.base import TargetRequest
                response = target.send_request(TargetRequest(prompt=prompt))
                if response.error:
                    request_error = response.error
                else:
                    output = response.content or ''
            except Exception as e:
                request_error = str(e)

            errored = False
            is_safe = False

            if request_error is not None:
                logger.error(f"Inference error for toxicity row {idx}: {request_error}")
                errored = True
                reason = f"Target request failed: {request_error}"
            else:
                # Score with DeepEval ToxicityMetric
                #
                # A judge that breaks is a failure to measure, not a verdict
                # of safe. This used to be `except Exception: is_safe = True`,
                # so a dead judge produced score 1.0 and success True for
                # every row.
                try:
                    test_case = LLMTestCase(input=prompt, actual_output=output)
                    metric.measure(test_case)
                    # ToxicityMetric: score is toxicity level (0 = safe, 1 = toxic)
                    # We invert: success = not toxic (score < threshold)
                    if metric.score is None:
                        raise ValueError("judge returned no toxicity score")
                    is_safe = metric.score < metric.threshold
                    reason = metric.reason or ''
                except Exception as e:
                    logger.error(f"Toxicity metric failed for row {idx}: {e}")
                    errored = True
                    reason = f"Metric error: {e}"

            results.append({
                'original_idx': idx,
                'eval_type': 'toxicity',
                'instruction': instruction,
                'expected': expected,
                'output': output,
                'status': 'errored' if errored else 'scored',
                'score': None if errored else (1.0 if is_safe else 0.0),
                'success': False if errored else is_safe,
                'reason': reason,
            })

            if (idx + 1) % 5 == 0 or idx == len(rows) - 1:
                safe_so_far = sum(1 for r in results if r['success'])
                errored_so_far = sum(1 for r in results if r['status'] == 'errored')
                logger.info(
                    f"Toxicity eval: {idx + 1}/{len(rows)} — {safe_so_far} safe, "
                    f"{errored_so_far} not measured"
                )

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

        mode = config.get('eval_type', 'exact_match')

        # Toxicity mode: use DeepEval's ToxicityMetric
        if mode == 'toxicity':
            judge_target = config.get('backend_params', {}).get('judge_target')
            all_results = self._evaluate_toxicity_rows(
                list(dataset), target, config, columns, judge_target,
            )
            total = len(all_results)
            errored_n = sum(1 for r in all_results if r.get('status') == 'errored')
            scored_n = total - errored_n
            safe_count = sum(1 for r in all_results if r['success'])
            # Rate over what was actually measured. The errored rows are
            # reported separately so the run outcome can see them.
            overall_score = safe_count / scored_n if scored_n else 0.0
            # No rate when nothing was measured, same rule as the exact-match
            # and judge tasks below: `safety_rate: 0.0` is indistinguishable
            # from a model that was toxic on every row, and the report's
            # "not measured" branch keys on the metric being absent.
            toxicity_task = {
                'total': total,
                'safe': safe_count,
                'toxic': scored_n - safe_count,
                'scored_n': scored_n,
                'errored_n': errored_n,
            }
            if scored_n:
                toxicity_task['safety_rate'] = overall_score
            return {
                'overall_score': overall_score,
                'num_samples': total,
                'task_results': {'toxicity': toxicity_task},
                'detailed_results': all_results,
                'metadata': {
                    'backend': 'custom_eval',
                    'benchmark': benchmark_name,
                    'source': source,
                    'split': split,
                    'eval_type': 'toxicity',
                    'status': 'completed',
                },
            }

        # Split by eval_type
        exact_match_rows, judge_rows = self._split_by_eval_type(dataset, columns, config)

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

        # Calculate metrics. Rates are over what was actually measured - an
        # errored row is reported separately (scored_n/errored_n) so the run
        # outcome can see it, instead of being averaged in as a fake zero.
        total = len(all_results)
        em_total = len(exact_match_results)
        judge_total = len(judge_results)

        em_errored = sum(1 for r in exact_match_results if r.get('status') == 'errored')
        em_scored = em_total - em_errored
        em_correct = sum(1 for r in exact_match_results if r['success'])

        judge_errored = sum(1 for r in judge_results if r.get('status') == 'errored')
        judge_scored = judge_total - judge_errored
        judge_scored_rows = [r for r in judge_results if r.get('status') != 'errored']
        judge_avg = sum(r['score'] for r in judge_scored_rows) / judge_scored if judge_scored else 0.0

        scored_total = em_scored + judge_scored
        overall_score = 0.0
        if scored_total > 0:
            overall_score = (
                (em_correct / em_scored if em_scored else 0.0) * em_scored +
                judge_avg * judge_scored
            ) / scored_total

        # A task nobody could measure reports no rate at all, rather than a
        # rate of zero. `accuracy: 0.0` is indistinguishable from a model that
        # answered everything wrong, and the report's "not measured" branch
        # keys on the metric being ABSENT, so filling it in with a zero is
        # what stopped that branch ever firing. This branch made the case
        # reachable on a healthy target: a blank answer column, a match
        # timeout and a short lm-eval return all error every row.
        exact_match_task = {
            'total': em_total,
            'correct': em_correct,
            'scored_n': em_scored,
            'errored_n': em_errored,
        }
        if em_scored:
            exact_match_task['accuracy'] = em_correct / em_scored

        judge_task = {
            'total': judge_total,
            'scored_n': judge_scored,
            'errored_n': judge_errored,
        }
        if judge_scored:
            judge_task['avg_score'] = judge_avg
            judge_task['success_rate'] = (
                sum(1 for r in judge_scored_rows if r['success']) / judge_scored
            )

        # A task with no rows at all is not a task. Emitting both keys
        # unconditionally meant every exact-match-only benchmark reported a
        # judge task over zero rows, which the change above would otherwise
        # have promoted from a misleading 0.000 to a misleading "not measured".
        task_results = {}
        if em_total:
            task_results['exact_match'] = exact_match_task
        if judge_total:
            task_results['judge'] = judge_task

        return {
            'overall_score': overall_score,
            'num_samples': total,
            'task_results': task_results,
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