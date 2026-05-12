from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_pro',
        pretty_name='SWE-bench Pro',
        tags=[Tags.CODING],
        description="""
## Overview

SWE-bench Pro contains professional-difficulty software engineering tasks
with complex multi-step debugging and refactoring requirements.

## Evaluation Notes

- Requires Docker sandbox for code execution
- Uses unit test pass rate as metric
""",
        dataset_id='ScaleAI/SWE-bench_Pro',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{problem_statement}',
    )
)
class SWEBenchProAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt = record.get('problem_statement') or record.get('prompt') or record.get('input', '')
        target = record.get('patch') or record.get('answer') or record.get('expected_output', '')
        metadata = {
            'instance_id': record.get('instance_id', ''),
            'repo': record.get('repo', ''),
        }
        return Sample(input=prompt, target=str(target), metadata=metadata)
