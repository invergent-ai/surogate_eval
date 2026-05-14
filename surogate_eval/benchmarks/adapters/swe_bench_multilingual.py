from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='swe_bench_multilingual',
        pretty_name='SWE-bench Multilingual',
        tags=[Tags.CODING],
        description="""
## Overview

SWE-bench Multilingual extends SWE-bench to cross-language software
engineering tasks spanning Python, Java, TypeScript, Go, Rust, and C++.

## Evaluation Notes

- Requires Docker sandbox for code execution
- Uses unit test pass rate as metric
- Tests multi-language code understanding
""",
        dataset_id='SWE-bench/SWE-bench_Multilingual',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='',
    )
)
class SWEBenchMultilingualAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt = record.get('problem_statement') or record.get('prompt') or record.get('input', '')
        target = record.get('patch') or record.get('answer') or record.get('expected_output', '')
        metadata = {
            'instance_id': record.get('instance_id', ''),
            'repo': record.get('repo', ''),
            'language': record.get('language', ''),
        }
        return Sample(input=[ChatMessageUser(content=prompt)] if prompt else prompt, target=str(target), metadata=metadata)
