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
        problem = record.get('problem_statement', '')
        hints = record.get('hints_text', '')
        repo = record.get('repo', '')

        parts = [
            "You will be provided with a partial code base and an issue statement "
            "explaining a problem to resolve. Generate a patch (unified diff format) "
            "that resolves the issue.\n"
        ]
        if problem:
            parts.append(f"<issue>\n{problem}\n</issue>\n")
        if hints:
            parts.append(f"<hints>\n{hints}\n</hints>\n")
        if repo:
            parts.append(f"Repository: {repo}\n")
        parts.append(
            "Please provide your fix as a unified diff (patch). "
            "Only output the diff, nothing else."
        )

        prompt = "\n".join(parts)
        target = record.get('patch') or ''
        metadata = {
            'instance_id': record.get('instance_id', ''),
            'repo': repo,
        }
        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target=str(target),
            metadata=metadata,
        )
