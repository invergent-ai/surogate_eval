from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
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
        prompt_template='',
    )
)
class SWEBenchProAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        problem = record.get('problem_statement', '')
        requirements = record.get('requirements', '')
        interface = record.get('interface', '')
        repo_lang = record.get('repo_language', '')

        parts = [
            "You will be provided with an issue statement and technical context. "
            "Generate a patch (unified diff format) that resolves the issue.\n"
        ]
        if problem:
            parts.append(f"<issue>\n{problem}\n</issue>\n")
        if requirements:
            parts.append(f"<requirements>\n{requirements}\n</requirements>\n")
        if interface:
            parts.append(f"<interface>\n{interface}\n</interface>\n")
        if repo_lang:
            parts.append(f"Language: {repo_lang}\n")
        parts.append(
            "Please provide your fix as a unified diff (patch). "
            "Only output the diff, nothing else."
        )

        prompt = "\n".join(parts)
        target = record.get('patch') or ''
        metadata = {
            'instance_id': record.get('instance_id', ''),
            'repo': record.get('repo', ''),
            'difficulty': record.get('issue_specificity', ''),
        }
        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target=str(target),
            metadata=metadata,
        )
