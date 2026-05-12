from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='vita_bench',
        pretty_name='VITA-Bench',
        tags=[Tags.REASONING],
        description="""
## Overview

VITA-Bench evaluates visual-interactive task automation — GUI navigation,
web browsing, and app interaction for agent systems.

## Evaluation Notes

- Tests multi-step UI interaction planning
- Requires understanding of visual layouts and action sequences
""",
        dataset_id='meituan/VitaBench',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{prompt}',
    )
)
class VitaBenchAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt = record.get('prompt') or record.get('question') or record.get('input', '')
        target = record.get('answer') or record.get('target') or record.get('expected_output', '')
        return Sample(input=prompt, target=str(target))
