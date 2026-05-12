from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='deep_planning',
        pretty_name='DeepPlanning',
        tags=[Tags.REASONING],
        description="""
## Overview

DeepPlanning evaluates LLM agents on complex multi-constraint planning tasks
requiring long-horizon reasoning and resource allocation.

## Evaluation Notes

- Uses LLM-as-judge for scoring plan quality
- Tests constraint satisfaction, feasibility, and optimality
""",
        dataset_id='Qwen/DeepPlanning',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{prompt}',
    )
)
class DeepPlanningAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt = record.get('prompt') or record.get('question') or record.get('input', '')
        target = record.get('answer') or record.get('target') or record.get('expected_output', '')
        return Sample(input=prompt, target=str(target))
