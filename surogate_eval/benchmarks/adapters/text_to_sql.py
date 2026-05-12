from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='text_to_sql',
        pretty_name='Text-to-SQL',
        tags=[Tags.CODING],
        description="""
## Overview

Text-to-SQL benchmark using synthetic data to evaluate natural language
to SQL generation correctness against gold queries.

## Evaluation Notes

- Dataset: swift/synthetic_text_to_sql on ModelScope
- Tests SQL generation accuracy via execution match
""",
        dataset_id='swift/synthetic_text_to_sql',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='Generate the SQL query for the following question:\n\n{prompt}',
    )
)
class TextToSQLAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt = record.get('prompt') or record.get('question') or record.get('input', '')
        target = record.get('sql') or record.get('query') or record.get('response') or record.get('target', '')
        return Sample(input=prompt, target=str(target))
