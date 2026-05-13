"""Text-to-SQL adapter for evalscope.

Uses the b-mc2/sql-create-context dataset which has natural language
questions with table schemas and gold SQL queries.

Dataset columns: question, context (CREATE TABLE), answer (SQL)
"""

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

Text-to-SQL benchmark evaluating natural language to SQL generation
using the sql-create-context dataset with table schemas.

## Dataset

b-mc2/sql-create-context — 78,577 examples with question, schema, and
gold SQL query.

## Evaluation

Exact match between generated SQL and gold query (normalized).
""",
        dataset_id='b-mc2/sql-create-context',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='{prompt}',
    )
)
class TextToSQLAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record.get('question', '')
        context = record.get('context', '')
        answer = record.get('answer', '')

        prompt = (
            f"Given the following SQL table schema:\n\n"
            f"{context}\n\n"
            f"Write a SQL query to answer: {question}\n\n"
            f"Return only the SQL query, nothing else."
        )

        return Sample(input=prompt, target=str(answer))
