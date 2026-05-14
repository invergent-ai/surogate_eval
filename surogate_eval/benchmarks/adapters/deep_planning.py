"""DeepPlanning benchmark adapter — STUB, NOT FUNCTIONAL.

Qwen/DeepPlanning is a database-backed travel planning benchmark.
The dataset contains CSV databases (trains, hotels, flights,
restaurants, attractions) — NOT prompt/answer pairs.  A proper
adapter would need to expose these as queryable tools and evaluate
multi-step plans.

This adapter is currently a stub that registers the benchmark name
so evalscope doesn't crash.  Full implementation requires a custom
multi-turn backend with database-query tools.

TODO: Proper implementation needs:
  1. Natural language planning queries (not raw CSV)
  2. Database query tools the model can call (SQL or structured)
  3. Multi-constraint validation (budget, timing, preferences)
  4. Reference plans for scoring (or LLM judge with rubric)

The benchmark is DISABLED in the ops frontend until this is built.
"""

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser
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

## Status

**Not yet implemented** — the dataset is structured as CSV databases
(trains, hotels, flights, etc.), not prompt/answer pairs.  Requires a
custom multi-turn backend with database-query tools.
""",
        dataset_id='Qwen/DeepPlanning',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='',
    )
)
class DeepPlanningAdapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Dataset rows are CSV blobs (database tables), not prompts.
        # Extract whatever text we can for a minimal run.
        key = record.get('__key__', '')
        csv_data = record.get('csv')
        if csv_data and isinstance(csv_data, bytes):
            csv_data = csv_data.decode('utf-8', errors='replace')

        prompt = (
            f"You are given a travel database entry ({key}). "
            f"Summarize the key information:\n\n{(csv_data or '')[:2000]}"
        )
        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target='',
            metadata={'key': key},
        )
