"""Wide Search adapter — multi-hop web research evaluation.

Uses the DeepResearch-9K dataset: 12,974 multi-hop questions across three
difficulty levels requiring web search, reasoning chains, and information
synthesis to arrive at a final answer.

Difficulty levels:
  - L1 (simple): single-hop factual questions
  - L2 (moderate): multi-hop questions requiring 2-3 search steps
  - L3 (hard): deep chain reasoning requiring 5+ steps and synthesis

The model is expected to use its web search tools to find and synthesize
information. Scoring uses LLM-as-judge to compare the model's answer
against the reference answer, accounting for equivalent phrasings.

Dataset columns: question, difficulty, search trajectory, final answer
"""

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the following question "
    "accurately. If you have access to tools such as web search, use "
    "them to verify your answer. Provide a concise, factual answer."
)


@register_benchmark(
    BenchmarkMeta(
        name='wide_search',
        pretty_name='WideSearch (DeepResearch-9K)',
        tags=[Tags.REASONING, Tags.QA],
        description="""\
## Overview

Multi-hop web research benchmark using DeepResearch-9K — questions
requiring web search, multi-step reasoning, and information synthesis.

## Dataset

artillerywu/DeepResearch-9K — 12,974 questions across three difficulty
levels (L1 simple, L2 multi-hop, L3 deep chain reasoning).

## Evaluation

LLM-as-judge compares model answers against reference answers,
accounting for equivalent phrasings and partial correctness.
""",
        dataset_id='artillerywu/DeepResearch-9K',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template='',
    )
)
class WideSearchAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record.get('question', '')
        answer = record.get('final answer', '')
        difficulty = record.get('difficulty', 0)
        trajectory = record.get('search trajectory', [])

        # Count reference search steps for metadata
        num_steps = len(trajectory) if isinstance(trajectory, list) else 0

        prompt = question

        return Sample(
            input=[
                ChatMessageSystem(content=_SYSTEM_PROMPT),
                ChatMessageUser(content=prompt),
            ],
            target=str(answer),
            metadata={
                'difficulty': difficulty,
                'num_reference_steps': num_steps,
            },
        )
