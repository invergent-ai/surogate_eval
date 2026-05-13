"""MT-Bench adapter for evalscope.

MT-Bench is a multi-turn conversation benchmark with 80 questions across
8 categories. Each question has 2 turns — the model answers turn 1, then
turn 2 is a follow-up that tests instruction-following and coherence.

Scoring is done by an LLM judge on a 1-10 scale. Since evalscope's native
pipeline is single-turn, this adapter concatenates both turns into one
prompt and uses the judge for scoring.

Dataset: HuggingFaceH4/mt_bench_prompts
"""

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


MT_BENCH_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's questions thoughtfully "
    "and thoroughly."
)

MT_BENCH_JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. You will rate the response on a scale of 1-10 where 1 is very poor and 10 is excellent. Consider the following criteria:

1. Helpfulness: Does the response address the question?
2. Relevance: Is the response on-topic?
3. Accuracy: Is the information correct?
4. Depth: Does it provide sufficient detail?
5. Creativity: For creative tasks, is the response engaging?
6. Coherence: Is the response well-structured?

Please provide your rating as a single number between 1 and 10, followed by a brief explanation.

Rating:"""


@register_benchmark(
    BenchmarkMeta(
        name='mt_bench',
        pretty_name='MT-Bench',
        tags=[Tags.MULTI_TURN, Tags.INSTRUCTION_FOLLOWING],
        description="""
## Overview

MT-Bench evaluates multi-turn conversation quality across 8 categories:
writing, roleplay, extraction, reasoning, math, coding, STEM, and humanities.

Each question has 2 turns. The model answers turn 1, then turn 2 is a
follow-up that tests instruction-following and contextual understanding.

## Scoring

Uses LLM-as-judge to rate responses on a 1-10 scale.

## Dataset

80 questions from HuggingFaceH4/mt_bench_prompts.
""",
        dataset_id='HuggingFaceH4/mt_bench_prompts',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # mt_bench only has a train split
        prompt_template='{prompt}',
    )
)
class MTBenchAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompts = record.get('prompt', [])
        category = record.get('category', '')
        reference = record.get('reference', [])

        # Build multi-turn prompt: concatenate both turns with clear markers.
        if isinstance(prompts, list) and len(prompts) >= 2:
            prompt = (
                f"[Category: {category}]\n\n"
                f"Turn 1: {prompts[0]}\n\n"
                f"Turn 2 (follow-up): {prompts[1]}\n\n"
                "Please answer both turns. First answer Turn 1, then answer "
                "Turn 2 as a follow-up."
            )
        elif isinstance(prompts, list) and len(prompts) == 1:
            prompt = f"[Category: {category}]\n\n{prompts[0]}"
        else:
            prompt = str(prompts)

        # Reference answer (if available)
        target = ""
        if isinstance(reference, list) and reference:
            target = " | ".join(str(r) for r in reference)

        return Sample(input=prompt, target=target)
