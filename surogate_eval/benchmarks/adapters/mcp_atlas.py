"""MCP-Atlas benchmark adapter — multi-turn trajectory replay.

ScaleAI/MCP-Atlas evaluates multi-step MCP tool use.  Each sample has:
  - PROMPT: user question
  - ENABLED_TOOLS: list of tool name strings available to the model
  - TRAJECTORY: reference multi-turn assistant/tool message sequence
  - GTFA_CLAIMS: factual assertions the final answer must satisfy

This adapter overrides ``_on_inference`` to run a multi-turn loop:
the model generates a response; if it makes a tool call, we look up
the matching tool response from the reference TRAJECTORY and feed it
back as a tool message.  The loop repeats until the model produces a
final text response (no tool calls) or we hit the turn limit.

Scoring checks how many GTFA_CLAIMS appear in the model's final
response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from evalscope.api.metric.scorer import Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

logger = logging.getLogger(__name__)

MAX_TURNS = 15  # safety limit for the agent loop


_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use the available tools to gather information and answer the user's question accurately. "
    "When you have enough information, provide a clear final answer."
)


def _parse_json_field(raw: Any) -> Any:
    """Parse a field that may be a JSON string or already decoded."""
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: field may use Python repr (single quotes) instead of JSON.
        try:
            import ast
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
    return raw or []


def _build_trajectory_lookup(trajectory: List[dict]) -> Dict[str, dict]:
    """Index tool responses from the reference trajectory by the preceding tool call name.

    Returns a dict mapping ``tool_call_name`` → tool response dict so we
    can replay the correct response when the model calls a tool.

    The trajectory alternates assistant (with tool_calls) and tool messages.
    We pair each assistant tool call with the immediately following tool
    message that shares the same ``tool_call_id``.
    """
    lookup: Dict[str, list] = {}
    for i, entry in enumerate(trajectory):
        if entry.get("role") == "assistant" and entry.get("tool_calls"):
            for tc in entry["tool_calls"]:
                tc_id = tc.get("id", "")
                func_name = tc.get("function", {}).get("name", "")
                # Find matching tool response
                for j in range(i + 1, min(i + 5, len(trajectory))):
                    resp = trajectory[j]
                    if resp.get("role") == "tool":
                        resp_id = resp.get("tool_call_id", "")
                        if resp_id == tc_id or not resp_id:
                            key = func_name
                            if key not in lookup:
                                lookup[key] = []
                            lookup[key].append(resp)
                            break
    return lookup


def _find_tool_response(
    lookup: Dict[str, list],
    call_counters: Dict[str, int],
    tool_name: str,
) -> Optional[str]:
    """Find the next tool response for *tool_name* from the reference trajectory."""
    responses = lookup.get(tool_name, [])
    idx = call_counters.get(tool_name, 0)
    call_counters[tool_name] = idx + 1
    if idx < len(responses):
        resp = responses[idx]
        content = resp.get("content", "")
        if isinstance(content, list):
            # Content may be a list of dicts with 'text' keys
            parts = []
            for c in content:
                if isinstance(c, dict):
                    parts.append(c.get("text", str(c)))
                else:
                    parts.append(str(c))
            return "\n".join(parts)
        return str(content)
    return None


@register_benchmark(
    BenchmarkMeta(
        name='mcp_atlas',
        pretty_name='MCP-Atlas',
        tags=[Tags.REASONING],
        description="""
## Overview

MCP-Atlas evaluates large-scale Model Context Protocol tool use, covering
diverse MCP server types and multi-step tool chaining scenarios.

## Evaluation Notes

- Multi-turn trajectory replay: tool responses come from the reference trajectory
- Scores against GTFA_CLAIMS (factual assertions) in the final response
- Tests tool discovery, invocation sequencing, and answer synthesis
""",
        dataset_id='ScaleAI/MCP-Atlas',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='train',
        prompt_template='',
    )
)
class MCPAtlasAdapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt = record.get('PROMPT') or record.get('prompt') or ''
        tools_raw = record.get('ENABLED_TOOLS') or record.get('enabled_tools') or '[]'
        claims_raw = record.get('GTFA_CLAIMS') or record.get('gtfa_claims') or '[]'
        traj_raw = record.get('TRAJECTORY') or record.get('trajectory') or '[]'

        tools: List[str] = _parse_json_field(tools_raw)
        claims: List[str] = _parse_json_field(claims_raw)
        trajectory: List[dict] = _parse_json_field(traj_raw)

        return Sample(
            input=[
                ChatMessageSystem(content=_SYSTEM_PROMPT),
                ChatMessageUser(content=prompt),
            ],
            target=' '.join(claims) if claims else '',
            metadata={
                'claims': claims,
                'tool_names': tools,
                'trajectory': trajectory,
            },
        )

    def _on_inference(self, model, sample: Sample) -> ModelOutput:
        """Multi-turn agent loop with trajectory replay."""
        trajectory = sample.metadata.get('trajectory', [])
        lookup = _build_trajectory_lookup(trajectory)
        call_counters: Dict[str, int] = {}

        messages: List[ChatMessage] = list(sample.input)
        last_output: Optional[ModelOutput] = None

        for turn in range(MAX_TURNS):
            output = model.generate(input=messages, tools=sample.tools)
            last_output = output

            if output.empty or output.error:
                break

            choice = output.choices[0]
            # Append assistant message to conversation
            messages.append(choice.message)

            # If no tool calls, the model is done
            if not choice.message.tool_calls or choice.stop_reason != 'tool_calls':
                break

            # Process each tool call — look up response from trajectory
            all_found = True
            for tc in choice.message.tool_calls:
                tool_name = tc.function.name
                tool_response = _find_tool_response(lookup, call_counters, tool_name)
                if tool_response is None:
                    # No matching response in trajectory — tell the model
                    tool_response = f"[Tool '{tool_name}' not available in reference trajectory]"
                    all_found = False

                messages.append(ChatMessageTool(
                    content=tool_response,
                    tool_call_id=tc.id,
                    function=tool_name,
                ))

            if not all_found:
                logger.debug(
                    "MCP-Atlas: model called tool(s) not in reference trajectory at turn %d",
                    turn,
                )

        return last_output or ModelOutput.from_content(model=model.model_name, content="")

    def match_score(self, original_prediction, filtered_prediction, reference, task_state, **kwargs) -> Score:
        """Score by checking how many GTFA_CLAIMS appear in the model output."""
        response = (filtered_prediction or '').lower()
        claims = task_state.metadata.get('claims', [])

        if not claims:
            return Score(extracted_prediction=filtered_prediction, prediction=original_prediction, value={'acc': 0.0})

        hits = sum(1 for c in claims if c.lower() in response)
        score = hits / len(claims)

        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'acc': score},
            metadata={'claims_hit': hits, 'claims_total': len(claims)},
        )
