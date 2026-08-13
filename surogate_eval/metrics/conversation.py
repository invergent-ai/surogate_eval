# surogate/eval/metrics/conversation.py
from typing import Dict, Any, Optional, Union, List

from .base import MetricResult, MetricType, LLMJudgeMetric
from .registry import register_metric
import json
import re

from ..datasets import MultiTurnTestCase, TestCase
from ..errors import JudgeParseError, JudgeUnavailableError
from ..targets import TargetResponse
from ..utils.logger import get_logger
from surogate_eval.benchmarks.pass_rule import LEGACY_JUDGE_MIN, row_passed

logger = get_logger()


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response_text, re.DOTALL)

    if match:
        return match.group(1).strip()

    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, response_text, re.DOTALL)

    if match:
        return match.group(0).strip()

    return response_text.strip()


@register_metric(MetricType.CONVERSATION_COHERENCE)
class ConversationCoherenceMetric(LLMJudgeMetric):
    """Measure coherence in multi-turn conversations using LLM judge."""

    def _validate_config(self):
        """Validate configuration."""
        if 'window_size' not in self.config:
            self.config['window_size'] = 3  # Look at last 3 turns

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate conversation coherence using LLM judge.

        Args:
            test_case: Must be MultiTurnTestCase
            actual_output: Actual output from target
            target_response: Full response from target

        Returns:
            Metric result
        """
        if not isinstance(test_case, MultiTurnTestCase):
            # A misconfigured metric never measured anything. Scoring it
            # 0.0 wrote a configuration mistake into the average as if
            # the model had answered badly.
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason="Conversation coherence requires multi-turn test case",
                metadata={'error_kind': 'config'},
            )

        if not actual_output:
            return self._no_output_result(target_response)

        try:
            if not self.judge_target:
                return MetricResult.errored(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    reason="No judge target set; cannot assess coherence.",
                    metadata={'error_kind': 'no_judge'},
                )

            window_size = self.config['window_size']
            num_turns = len(test_case.turns)

            if num_turns < 2:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=1.0,
                    success=True,
                    reason="Single turn, trivially coherent"
                )

            # Get recent conversation context
            recent_turns = test_case.turns[-window_size:] if len(test_case.turns) > window_size else test_case.turns
            conversation_text = "\n".join([f"{turn.role}: {turn.content}" for turn in recent_turns])
            conversation_text += f"\nassistant: {actual_output}"

            # Build coherence evaluation prompt
            prompt = f"""Evaluate the coherence of this conversation on a scale of 0-10, where:
- 0 = Completely incoherent, responses don't make sense in context
- 10 = Perfectly coherent, natural flow and logical progression

Consider:
- Does each response follow logically from previous turns?
- Are topic transitions smooth and natural?
- Does the assistant maintain context appropriately?
- Is the conversation easy to follow?

Conversation:
{conversation_text}

Provide your evaluation in JSON format:
{{"coherence_score": <0-10>, "reason": "<brief explanation>", "issues": ["<issue1>", "<issue2>", ...]}}"""

            from surogate_eval.targets.base import TargetRequest
            request = TargetRequest(prompt=prompt)
            response = self.judge_target.send_request(request)

            if response.error:
                raise JudgeUnavailableError(f"judge request failed: {response.error}")

            # Parse response
            try:
                clean_json = extract_json_from_response(response.content)
                result = json.loads(clean_json)

                coherence_raw = float(result['coherence_score'])
                coherence_score = coherence_raw / 10.0  # Normalize to 0-1
                reason = result.get('reason', 'No reason provided')
                issues = result.get('issues', [])

                logger.debug(f"✅ Parsed coherence: {coherence_score:.2f}")

            except Exception as e:
                logger.warning(f"Failed to parse coherence response: {e}")
                logger.debug(f"Raw response: {response.content[:300]}")
                raise JudgeParseError(
                    f"judge returned unparseable coherence output: {e}"
                ) from e

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=coherence_score,
                success=row_passed(
                    coherence_score,
                    legacy_minimum=LEGACY_JUDGE_MIN, legacy_inclusive=True,
                ),
                reason=f"Coherence: {coherence_score:.2f} - {reason[:100]}",
                metadata={
                    'coherence_raw_score': coherence_raw,
                    'window_size': window_size,
                    'num_turns': num_turns,
                    'issues': issues
                }
            )

        except Exception as e:
            logger.error(f"Coherence evaluation failed: {e}")
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Evaluation error: {e}",
                metadata={'error_kind': type(e).__name__},
            )


@register_metric(MetricType.CONTEXT_RETENTION)
class ContextRetentionMetric(LLMJudgeMetric):
    """Measure how well context is retained across turns using LLM judge."""

    def _validate_config(self):
        """Validate configuration."""
        if 'key_info_threshold' not in self.config:
            self.config['key_info_threshold'] = 0.7

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate context retention using LLM judge.

        Args:
            test_case: Must be MultiTurnTestCase
            actual_output: Actual output from target
            target_response: Full response from target

        Returns:
            Metric result
        """
        if not isinstance(test_case, MultiTurnTestCase):
            # A misconfigured metric never measured anything. Scoring it
            # 0.0 wrote a configuration mistake into the average as if
            # the model had answered badly.
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason="Context retention requires multi-turn test case",
                metadata={'error_kind': 'config'},
            )

        if not actual_output:
            return self._no_output_result(target_response)

        try:
            if not self.judge_target:
                return MetricResult.errored(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    reason="No judge target set; cannot assess context retention.",
                    metadata={'error_kind': 'no_judge'},
                )

            # Build full conversation
            conversation_text = "\n".join([f"{turn.role}: {turn.content}" for turn in test_case.turns])
            conversation_text += f"\nassistant: {actual_output}"

            # Build context retention evaluation prompt
            prompt = f"""Evaluate how well the assistant retains and uses context from earlier in the conversation on a scale of 0-10, where:
- 0 = No context retention, ignores previous information
- 10 = Perfect context retention, appropriately references and builds on earlier information

Consider:
- Does the assistant remember key information from earlier turns?
- Does it appropriately reference or build upon previous context?
- Are there contradictions with earlier statements?
- Does it ask for information already provided?

Conversation:
{conversation_text}

Provide your evaluation in JSON format:
{{"retention_score": <0-10>, "reason": "<brief explanation>", "retained_info": ["<info1>", "<info2>", ...], "missed_info": ["<missed1>", "<missed2>", ...]}}"""

            from surogate_eval.targets.base import TargetRequest
            request = TargetRequest(prompt=prompt)
            response = self.judge_target.send_request(request)

            if response.error:
                raise JudgeUnavailableError(f"judge request failed: {response.error}")

            # Parse response
            try:
                clean_json = extract_json_from_response(response.content)
                result = json.loads(clean_json)

                retention_raw = float(result['retention_score'])
                retention_score = retention_raw / 10.0  # Normalize to 0-1
                reason = result.get('reason', 'No reason provided')
                retained_info = result.get('retained_info', [])
                missed_info = result.get('missed_info', [])

                logger.debug(f"✅ Parsed context retention: {retention_score:.2f}")

            except Exception as e:
                logger.warning(f"Failed to parse retention response: {e}")
                logger.debug(f"Raw response: {response.content[:300]}")
                raise JudgeParseError(
                    f"judge returned unparseable context retention output: {e}"
                ) from e

            threshold = self.config['key_info_threshold']

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=retention_score,
                success=retention_score >= threshold,
                reason=f"Context retention: {retention_score:.2f} - {reason[:100]}",
                metadata={
                    'retention_raw_score': retention_raw,
                    'threshold': threshold,
                    'num_turns': len(test_case.turns),
                    'retained_info': retained_info,
                    'missed_info': missed_info
                }
            )

        except Exception as e:
            logger.error(f"Context retention evaluation failed: {e}")
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Evaluation error: {e}",
                metadata={'error_kind': type(e).__name__},
            )


@register_metric(MetricType.TURN_ANALYSIS)
class TurnAnalysisMetric(LLMJudgeMetric):
    """Analyze individual turns in a conversation using LLM judge."""

    def _validate_config(self):
        """Validate configuration."""
        if 'analyze_all_turns' not in self.config:
            self.config['analyze_all_turns'] = True

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Analyze each turn using LLM judge.

        Args:
            test_case: Must be MultiTurnTestCase
            actual_output: Actual output from target
            target_response: Full response from target

        Returns:
            Metric result with per-turn scores
        """
        if not isinstance(test_case, MultiTurnTestCase):
            # A misconfigured metric never measured anything. Scoring it
            # 0.0 wrote a configuration mistake into the average as if
            # the model had answered badly.
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason="Turn analysis requires multi-turn test case",
                metadata={'error_kind': 'config'},
            )

        if not actual_output:
            return self._no_output_result(target_response)

        try:
            if not self.judge_target:
                return MetricResult.errored(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    reason="No judge target set; cannot assess turn quality.",
                    metadata={'error_kind': 'no_judge'},
                )

            # Analyze the final assistant turn (the actual_output)
            # Build context from previous turns
            context_turns = "\n".join([f"{turn.role}: {turn.content}" for turn in test_case.turns[-3:]])

            prompt = f"""Analyze the quality of this assistant response on a scale of 0-10, where:
- 0 = Poor quality (irrelevant, unhelpful, or inappropriate)
- 10 = Excellent quality (relevant, helpful, accurate, and well-structured)

Consider:
- Relevance to the user's query
- Helpfulness and completeness
- Accuracy of information
- Clarity and structure
- Appropriateness of tone

Recent conversation context:
{context_turns}

Assistant's response:
{actual_output}

Provide your evaluation in JSON format:
{{"quality_score": <0-10>, "reason": "<brief explanation>", "strengths": ["<strength1>", "<strength2>", ...], "weaknesses": ["<weakness1>", "<weakness2>", ...]}}"""

            from surogate_eval.targets.base import TargetRequest
            request = TargetRequest(prompt=prompt)
            response = self.judge_target.send_request(request)

            if response.error:
                raise JudgeUnavailableError(f"judge request failed: {response.error}")

            # Parse response
            try:
                clean_json = extract_json_from_response(response.content)
                result = json.loads(clean_json)

                quality_raw = float(result['quality_score'])
                quality_score = quality_raw / 10.0  # Normalize to 0-1
                reason = result.get('reason', 'No reason provided')
                strengths = result.get('strengths', [])
                weaknesses = result.get('weaknesses', [])

                logger.debug(f"✅ Parsed turn quality: {quality_score:.2f}")

            except Exception as e:
                logger.warning(f"Failed to parse turn analysis response: {e}")
                logger.debug(f"Raw response: {response.content[:300]}")
                raise JudgeParseError(
                    f"judge returned unparseable turn analysis output: {e}"
                ) from e

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=quality_score,
                success=row_passed(
                    quality_score,
                    legacy_minimum=LEGACY_JUDGE_MIN, legacy_inclusive=True,
                ),
                reason=f"Turn quality: {quality_score:.2f} - {reason[:100]}",
                metadata={
                    'quality_raw_score': quality_raw,
                    'num_turns': len(test_case.turns),
                    'strengths': strengths,
                    'weaknesses': weaknesses
                }
            )

        except Exception as e:
            logger.error(f"Turn analysis failed: {e}")
            return MetricResult.errored(
                metric_name=self.name,
                metric_type=self.metric_type,
                reason=f"Evaluation error: {e}",
                metadata={'error_kind': type(e).__name__},
            )