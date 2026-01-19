# surogate_eval/models/deepeval_wrapper.py
import json
from typing import Union, Optional
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from ..targets.base import BaseTarget, TargetRequest
from ..utils.logger import get_logger

logger = get_logger()


class DeepEvalTargetWrapper(DeepEvalBaseLLM):
    """Wraps any BaseTarget for use with DeepEval/DeepTeam."""

    def __init__(self, target: BaseTarget):
        self.target = target
        # Don't call super().__init__() - we manage the model ourselves

    def load_model(self):
        return self.target

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        logger.info(f"=== DeepEvalTargetWrapper.generate CALLED ===")
        logger.info(f"Schema: {schema is not None}")
        logger.info(f"Prompt length: {len(prompt)}")
        logger.info(f"Prompt preview: {prompt[:300]}...")

        params = {}
        if schema:
            try:
                json_schema = schema.model_json_schema()
                params = {"extra_body": {"guided_json": json_schema}}
            except Exception as e:
                logger.warning(f"Failed to get JSON schema: {e}")

        request = TargetRequest(prompt=prompt, parameters=params if params else None)
        response = self.target.send_request(request)

        logger.info(f"Response content length: {len(response.content) if response.content else 0}")
        logger.info(f"Response error: {response.error}")
        logger.info(f"Response preview: {response.content[:300] if response.content else 'EMPTY'}...")

        if response.error:
            logger.error(f"Target returned error: {response.error}")
            if schema:
                return schema.model_construct()
            return ""

        if not response.content:
            logger.warning(f"Empty response from target {self.target.name}")
            if schema:
                return schema.model_construct()
            return ""

        content = response.content

        # Handle thinking models - extract content after </think> tag
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
            logger.info(f"Extracted content after </think>: {content[:300]}...")

        if schema:
            try:
                json_result = json.loads(content)
                return schema(**json_result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Try to extract JSON from markdown or other wrappers
                try:
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    # Try to find JSON object/array
                    elif "{" in content:
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        content = content[start:end]
                    elif "[" in content:
                        start = content.find("[")
                        end = content.rfind("]") + 1
                        content = content[start:end]

                    logger.info(f"Extracted JSON: {content[:300]}...")
                    json_result = json.loads(content)
                    return schema(**json_result)
                except Exception as ex:
                    logger.error(f"Could not extract JSON from response: {ex}")
                    return schema.model_construct()

        return content

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        logger.debug(f"DeepEvalTargetWrapper.a_generate called")
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.target.name