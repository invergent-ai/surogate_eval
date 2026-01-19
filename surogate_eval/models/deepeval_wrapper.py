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

        # If schema is provided, try to use guided decoding
        if schema:
            try:
                json_schema = schema.model_json_schema()
                # For vLLM guided decoding
                params = {"extra_body": {"guided_json": json_schema}}
                logger.debug(f"Using guided JSON schema")
            except Exception as e:
                logger.warning(f"Failed to get JSON schema: {e}")

        request = TargetRequest(prompt=prompt, parameters=params if params else None)
        response = self.target.send_request(request)
        logger.info(f"Response content length: {len(response.content) if response.content else 0}")
        logger.info(f"Response error: {response.error}")
        logger.info(f"Response preview: {response.content[:200] if response.content else 'EMPTY'}...")

        if response.error:
            logger.error(f"Target returned error: {response.error}")
            # Return empty but valid response to avoid JSON parse errors
            if schema:
                try:
                    # Try to create a minimal valid instance
                    return schema.model_construct()
                except:
                    pass
            return ""

        if not response.content:
            logger.warning(f"Empty response from target {self.target.name}")
            if schema:
                try:
                    return schema.model_construct()
                except:
                    pass
            return ""

        if schema:
            try:
                # Parse JSON and return schema instance
                json_result = json.loads(response.content)
                return schema(**json_result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response content: {response.content[:500]}")
                # Try to extract JSON from response
                try:
                    # Sometimes models wrap JSON in markdown
                    content = response.content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    json_result = json.loads(content.strip())
                    return schema(**json_result)
                except:
                    logger.error(f"Could not extract JSON from response")
                    try:
                        return schema.model_construct()
                    except:
                        raise
            except Exception as e:
                logger.error(f"Failed to create schema instance: {e}")
                raise

        return response.content

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        logger.debug(f"DeepEvalTargetWrapper.a_generate called")
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.target.name