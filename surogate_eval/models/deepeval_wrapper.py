# surogate_eval/models/deepeval_wrapper.py
import json
from typing import Union
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from ..targets.base import BaseTarget, TargetRequest
from ..utils.logger import get_logger

logger = get_logger()


class DeepEvalTargetWrapper(DeepEvalBaseLLM):
    """Wraps any BaseTarget for use with DeepEval/DeepTeam."""

    def __init__(self, target: BaseTarget):
        self.target = target

    def load_model(self):
        return self.target

    def generate(self, prompt: str, schema: BaseModel = None) -> Union[str, BaseModel]:
        logger.debug(f"DeepEvalTargetWrapper.generate called with prompt length: {len(prompt)}")

        params = None
        if schema:
            params = {"extra_body": {"guided_json": schema.model_json_schema()}}

        request = TargetRequest(prompt=prompt, parameters=params)
        response = self.target.send_request(request)

        logger.debug(
            f"DeepEvalTargetWrapper got response: content_length={len(response.content)}, error={response.error}")

        if response.error:
            logger.error(f"Target returned error: {response.error}")

        if not response.content:
            logger.warning(f"Empty response from target {self.target.name}")

        if schema:
            return schema(**json.loads(response.content))
        return response.content

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> Union[str, BaseModel]:
        logger.debug(f"DeepEvalTargetWrapper.a_generate called")
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.target.name