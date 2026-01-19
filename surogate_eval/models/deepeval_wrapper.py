# surogate_eval/models/deepeval_wrapper.py
import json
from typing import Union
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from ..targets.base import BaseTarget, TargetRequest


class DeepEvalTargetWrapper(DeepEvalBaseLLM):
    """Wraps any BaseTarget for use with DeepEval/DeepTeam."""

    def __init__(self, target: BaseTarget):
        self.target = target

    def load_model(self):
        return self.target

    def generate(self, prompt: str, schema: BaseModel = None) -> Union[str, BaseModel]:
        params = None
        if schema:
            # vLLM guided decoding
            params = {"extra_body": {"guided_json": schema.model_json_schema()}}

        request = TargetRequest(prompt=prompt, parameters=params)
        response = self.target.send_request(request)

        if schema:
            return schema(**json.loads(response.content))
        return response.content

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> Union[str, BaseModel]:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.target.name