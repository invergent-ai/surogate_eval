import pytest
from pydantic import BaseModel

from surogate_eval.errors import JudgeParseError, JudgeUnavailableError
from surogate_eval.models.deepeval_wrapper import DeepEvalTargetWrapper
from surogate_eval.targets.base import TargetResponse


class Verdict(BaseModel):
    score: int


class FakeTarget:
    """A target under our control. Never touches the network."""

    def __init__(self, content="", error=None):
        self.name = "fake-judge"
        self.config = {"base_url": "https://api.openai.com/v1"}
        self._content = content
        self._error = error

    def send_request(self, request):
        return TargetResponse(
            content=self._content, raw_response={}, error=self._error,
        )


def test_target_error_raises_unavailable():
    wrapper = DeepEvalTargetWrapper(FakeTarget(error="HTTP 500"))
    with pytest.raises(JudgeUnavailableError):
        wrapper.generate("grade this", Verdict)


def test_empty_content_raises_unavailable():
    wrapper = DeepEvalTargetWrapper(FakeTarget(content=""))
    with pytest.raises(JudgeUnavailableError):
        wrapper.generate("grade this", Verdict)


def test_unparseable_content_raises_parse_error():
    """The judge answered in prose. Common with small judges."""
    wrapper = DeepEvalTargetWrapper(
        FakeTarget(content="I think the answer is pretty good overall.")
    )
    with pytest.raises(JudgeParseError):
        wrapper.generate("grade this", Verdict)


def test_valid_json_still_parses():
    wrapper = DeepEvalTargetWrapper(FakeTarget(content='{"score": 7}'))
    assert wrapper.generate("grade this", Verdict).score == 7


def test_markdown_wrapped_json_still_parses():
    wrapper = DeepEvalTargetWrapper(
        FakeTarget(content='```json\n{"score": 3}\n```')
    )
    assert wrapper.generate("grade this", Verdict).score == 3


def test_schemaless_call_raises_on_error():
    """Without a schema the old code returned "". Still an error."""
    wrapper = DeepEvalTargetWrapper(FakeTarget(error="HTTP 500"))
    with pytest.raises(JudgeUnavailableError):
        wrapper.generate("just text")


def test_empty_schema_helper_is_gone():
    """It fabricated malformed objects; nothing should resurrect it."""
    assert not hasattr(DeepEvalTargetWrapper, "_empty_schema")
