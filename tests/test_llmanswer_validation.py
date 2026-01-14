import json
import pytest
from pydantic import ValidationError
from contracts import LLMAnswer

def test_llmanswer_valid_json_passes():
    raw = '{"cause":"Manufacturing","confidence":0.8,"source":"doc1.pdf"}'
    data = json.loads(raw)
    ans = LLMAnswer.model_validate(data)
    assert ans.cause == "Manufacturing"
    assert ans.confidence == 0.8
    assert ans.source == "doc1.pdf"

def test_llmanswer_invalid_json_fails_decode():
    raw = 'cause: Manufacturing, confidence: 0.8'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

def test_llmanswer_wrong_schema_fails_validation():
    raw = '{"cause":"Other","confidence":0.8,"source":"doc1.pdf"}'
    data = json.loads(raw)
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate(data)

def test_llmanswer_confidence_out_of_range_fails():
    raw = '{"cause":"Manufacturing","confidence":1.5,"source":"doc1.pdf"}'
    data = json.loads(raw)
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate(data)
