from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    text: str
    file_name: str
    chunk_id: str
    source:str


class Hit(BaseModel):
    id:str
    score: float = Field(...,ge=0.0, le = 1.0)
    document : str
    metadata: Dict[str, Any]


class DocumentText(BaseModel):
    file_name: str
    text: str

class LLMAnswer(BaseModel):
    cause: Literal["Manufacturing", "DemandSpike", "Quality", "Logistics", "Unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1)