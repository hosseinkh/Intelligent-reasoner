from typing import Dict, Any, Literal,List, Optional
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
################ Version 2 aditional contracts ###################

ToolName = Literal["rag_search","llm_answer"]
ToolChoice = ToolName | None

class ToolCall(BaseModel):
    name : ToolName
    input_text : str
    output_summary : str
    status : str
    ms : int
    error : str | None = None

class TraceCall(BaseModel):
    run_id : str
    query : str
    router_policy: str
    router_trigger : Optional[str] = None
    rag_enabled: bool
    tool_calls : List[ToolCall]
    status : str
    final_answer : str
    sources: Optional[List[str]] = None
    retrieval_stats:Optional[Dict] = None
    total_ms : int
