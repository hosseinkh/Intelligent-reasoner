from fastapi import FastAPI
from pydantic import BaseModel

from agent.runner import agent_run
from contracts import TraceCall


# 1) The FastAPI "application" object (the HTTP router / server entry)
app = FastAPI(title="Intelligent Reasoner API")


# 2) Request schema (what the client must send)
class AskRequest(BaseModel):
    query: str


# 3) Endpoint: POST /ask
@app.post("/ask", response_model=TraceCall)
def ask(req: AskRequest) -> TraceCall:
    # req.query is the text sent by the client
    return agent_run(req.query)
