from api.logging_config import base_logger
from logging import LoggerAdapter
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import uuid
from agent.runner import agent_run
from contracts import TraceCall
from api.config import LOG_LEVEL, RAG_ENABLED, RAG_TOP_K, MODEL, MAX_TRIES, SERVICE_NAME, API_VERSION, GIT_SHA, BUILD_TIME


# 1) The FastAPI "application" object (the HTTP router / server entry)
app = FastAPI(title="Intelligent Reasoner API")
"""
base_logger.info("Starting Intelligent Reasoner API.")
base_logger.info(f"The logging level is {LOG_LEVEL}.")
base_logger.info(f"The reasoner model is {MODEL}.")
base_logger.info(f"The status of using RAG is {RAG_ENABLED}.")
base_logger.info(f"The {RAG_TOP_K} similar contents will be used in the reasoning.")
"""
# 2) Request schema (what the client must send)
class AskRequest(BaseModel):
    query: str


# 3) Endpoint: POST /ask
@app.post("/ask", response_model=TraceCall)
def ask(req: AskRequest) -> TraceCall:
    # req.query is the text sent by the client
    request_id = str(uuid.uuid4())
    logger = LoggerAdapter(base_logger, {"request_id": request_id})
    logger.info(f"Received query:{req.query}")
    try:
        return agent_run(req.query,logger)
    except TimeoutError as e:
        logger.error(f"Unhandled error {e}", exc_info =True)
        raise HTTPException(
            status_code = 504,
            detail={
                "error_code":"LLM_TIMEOUT",
                "message":str(e),
                "request_id":request_id
            },
        ) 
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info = True)
        raise HTTPException(status_code = 500,
        detail={
            "error_code":"INTERNAL_ERROR",
            "message":"Unexpected server error",
            "request_id":request_id,
        },
        )       

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/ready")
def ready():
    return {
        "ready" : True,
        "rag_enabled" : RAG_ENABLED
    }

@app.get("/config")
def config():
    return {
        "log level" : LOG_LEVEL,
        "rag enabled" : RAG_ENABLED,
        "rag top k" : RAG_TOP_K,
        "model" : MODEL,
        "max tries" : MAX_TRIES
    }

@app.get("/version")
def version():
    return {
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "git_sha": GIT_SHA,
        "build_time": BUILD_TIME,
    }
