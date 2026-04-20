from api.logging_config import base_logger
from logging import LoggerAdapter
from fastapi import FastAPI, HTTPException, Request,Response
from pydantic import BaseModel
import uuid
from agent.runner import agent_run
from contracts import TraceCall
from ingest_selector import run_ingest
import base64
import json
from api.config import (
    LOG_LEVEL,
    RAG_ENABLED,
    RAG_TOP_K,
    MODEL,
    MAX_TRIES,
    SERVICE_NAME,
    API_VERSION,
    GIT_SHA,
    BUILD_TIME,
    LLM_PROVIDER,
    RAG_ENABLED,
    GOOGLE_CLOUD_PROJECT,VERTEX_LOCATION,VERTEX_MODEL
)
from api.logging_config import setup_logging,MergeExtraAdapter,base_logger
import time
import httpx
from prometheus_client import generate_latest , CONTENT_TYPE_LATEST
from api.metrics import HTTP_REQUEST_LATENCY_S,HTTP_REQUEST_TOTAL

setup_logging()

# 1) The FastAPI "application" object (the HTTP router / server entry)
app = FastAPI(title="Intelligent Reasoner API")
"""
base_logger.info("Starting Intelligent Reasoner API.")
base_logger.info(f"The logging level is {LOG_LEVEL}.")
base_logger.info(f"The reasoner model is {MODEL}.")
base_logger.info(f"The status of using RAG is {RAG_ENABLED}.")
base_logger.info(f"The {RAG_TOP_K} similar contents will be used in the reasoning.")
"""


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):

    request_id = str(uuid.uuid4())
    logger = MergeExtraAdapter(base_logger, {"request_id": request_id})
    request.state.logger = logger
    request.state.request_id = request_id

    start = time.perf_counter()
    logger.info(
        "request.state",
        extra={
            "event": "request.start",
            "method": request.method,
            "path": request.url.path,
        },
    )
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "request.error",
            extra={
                "event": "request.error",
                "method": request.method,
                "path": request.url.path,
                "ms": duration_ms,
            },
        )
        raise
    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "request.done",
        extra={
            "event": "request.done",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    response.headers["X-request-id"] = request_id
    HTTP_REQUEST_TOTAL.labels(
        method = request.method,
        path = request.url.path,
        status = response.status_code
    ).inc()
    HTTP_REQUEST_LATENCY_S.labels(
        method = request.method,
        path = request.url.path,
    ).observe(duration_ms/1000)
    return response


# 2) Request schema (what the client must send)
class AskRequest(BaseModel):
    query: str


# 3) Endpoint: POST /ask
@app.post("/ask", response_model=TraceCall)
def ask(req: AskRequest, request: Request) -> TraceCall:
    # req.query is the text sent by the client
    logger = getattr(request.state, "logger", None)
    request_id = getattr(request.state, "request_id", None)
    logger.info(f"Received query:{req.query}")
    try:
        return agent_run(req.query, logger)
    except TimeoutError as e:
        # logger.error(f"Unhandled error {e}", exc_info =True)
        logger.exception("Unhandled error {e}")
        raise HTTPException(
            status_code=504,
            detail={
                "error_code": "LLM_TIMEOUT",
                "message": str(e),
                "request_id": request_id,
            },
        )
    except Exception as e:
        # logger.error(f"Unhandled error: {e}", exc_info = True)
        logger.exception(f"Unhandled error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "INTERNAL_ERROR",
                "message": "Unexpected server error",
                "request_id": request_id,
            },
        )

@app.post("/ingest")
def ingest_documents():
    try:
        stored = run_ingest()
        return {
            "status": "ok",
            "stored_chunks": stored
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/ready")
def ready():
    checks = {}
    
    if LLM_PROVIDER == "local":
        try:
            r = httpx.get("127.0.0.1:11434/api/tags", timeout = 1.5)
            checks["ollama"] = (r.status_code == 200)
        except Exception:
            checks["ollama"] = False
    
    elif LLM_PROVIDER == "vertex":
        checks["vertex_env"] = all([bool(GOOGLE_CLOUD_PROJECT),bool(VERTEX_LOCATION), bool(VERTEX_MODEL)])
    else: checks["llm_provider"] = False
    
    if RAG_ENABLED:
        from rag_store import similar
        try:
            _= similar("ping", m= 1, where = None)
            checks["rag"] = True
        except Exception:
            checks["rag"] = False
    else:
        checks["rag"] = "disabled"
        
    ok = all( v is True or v == "disabled" for v in checks.values())
    if not ok:
        raise HTTPException(status_code = 503,
                            detail = {"ready": False, "checks":checks})
    return {"ready": True, "checks":checks}
        
            


@app.get("/health")
def health():
    return {"ready": True, "rag_enabled": RAG_ENABLED}


@app.get("/config")
def config():
    return {
        "log level": LOG_LEVEL,
        "rag enabled": RAG_ENABLED,
        "rag top k": RAG_TOP_K,
        "model": MODEL,
        "max tries": MAX_TRIES,
    }


@app.get("/version")
def version():
    return {
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "git_sha": GIT_SHA,
        "build_time": BUILD_TIME,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(),
                    media_type = CONTENT_TYPE_LATEST)
    


@app.post("/ingest_event")
async def ingest_event(payload: dict):
    try:
        message = payload.get("message", {})
        data_b64 = message.get("data")

        if not data_b64:
            return {"status": "ignored", "reason": "no data"}

        decoded = base64.b64decode(data_b64).decode("utf-8")
        event = json.loads(decoded)

        bucket = event.get("bucket")
        name = event.get("name")

        if not bucket or not name:
            return {"status": "ignored", "reason": "missing bucket or name"}

        stored = run_ingest(bucket_name = bucket, file_name = name)

        return {
            "status": "ok",
            "bucket": bucket,
            "file": name,
            "stored_chunks": stored,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
