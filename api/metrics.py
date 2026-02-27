from prometheus_client import Counter, Histogram

HTTP_REQUEST_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_REQUEST_LATENCY_S = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method","path"]
)

RAG_SEARCH_TOTAL = Counter(
    "rag_search_total",
    "Total Rag search executed"
)

LLM_CALLS_TOTAL = Counter(
    "llm_calls_total",
    "Total LLM calls executed",
    ["provider"],
)

LLM_VALIDATION_FAIL_TOTAL = Counter(
    "llm_validation_fail_total",
    "Total LLM validation failures",
    ["reason"],
)

LLM_TOKEN_TOTAL = Counter(
    "llm_token_total",
    "Total LLM tokens used",
    ["provider", "type"],
)

LLM_LATENCY_SECONDS = Histogram(
    "llm_latency_seconds",
    "LLM Response Latency",
    ["provider"],
    buckets=(0.1,0.25,0.5,1,2,5,10,20)
)

RAG_SEARCH_LATENCY_S = Histogram(
    "rag_search_latency",
    "rag search latency",
    buckets=(0.1,0.25,0.5,1,2,5,10,20)
)

PROMPT_MAKING_LATENCY_S = Histogram(
    "prompt_making_latency",
    "prompt making latency",
    buckets=(0.1,0.25,0.5,1,2,5,10,20)
)