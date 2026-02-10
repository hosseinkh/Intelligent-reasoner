$env:LOG_LEVEL = "INFO"
$env:RAG_ENABLED = "true"
$env:RAG_TOP_K = "3"
$env:MODEL = "qwen2.5:1.5b-instruct"
$env:MAX_TRIES =  "2"

uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload