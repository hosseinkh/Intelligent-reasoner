import os
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RAG_TOP_K = int(os.getenv("RAG_TOP_K","3"))
MODEL = os.getenv("MODEL", "qwen2.5:1.5b-instruct")
MAX_TRIES = int(os.getenv("MAX_TRIES", "2"))
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() =="true"

API_VERSION = os.getenv("API_VERSION", "v2-api")
GIT_SHA = os.getenv("GIT_SHA", "unknown")          # filled later by CI/CD
BUILD_TIME = os.getenv("BUILD_TIME", "unknown")    # optional, filled later
SERVICE_NAME = os.getenv("SERVICE_NAME", "intelligent-reasoner")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))