import os
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RAG_TOP_K = int(os.getenv("RAG_TOP_K","2"))
MODEL = os.getenv("MODEL", "qwen2.5:1.5b-instruct")
MAX_TRIES = int(os.getenv("MAX_TRIES", "2"))
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() =="true"
NUM_CHAR_OUT = int(os.getenv("NUM_CHAR_OUT","200"))
TEMPRATURE = float(os.getenv("TEMPRATURE", "0.0"))
CACHE_TTL_S = int(os.getenv("CACHE_TTL_S","300"))
API_VERSION = os.getenv("API_VERSION", "v2-api")
GIT_SHA = os.getenv("GIT_SHA", "unknown")          # filled later by CI/CD
BUILD_TIME = os.getenv("BUILD_TIME", "unknown")    # optional, filled later
SERVICE_NAME = os.getenv("SERVICE_NAME", "intelligent-reasoner")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").strip().lower()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "europe-west9")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", MODEL)