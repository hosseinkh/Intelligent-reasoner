from api.config import RAG_BACKEND

if RAG_BACKEND == "cloudsql":
    from rag_store_cloudsql import *
else:
    from rag_store import *