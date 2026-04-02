from api.config import RAG_BACKEND


def upsert_text(text, metadata, record_id=None):
    if RAG_BACKEND == "cloudsql":
        from rag_store_cloudsql import upsert_text as impl
    else:
        from rag_store import upsert_text as impl
    return impl(text, metadata, record_id)


def similar(text, k, where=None):
    if RAG_BACKEND == "cloudsql":
        from rag_store_cloudsql import similar as impl
    else:
        from rag_store import similar as impl
    return impl(text, k, where)


def count_embeddings():
    if RAG_BACKEND == "cloudsql":
        from rag_store_cloudsql import count_embeddings as impl
    else:
        from rag_store import count_embeddings as impl
    return impl()


def restore_store():
    if RAG_BACKEND == "cloudsql":
        from rag_store_cloudsql import restore_store as impl
    else:
        from rag_store import restore_store as impl
    return impl()