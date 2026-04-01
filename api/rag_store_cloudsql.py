import os
import json
import uuid
from typing import Dict, List, Optional, Any

import psycopg2
from chromadb.utils import embedding_functions

from contracts import Hit
from api.config import RAG_TOP_K

_conn = None
_embed_function = embedding_functions.DefaultEmbeddingFunction()


def init():
    global _conn
    if _conn is not None:
        return

    _conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432"),
    )
    _conn.autocommit = True


def restore_store():
    init()
    with _conn.cursor() as cur:
        cur.execute("DELETE FROM documents;")


def count_embeddings():
    init()
    with _conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents;")
        return cur.fetchone()[0]


def embed(text: str):
    return _embed_function([text])


def upsert_text(text: str, metadata: Dict[str, Any], record_id: Optional[str] = None):
    init()
    text_embed = embed(text)[0]
    rec_id = record_id or str(uuid.uuid4())

    with _conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (id, content, embedding, metadata)
            VALUES (%s, %s, %s::vector, %s::jsonb)
            ON CONFLICT (id)
            DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata;
            """,
            (
                rec_id,
                text,
                str(list(text_embed)),
                json.dumps(metadata),
            ),
        )

    return rec_id


def similar(text: str, k: int = RAG_TOP_K, where: Optional[Dict[str, Any]] = None):
    init()
    query_embed = embed(text)[0]

    sql = """
        SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    with _conn.cursor() as cur:
        cur.execute(sql, (str(list(query_embed)), str(list(query_embed)), k))
        rows = cur.fetchall()

    hits = []
    for row in rows:
        hits.append(
            Hit(
                id=row[0],
                document=row[1],
                metadata=row[2],
                score=float(row[3]),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits
