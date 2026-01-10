import chromadb
from chromadb.utils import embedding_functions
import os
from typing import Dict, List, Optional,Any
import uuid
from contracts import Hit

_client = None
_collection = None

base_dir = os.path.dirname(os.path.abspath(__file__))
_persist_path = os.path.join(base_dir, "RAG_Memeory")

_embed_function = embedding_functions.DefaultEmbeddingFunction()

def init(persist_path : Optional[str] = None , collection_name:Optional[str] = None):
    global _client, _collection
    p_path = persist_path or _persist_path
    c_name = collection_name or "rag_memory"
    if _client is not None:
        return

    _client = chromadb.PersistentClient(
        path = p_path
    )

    _collection = _client.get_or_create_collection(
        name = c_name,
        metadata = {"hnsw:space": "cosine"}
    )

def count_embeddings():
    init()
    return _collection.count()

def embed(text: str):
    text = [text]

    return _embed_function(text)


def upsert_text(text : str , metadata: Dict[str,Any], record_id: Optional[str] = None):
    init()
    text_embed = embed(text)[0]

    id = record_id or str(uuid.uuid4())

    _collection.upsert(
        ids = [id],
        documents = [text],
        embeddings = [text_embed],
        metadatas = [metadata]
    )

    return id

def similar( text: str, k : int = 3, where: Optional[Dict[str,Any]]= None):
    init()
    embed_query = embed(text)[0]

    results = _collection.query(
        query_embeddings = [embed_query],
        n_results = k,
        where = where,
        include = ['documents','distances','metadatas']
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents",[[]])[0]
    distances= results.get("distances",[[]])[0]
    metadatas = results.get("metadatas",[[]])[0]
    hits = []

    for i in range(len(ids)):
        similarity = 1 - distances[i]
        hits.append(Hit(
            
                id= ids[i],
                document = documents[i],
                score = similarity,
                metadata =  metadatas[i]
        )
        )

    hits.sort(key = lambda h: h.score, reverse = True)

    return hits



