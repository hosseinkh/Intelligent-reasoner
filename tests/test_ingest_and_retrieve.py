from ingest import ingest_into_rag
from rag_store import similar

def test_ingest_into_rag(folder = "test_folder", source = "tests"):
    stored = ingest_into_rag(folder, source)
    assert stored > 0

def test_similar():
    question = "What is the reason of shortage?"
    hits = similar(question, k =3)
    assert len(hits) > 0



