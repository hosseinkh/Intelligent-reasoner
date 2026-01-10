from rag_store import init, upsert_text, count_embeddings, similar

def test_chroma_upsert_similar(tmp_path):

    init(str(tmp_path), "rag_test")

    before = count_embeddings()

    upsert_text("Aspirin is used for pain.", {"file_name": "a.txt", "source": "test"}, record_id="a::1")
    upsert_text("Paracetamol reduces fever.", {"file_name": "b.txt", "source": "test"}, record_id="b::1")
    upsert_text("Ibuprofen is an NSAID.", {"file_name": "c.txt", "source": "test"}, record_id="c::1")

    after = count_embeddings()
    assert after == before + 3

    hits = similar("what is used for fever?" , k = 3)
    assert len(hits) > 1
    assert hits[0].score is not None
    assert hits[0].document is not None



