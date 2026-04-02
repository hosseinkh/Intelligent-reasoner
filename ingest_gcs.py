from extractors_gcs import extract_text_gcs
from text_ops import text_to_sentence, chunk_sentences, make_chunks
from rag_store_selector import upsert_text
from api.config import GCS_BUCKET_NAME, GCS_PREFIX


def ingest_into_rag_gcs(bucket_name: str, prefix: str, source: str):
    documents_text = extract_text_gcs(bucket_name, prefix)

    stored = 0

    for doc in documents_text:
        documents_text_to_sentence = text_to_sentence(doc)

        if not documents_text_to_sentence:
            continue

        documents_chunkings = chunk_sentences(
            documents_text_to_sentence,
            list(documents_text_to_sentence.keys())[0]
        )

        final_chunks = make_chunks(documents_chunkings, source=source)

        for chunk in final_chunks:
            text = chunk.text
            rid = chunk.chunk_id
            metadata = {
                "file_name": chunk.file_name,
                "source": chunk.source
            }
            record_id = upsert_text(text, metadata, rid)
            stored += 1
            print(
                f"the chunk with id {rid} for filename {chunk.file_name} "
                f"is stored in RAG Memory. The record id is {record_id}."
            )

    return stored


def main():
    ingest_into_rag_gcs(
        bucket_name=GCS_BUCKET_NAME,
        prefix=GCS_PREFIX,
        source="GCSDocs"
    )


if __name__ == "__main__":
    main()