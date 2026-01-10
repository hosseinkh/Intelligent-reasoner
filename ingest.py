
from extractors import extract_text
from text_ops import text_to_sentence, chunk_sentences ,make_chunks
from rag_store import upsert_text

def ingest_into_rag():

    documents_text = extract_text()
    for doc in documents_text:
        documents_text_to_sentence = text_to_sentence(doc)
        if not documents_text_to_sentence :
            continue
        documents_chunkings = chunk_sentences(documents_text_to_sentence,list(documents_text_to_sentence.keys())[0])
        final_chunks = make_chunks(documents_chunkings, source = "MedDocs")

        for chunk in final_chunks:
            text = chunk.text
            rid = chunk.chunk_id
            metadata = {"file_name": chunk.file_name , "source":chunk.source}
            record_id = upsert_text(text, metadata,rid)
            print(f"the chunk with id {rid} for filename{chunk.file_name} is stored in RAG Memory. The record id is {record_id}.")

def main():
    ingest_into_rag()

if __name__ == "__main__":
    main()
        

        




