from ingest import ingest_into_rag
from rag_store import count_embeddings, similar
def baseline():

    ingest_into_rag()
    number_of_embeddings = count_embeddings()
    query = "What is the main cause of medication shortage?"
    hits = similar(query, k = 3)
    if not hits:
        print("No similarity found between the context and the query")
        return 

    print(f"Here is the most similar text from hits: {hits[0].document}, and the similarity score is equal to {hits[0].score}" )

def main():
    baseline()

if __name__ == "__main__":
    main()        