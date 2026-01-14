from ingest import ingest_into_rag
from rag_store import count_embeddings
import argparse

def main():
    parser = argparse.ArgumentParser(description = "From which folder enrich the Rag, and how to call the source data?")
    parser.add_argument("--folder","-f", default = "MedDoc" ,help = "What is the name of folder to read filed from it?")
    parser.add_argument('--source','-s', default = "MedDocs",help = "What is the logical name for this source data?")
    args = parser.parse_args()

    ingest_into_rag(folder = args.folder, source = args.source)
    count = count_embeddings()
    print(f"Indexing done. Total number is {count}")

if __name__ == "__main__":
    main()
