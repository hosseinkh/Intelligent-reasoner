from api.config import DOC_SOURCE, GCS_BUCKET_NAME, GCS_PREFIX


def run_ingest():
    if DOC_SOURCE == "gcs":
        from ingest_gcs import ingest_into_rag_gcs
        return ingest_into_rag_gcs(
            bucket_name=GCS_BUCKET_NAME,
            prefix=GCS_PREFIX,
            source="GCSDocs"
        )
    else:
        from ingest import ingest_into_rag
        return ingest_into_rag(
            folder="MedDoc",
            source="MedDocs"
        )


def main():
    stored = run_ingest()
    print(f"stored {stored} chunks.")


if __name__ == "__main__":
    main()