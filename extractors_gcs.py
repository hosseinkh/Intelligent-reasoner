from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List,Optional

from google.cloud import storage

from contracts import DocumentText
from extractors import extract_from_pdf, extract_from_docx


def extract_text_gcs(bucket_name: str, prefix: str = "", file_name:Optional[str]= None) -> List[DocumentText]:
    client = storage.Client()
    if file_name is None:
        blobs = client.list_blobs(bucket_name, prefix=prefix)
    else: 
        blob = client.bucket(bucket_name = bucket_name).blob(file_name)
        blobs =[blob]

    docs = []

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            file_name = Path(blob.name).name

            if file_name.startswith("~$"):
                continue

            suffix = Path(file_name).suffix.lower()

            if suffix not in [".pdf", ".docx"]:
                continue

            local_file = tmp_path / file_name
            blob.download_to_filename(str(local_file))

            if suffix == ".pdf":
                text = extract_from_pdf(local_file)
            elif suffix == ".docx":
                text = extract_from_docx(local_file)
            else:
                continue

            text = text.strip()
            if not text:
                continue

            docs.append(
                DocumentText(
                    file_name=file_name,
                    text=text
                )
            )

    return docs


