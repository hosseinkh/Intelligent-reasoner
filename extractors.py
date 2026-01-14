from pathlib import Path
from docx import Document
from pypdf import PdfReader
from typing import Dict,List
from contracts import DocumentText

def extract_text(folder: str = "MedDoc")-> List[DocumentText]:
    base = Path(folder)

    if not base.exists():
        print(f"folder not found {base.resolve()}")
        return []

    docs = []
    print(f"scanning the folder {base.resolve()}")

    for path in base.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue

        
        suffix = path.suffix.lower()

        if suffix == ".docx":
            text = extract_from_docx(path)

        elif suffix == ".pdf":
            text = extract_from_pdf(path)
        else:
            continue    

        text = text.strip()
        if not text:
            continue

        docs.append(DocumentText(
            file_name = path.name,
            text = text
        ))
    return docs





def extract_from_pdf(path : Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            parts.append(text)
    
    return "\n".join(parts)



def extract_from_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
           parts.append(text)

    return "\n".join(parts)

