# Intelligent-Reasober

This project implements a minimal **Retrieval-Augmented Generation (RAG)** system.

It indexes PDF/DOCX documents into a vector store and answers questions using an LLM,
with **strict JSON validation** enforced by Pydantic.

The goal is to demonstrate a clean **end-to-end pipeline**:
indexing → retrieval → reasoning → validated output.

---

## Project Structure

The project exposes two explicit actions:

- **index**: prepare the knowledge base  
  (`index.py`)
- **ask**: answer a question using the indexed knowledge  
  (`ask.py`)

Core logic is implemented in reusable modules (`ingest`, `rag_store`, `reasoning`).

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux / macOS

pip install -r requirements.txt

Index Documents

Index documents from a folder (PDF / DOCX):

python index.py


Or specify a custom folder and logical source name:

python index.py --folder MedDoc --source MedDocs


This step must be run before asking questions.

Ask a Question

Ask a question using the indexed data:

python ask.py -q "What is the reason for medication shortage?"


Optional parameters:

python ask.py -q "..." --k 5

Output Contract

The system always returns a validated JSON object with the following schema:

{
  "cause": "Manufacturing | DemandSpike | Quality | Logistics | Unknown",
  "confidence": 0.0,
  "source": "file name"
}


The LLM is forced to return only JSON

Output is validated using Pydantic

If the answer cannot be inferred, "Unknown" is returned

Design Principles

Explicit separation between index and ask

No hidden ingestion during querying

Strict validation of LLM outputs

Minimal, inspectable, reproducible pipeline

Limitations

Answers are restricted to indexed documents

No guarantee of correctness beyond provided context

Designed as a minimal, educational RAG system

Notes

Document folders (e.g. MedDoc/) are intentionally not versioned.
Provide your own documents when running the pipeline.


---

### What this README does well
- Explains **what the product does**
- Explains **how to run it**
- Shows **clear contracts**
- Avoids overclaiming
- Matches your actual code

This is **exactly** what a reviewer expects.

⏸️  
If you want, next we can:
- review `.gitignore` line by line, or  
- do the **final commit + version tag**, or  
- discuss **how to present this repo in interviews**.
