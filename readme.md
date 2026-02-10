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

*Intelligent Reasoner – V2 (Agent Architecture)*

This project implements a traceable agentic reasoning system exposed via FastAPI, with optional RAG, structured logging, configuration via environment variables, and a smoke test for deployment validation.

Features

Agent-based architecture with routing (llm_answer vs rag_search)

Full execution trace (TraceCall, ToolCall)

Optional RAG controlled by environment variables

FastAPI REST API

Structured logging with request ID

Timeouts and retries for LLM calls

Health / readiness / config endpoints

Smoke test script for deployment verification



Project Structure
.
├── agent/
│   └── runner.py
├── api/
│   ├── app.py
│   ├── config.py
│   └── logging_config.py
├── reasoning.py
├── contracts.py
├── scripts/
│   └── smoke.py
├── tests/
├── requirements.txt
└── README.md


Installation
Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

Install dependencies:
pip install -r requirements.txt


Configuration (Environment Variables)
All variables have safe defaults.
export MODEL_NAME=qwen2.5
export TOP_K=3
export MAX_RETRIES=2
export LLM_TIMEOUT_S=5
export RAG_ENABLED=true
export LOG_LEVEL=INFO

(Windows PowerShell)
setx MODEL_NAME qwen2.5
setx TOP_K 3
setx MAX_RETRIES 2
setx LLM_TIMEOUT_S 5
setx RAG_ENABLED true
setx LOG_LEVEL INFO


Run the API
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

API documentation:


Swagger UI: http://127.0.0.1:8000/docs



API Endpoints
Health
GET /health

Ready (environment + feature flags)
GET /ready

Config (runtime configuration)
GET /config

Ask (main reasoning endpoint)
POST /ask
{
  "query": "Pourquoi un médicament est en pénurie ?"
}

Response: TraceCall (full execution trace)

Logging


Structured logs with timestamps, level, logger name, request_id


Request ID injected per API call


Safe logging for background threads (filters applied)


Example log:
2026-01-27 14:02:40 | INFO | intelligent-reasoner | <request_id> | agent_run started


Smoke Test (Deployment Validation)
Run after the API is up:
python scripts/smoke.py

Checks:


/health


/ready


/ask


This is the minimum validation required for cloud deployment.

Testing
Run unit tests:
pytest


Versioning
This version is frozen and tagged.


Deterministic agent flow


Configurable routing


Production-ready API surface

