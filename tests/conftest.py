import sys
from pathlib import Path

# Add the project root (modular_new/) to sys.path so tests can import modules like ingest.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
