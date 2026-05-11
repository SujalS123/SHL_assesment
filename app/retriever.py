# retriever.py — Loads the FAISS index once at startup, searches at query time.
#
# Two-phase RAG:
#   - Index built OFFLINE by scripts/build_index.py (run once)
#   - Search runs ONLINE for every relevant chat turn
#
# We use cosine similarity via inner product on normalized vectors.
# all-MiniLM-L6-v2 produces 384-dim vectors — fast, free, good enough for ~200 items.

import json
import numpy as np
import faiss
from pathlib import Path
from fastembed import TextEmbedding

# Paths — resolved relative to this file so they work from any working directory
DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH   = DATA_DIR / "faiss_index.bin"

# Load once at module import time (FastAPI startup)
print("Loading embedding model...")
_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

print("Loading catalog...")
with open(CATALOG_PATH) as f:
    _catalog: list[dict] = json.load(f)

print("Loading FAISS index...")
_index = faiss.read_index(str(INDEX_PATH))

print(f"Retriever ready — {len(_catalog)} assessments indexed.")


def retrieve(query: str, k: int = 10) -> list[dict]:
    """
    Embed the query and return the top-k most similar catalog entries.

    Args:
        query: Natural language description of the hiring need.
        k:     How many results to return. Capped at catalog size.

    Returns:
        List of catalog dicts (name, url, test_type, description, competencies).
    """
    k = min(k, len(_catalog))

    # Embed and normalize to unit length so inner product == cosine similarity
    query_vec = np.array(list(_model.embed([query])), dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    # FAISS returns (scores, indices) — we only need indices
    _, indices = _index.search(query_vec, k)

    return [_catalog[i] for i in indices[0] if i != -1]
