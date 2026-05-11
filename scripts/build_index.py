#!/usr/bin/env python3
# scripts/build_index.py — Embeds catalog.json and builds FAISS index.
#
# Run ONCE offline AFTER scrape.py: python scripts/build_index.py
# Output: data/faiss_index.bin
#
# What this does:
#   1. Load catalog.json
#   2. Convert each assessment to a text description for embedding
#   3. Embed with all-MiniLM-L6-v2 (384-dim, free, local, fast)
#   4. Build a FAISS IndexFlatIP (exact search via inner product = cosine on normalized vecs)
#   5. Save index to disk

import json
import numpy as np
import faiss
from pathlib import Path
from fastembed import TextEmbedding

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH   = DATA_DIR / "faiss_index.bin"


def assessment_to_text(a: dict) -> str:
    """
    Convert one catalog entry to a single string for embedding.
    The richer this text, the better the semantic search quality.
    """
    parts = [a.get("name", "")]

    if a.get("description"):
        parts.append(a["description"])

    if a.get("test_type"):
        type_label = {
            "A": "ability aptitude cognitive",
            "P": "personality behavioural",
            "K": "knowledge skills technical",
            "S": "situational judgment",
            "B": "biodata",
            "M": "motivation",
        }.get(a["test_type"], "")
        parts.append(f"Test type: {type_label}")

    if a.get("competencies"):
        parts.append("Competencies: " + ", ".join(a["competencies"]))

    return ". ".join(filter(None, parts))


def main():
    print("Loading catalog...")
    with open(CATALOG_PATH) as f:
        catalog = json.load(f)
    print(f"  {len(catalog)} assessments loaded.")

    print("Building text representations...")
    texts = [assessment_to_text(a) for a in catalog]

    print("Loading embedding model (all-MiniLM-L6-v2 via fastembed)...")
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding all assessments (this may take 1-2 minutes)...")
    # fastembed returns a generator of numpy arrays
    embeddings = np.array(list(model.embed(texts, batch_size=32)), dtype=np.float32)
    
    # Normalize to unit vectors so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    print(f"  Embeddings shape: {embeddings.shape}")

    print("Building FAISS index (IndexFlatIP — exact, no approximation)...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors.")

    print(f"Saving index to {INDEX_PATH}...")
    faiss.write_index(index, str(INDEX_PATH))

    print("Done! faiss_index.bin is ready.")
    print("\nVerification — top 3 results for 'Java developer knowledge test':")
    query_vec = np.array(list(model.embed(["Java developer knowledge test"])), dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    scores, indices = index.search(query_vec, 3)
    for score, idx in zip(scores[0], indices[0]):
        print(f"  [{score:.3f}] {catalog[idx]['name']}")


if __name__ == "__main__":
    main()
