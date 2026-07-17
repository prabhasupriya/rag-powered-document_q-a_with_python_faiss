"""
vector_store.py

Wraps a FAISS IndexFlatL2 index plus a parallel `chunk_map` dictionary that
translates FAISS integer IDs back into the original text + metadata (FAISS
itself only stores raw float vectors, never the text). Also handles saving
the index and chunk_map to disk so the CLI's `index` and `ask` commands can
run as two separate processes.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector index with an in-memory/disk-persisted document store."""

    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = embedding_dimension
        # IndexFlatL2: exact (brute-force) nearest neighbor search using
        # Euclidean distance. Fine for < ~100k chunks, which covers this
        # assignment's scale. Because embeddings are normalized, L2 distance
        # ranks results identically to cosine similarity.
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.chunk_map: Dict[int, Dict[str, Any]] = {}
        self.current_id = 0

    def add_embeddings(self, embeddings: np.ndarray, documents: List[Any]) -> None:
        """Add a batch of (embedding, Document) pairs to the index."""
        if len(embeddings) != len(documents):
            raise ValueError("embeddings and documents must be the same length")
        if len(embeddings) == 0:
            return

        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)

        for doc in documents:
            self.chunk_map[self.current_id] = {
                "content": doc.content,
                "metadata": doc.metadata,
            }
            self.current_id += 1

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Return the top-k chunks (text + metadata + distance) for a query vector."""
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        query_vector = np.asarray([query_embedding], dtype="float32")
        distances, ids = self.index.search(query_vector, k)

        results = []
        for rank, (dist, faiss_id) in enumerate(zip(distances[0], ids[0])):
            if faiss_id == -1:
                continue
            entry = self.chunk_map.get(int(faiss_id))
            if entry is None:
                continue
            results.append(
                {
                    "rank": rank + 1,
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                    "distance": float(dist),
                }
            )
        return results

    def save(self, directory: str) -> None:
        """Persist the FAISS index and chunk_map to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "chunk_map.pkl"), "wb") as f:
            pickle.dump(self.chunk_map, f)

        meta = {
            "embedding_dimension": self.embedding_dimension,
            "current_id": self.current_id,
            "total_vectors": self.index.ntotal,
        }
        with open(os.path.join(directory, "store_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved vector store (%s vectors) to %s", self.index.ntotal, directory)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load a previously saved FAISS index + chunk_map from disk."""
        meta_path = os.path.join(directory, "store_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No vector store found at '{directory}'. Did you run the `index` command first?"
            )

        with open(meta_path) as f:
            meta = json.load(f)

        store = cls(embedding_dimension=meta["embedding_dimension"])
        store.index = faiss.read_index(os.path.join(directory, "index.faiss"))

        with open(os.path.join(directory, "chunk_map.pkl"), "rb") as f:
            store.chunk_map = pickle.load(f)

        store.current_id = meta["current_id"]
        return store
