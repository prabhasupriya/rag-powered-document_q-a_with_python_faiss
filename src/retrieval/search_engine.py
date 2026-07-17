"""
search_engine.py

Orchestrates the query-time retrieval step: embed the user's question with
the SAME embedder used at indexing time, then search the vector store for
the top-k most semantically similar chunks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.embeddings.embedder import TextEmbedder
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, embedder: TextEmbedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Embed `query` and return the top-k matching chunks with metadata."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        query_embedding = self.embedder.generate_embedding(query)
        results = self.vector_store.search(query_embedding, k=k)
        logger.info("Retrieved %s chunk(s) for query: %r", len(results), query)
        return results
