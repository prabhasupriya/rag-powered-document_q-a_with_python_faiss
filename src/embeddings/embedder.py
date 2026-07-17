"""
embedder.py

Wraps a sentence-transformers model to convert text chunks (and later,
user queries) into dense vector embeddings. Using the SAME model instance
for both indexing and querying is critical -- mixing embedding models
puts vectors in different mathematical spaces and silently breaks search.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class TextEmbedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        logger.info("Loading embedding model '%s' (first run downloads + caches it)...", model_name)
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts.

        Returns:
            np.ndarray of shape (len(texts), embedding_dimension), dtype float32.
        """
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype="float32")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so L2 distance behaves like cosine
        )
        return embeddings.astype("float32")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Convenience wrapper for embedding a single string (e.g. a query)."""
        return self.generate_embeddings([text])[0]
