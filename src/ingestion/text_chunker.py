"""
text_chunker.py

Splits long document text into overlapping, fixed-size character chunks.
Overlap preserves context across chunk boundaries so a sentence sliced
in half can still be matched semantically from either neighboring chunk.
"""

from __future__ import annotations

from typing import List

from src.ingestion.document_loader import Document


def chunk_text(document: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Slice a single Document's content into overlapping chunks.

    Args:
        document: The source Document (already-extracted, cleaned text).
        chunk_size: Max number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of new Document objects, each carrying the original metadata
        plus a `chunk_index` field, so citations can always be traced back
        to the source file (and page, for PDFs).

    Raises:
        ValueError: if chunk_overlap >= chunk_size (would infinite-loop or
            produce non-advancing chunks).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = document.content
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: List[Document] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_str = text[start:end]

        if chunk_str.strip():
            new_metadata = dict(document.metadata)
            new_metadata["chunk_index"] = str(chunk_index)
            chunks.append(Document(content=chunk_str, metadata=new_metadata))
            chunk_index += 1

        if end == len(text):
            break
        start += step

    return chunks


def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Apply chunk_text across a list of Documents (e.g. many PDF pages)."""
    all_chunks: List[Document] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
