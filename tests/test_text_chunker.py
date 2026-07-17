import pytest

from src.ingestion.document_loader import Document
from src.ingestion.text_chunker import chunk_text, chunk_documents


def make_doc(content: str) -> Document:
    return Document(content=content, metadata={"source": "test.txt", "page": "1"})


def test_chunk_count_matches_spec():
    """1000 chars, chunk_size=500, overlap=50 -> exactly 3 chunks."""
    text = "a" * 1000
    doc = make_doc(text)
    chunks = chunk_text(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 3


def test_overlap_shares_exact_substring():
    text = "".join(str(i % 10) for i in range(1000))
    doc = make_doc(text)
    chunks = chunk_text(doc, chunk_size=500, chunk_overlap=50)

    # The last 50 chars of chunk N should equal the first 50 chars of chunk N+1
    for i in range(len(chunks) - 1):
        end_of_current = chunks[i].content[-50:]
        start_of_next = chunks[i + 1].content[:50]
        assert end_of_current == start_of_next


def test_metadata_preserved_and_chunk_index_added():
    doc = make_doc("x" * 1200)
    chunks = chunk_text(doc, chunk_size=500, chunk_overlap=50)
    for idx, c in enumerate(chunks):
        assert c.metadata["source"] == "test.txt"
        assert c.metadata["page"] == "1"
        assert c.metadata["chunk_index"] == str(idx)


def test_empty_document_returns_no_chunks():
    doc = make_doc("")
    assert chunk_text(doc, chunk_size=500, chunk_overlap=50) == []


def test_short_document_returns_single_chunk():
    doc = make_doc("short text")
    chunks = chunk_text(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].content == "short text"


def test_invalid_overlap_raises():
    doc = make_doc("x" * 100)
    with pytest.raises(ValueError):
        chunk_text(doc, chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        chunk_text(doc, chunk_size=100, chunk_overlap=150)


def test_invalid_chunk_size_raises():
    doc = make_doc("x" * 100)
    with pytest.raises(ValueError):
        chunk_text(doc, chunk_size=0, chunk_overlap=0)


def test_chunk_documents_aggregates_across_multiple_docs():
    docs = [make_doc("a" * 600), make_doc("b" * 600)]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    # Each 600-char doc -> 2 chunks -> 4 total
    assert len(chunks) == 4
