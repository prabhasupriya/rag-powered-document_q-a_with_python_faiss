import tempfile

import numpy as np

from src.ingestion.document_loader import Document
from src.storage.vector_store import VectorStore


def make_doc(text, source="doc.txt"):
    return Document(content=text, metadata={"source": source, "page": "1"})


def test_add_and_search_returns_nearest_vector():
    store = VectorStore(embedding_dimension=4)
    docs = [make_doc("alpha"), make_doc("beta"), make_doc("gamma")]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    store.add_embeddings(embeddings, docs)

    # Query close to "beta"'s vector should return beta first.
    query = np.array([0.0, 0.9, 0.1, 0.0], dtype="float32")
    results = store.search(query, k=1)

    assert len(results) == 1
    assert results[0]["content"] == "beta"


def test_search_on_empty_index_returns_empty_list():
    store = VectorStore(embedding_dimension=4)
    query = np.zeros(4, dtype="float32")
    assert store.search(query, k=3) == []


def test_mismatched_lengths_raise():
    import pytest

    store = VectorStore(embedding_dimension=4)
    embeddings = np.zeros((2, 4), dtype="float32")
    docs = [make_doc("only one doc")]
    with pytest.raises(ValueError):
        store.add_embeddings(embeddings, docs)


def test_save_and_load_round_trip():
    store = VectorStore(embedding_dimension=3)
    docs = [make_doc("hello", source="a.txt"), make_doc("world", source="b.txt")]
    embeddings = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
    store.add_embeddings(embeddings, docs)

    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp)
        loaded = VectorStore.load(tmp)

        assert loaded.index.ntotal == 2
        results = loaded.search(np.array([1, 0, 0], dtype="float32"), k=1)
        assert results[0]["content"] == "hello"
        assert results[0]["metadata"]["source"] == "a.txt"
