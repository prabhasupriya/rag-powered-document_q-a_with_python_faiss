import os
import tempfile

from src.ingestion.document_loader import DirectoryProcessor, TextLoader, _clean_text


def test_text_loader_reads_utf8_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Hello world.\n\n\nThis has extra newlines.")

        docs = TextLoader().load(path)
        assert len(docs) == 1
        assert "Hello world." in docs[0].content
        assert docs[0].metadata["source"] == "sample.txt"


def test_text_loader_missing_file_returns_empty_list():
    docs = TextLoader().load("/nonexistent/path/file.txt")
    assert docs == []


def test_clean_text_collapses_whitespace():
    dirty = "Hello    world\n\n\n\nFoo   Bar"
    cleaned = _clean_text(dirty)
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned


def test_directory_processor_routes_txt_files():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "a.txt"), "w") as f:
            f.write("Document A content about apples.")
        with open(os.path.join(tmp, "b.txt"), "w") as f:
            f.write("Document B content about bananas.")
        # Unsupported extension should be skipped, not crash the run.
        with open(os.path.join(tmp, "c.docx"), "w") as f:
            f.write("should be ignored")

        docs = DirectoryProcessor().process(tmp)
        sources = sorted(d.metadata["source"] for d in docs)
        assert sources == ["a.txt", "b.txt"]


def test_directory_processor_raises_on_invalid_dir():
    import pytest

    with pytest.raises(NotADirectoryError):
        DirectoryProcessor().process("/definitely/not/a/real/directory")
