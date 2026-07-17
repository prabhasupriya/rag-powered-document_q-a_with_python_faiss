"""
document_loader.py

Handles extraction of raw text from source documents (PDF and TXT) and
routes an entire directory of mixed files through the correct loader.

Design notes:
- Every loader returns a list of `Document` objects. A PDF produces one
  `Document` per page (so page-level citations are possible); a TXT file
  produces a single `Document` for the whole file.
- Text is normalized (collapsed whitespace, stripped control characters)
  so that downstream chunking and embedding don't choke on PDF extraction
  artifacts (form feeds, repeated newlines, stray unicode).
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List

import pdfplumber

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# Guardrails from the task spec: don't let a single pathological directory
# blow up memory / runtime.
MAX_PAGES = 500
MAX_CHARACTERS = 1_000_000


@dataclass
class Document:
    """A unit of extracted text plus provenance metadata."""

    content: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        preview = self.content[:60].replace("\n", " ")
        return f"Document(source={self.metadata.get('source')}, page={self.metadata.get('page')}, text='{preview}...')"


def _clean_text(text: str) -> str:
    """Normalize extracted text: fix unicode, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Collapse runs of whitespace/newlines/form-feeds into single spaces,
    # but keep paragraph breaks somewhat legible.
    text = text.replace("\x0c", "\n")  # form feed -> newline
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


class PDFLoader:
    """Extracts text from a PDF, one Document per page."""

    def load(self, filepath: str) -> List[Document]:
        documents: List[Document] = []
        filename = os.path.basename(filepath)

        try:
            with pdfplumber.open(filepath) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    if page_number > MAX_PAGES:
                        logger.warning(
                            "Reached MAX_PAGES=%s while reading %s; truncating.",
                            MAX_PAGES,
                            filename,
                        )
                        break
                    try:
                        raw_text = page.extract_text() or ""
                    except Exception as exc:  # noqa: BLE001 - keep ingesting
                        logger.error(
                            "Failed to extract text from %s page %s: %s",
                            filename,
                            page_number,
                            exc,
                        )
                        continue

                    cleaned = _clean_text(raw_text)
                    if not cleaned:
                        # Likely a scanned/image-only page. Skip gracefully.
                        logger.info(
                            "No extractable text on %s page %s (possibly scanned image).",
                            filename,
                            page_number,
                        )
                        continue

                    documents.append(
                        Document(
                            content=cleaned,
                            metadata={"source": filename, "page": str(page_number)},
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not open PDF %s: %s", filepath, exc)

        return documents


class TextLoader:
    """Extracts text from a plain UTF-8 .txt file."""

    def load(self, filepath: str) -> List[Document]:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not read text file %s: %s", filepath, exc)
            return []

        cleaned = _clean_text(raw_text)
        if not cleaned:
            return []

        return [Document(content=cleaned, metadata={"source": filename, "page": "1"})]


class DirectoryProcessor:
    """Walks a directory, routes files to the correct loader, aggregates output."""

    def __init__(self) -> None:
        self.pdf_loader = PDFLoader()
        self.text_loader = TextLoader()

    def process(self, directory: str) -> List[Document]:
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"'{directory}' is not a valid directory")

        all_documents: List[Document] = []
        total_chars = 0

        for root, _dirs, files in os.walk(directory):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                filepath = os.path.join(root, filename)
                if ext == ".pdf":
                    docs = self.pdf_loader.load(filepath)
                else:
                    docs = self.text_loader.load(filepath)

                for doc in docs:
                    total_chars += len(doc.content)
                    if total_chars > MAX_CHARACTERS:
                        logger.warning(
                            "Reached MAX_CHARACTERS=%s; stopping ingestion early.",
                            MAX_CHARACTERS,
                        )
                        return all_documents
                    all_documents.append(doc)

                logger.info("Loaded %s document segment(s) from %s", len(docs), filename)

        return all_documents
