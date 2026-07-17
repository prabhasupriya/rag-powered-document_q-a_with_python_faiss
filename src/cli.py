"""
cli.py

Command-line entrypoint for the RAG system.

Usage:
    python -m src.cli index --path ./data
    python -m src.cli ask "What is the refund policy?"
    python -m src.cli ask "What is the refund policy?" --k 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

from src.embeddings.embedder import DEFAULT_MODEL_NAME, TextEmbedder
from src.generation.llm_client import LLMGenerator
from src.ingestion.document_loader import DirectoryProcessor
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.search_engine import SearchEngine
from src.storage.vector_store import VectorStore

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag.cli")

DEFAULT_INDEX_DIR = os.getenv("INDEX_DIR", "./index_store")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))


def cmd_index(args: argparse.Namespace) -> None:
    """Run the full indexing pipeline: load -> chunk -> embed -> store."""
    start = time.time()
    print(f"\n📂 Indexing documents from: {args.path}")

    processor = DirectoryProcessor()
    documents = processor.process(args.path)
    if not documents:
        print("⚠️  No supported documents (.pdf, .txt) found. Nothing to index.")
        sys.exit(1)
    print(f"   Loaded {len(documents)} page/file segment(s).")

    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"   Split into {len(chunks)} chunk(s) (size={args.chunk_size}, overlap={args.chunk_overlap}).")
    if not chunks:
        print("⚠️  No chunks produced. Aborting.")
        sys.exit(1)

    print(f"   Generating embeddings with '{args.model}' ...")
    embedder = TextEmbedder(model_name=args.model)
    texts = [c.content for c in chunks]
    embeddings = embedder.generate_embeddings(texts)

    store = VectorStore(embedding_dimension=embedder.embedding_dimension)
    store.add_embeddings(embeddings, chunks)
    store.save(args.output)

    elapsed = time.time() - start
    print(f"\n✅ Indexed {len(chunks)} chunks from {args.path} into '{args.output}' in {elapsed:.1f}s")


def cmd_ask(args: argparse.Namespace) -> None:
    """Run the retrieval + generation pipeline for a single question."""
    if not os.path.exists(args.index_dir):
        print(f"⚠️  No index found at '{args.index_dir}'. Run `index --path <dir>` first.")
        sys.exit(1)

    embedder = TextEmbedder(model_name=args.model)
    store = VectorStore.load(args.index_dir)
    engine = SearchEngine(embedder=embedder, vector_store=store)

    retrieved = engine.retrieve(args.query, k=args.k)

    print("\n" + "=" * 70)
    print(f"QUESTION: {args.query}")
    print("=" * 70)

    if not retrieved:
        print("\nNo relevant chunks found in the index.")
        return

    print(f"\n🔎 Retrieved {len(retrieved)} chunk(s):")
    for r in retrieved:
        src = r["metadata"].get("source", "unknown")
        page = r["metadata"].get("page", "?")
        preview = r["content"][:120].replace("\n", " ")
        print(f"   [{r['rank']}] {src} (page {page}, dist={r['distance']:.4f}): {preview}...")

    if args.no_llm:
        print("\n(--no-llm set: skipping generation step)")
        return

    try:
        generator = LLMGenerator()
    except ValueError as exc:
        print(f"\n⚠️  {exc}")
        print("   (Retrieval results above are still valid; generation was skipped.)")
        return

    print("\n💬 Generating answer...\n")
    answer = generator.generate_answer(args.query, retrieved)

    print("-" * 70)
    print("ANSWER:")
    print("-" * 70)
    print(answer)
    print("=" * 70 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="A from-scratch Retrieval-Augmented Generation (RAG) CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Ingest and index a directory of documents.")
    index_parser.add_argument("--path", required=True, help="Directory containing .pdf/.txt files.")
    index_parser.add_argument("--output", default=DEFAULT_INDEX_DIR, help="Where to save the vector index.")
    index_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    index_parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    index_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="sentence-transformers model name.")
    index_parser.set_defaults(func=cmd_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against a saved index.")
    ask_parser.add_argument("query", help="Natural language question.")
    ask_parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR, help="Path to a saved vector index.")
    ask_parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    ask_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="sentence-transformers model name.")
    ask_parser.add_argument("--no-llm", action="store_true", help="Only show retrieval results, skip LLM call.")
    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
