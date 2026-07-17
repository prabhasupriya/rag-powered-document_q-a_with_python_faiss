"""
build_evaluation_report.py

Runs the REAL end-to-end pipeline (SearchEngine + LLMGenerator, whatever
provider is set in .env) over a fixed set of 15 in-corpus questions + 1
negative control, and writes evaluation-report.md with genuine retrieved
chunks, genuine distances, and genuine LLM-generated answers.

Run this from the project root, after `python -m src.cli index --path ./data`:

    python scripts/build_evaluation_report.py

Requires .env to have a valid LLM_API_KEY for whichever LLM_PROVIDER you set
(anthropic / openai / groq).
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.embeddings.embedder import TextEmbedder, DEFAULT_MODEL_NAME
from src.storage.vector_store import VectorStore
from src.retrieval.search_engine import SearchEngine
from src.generation.llm_client import LLMGenerator

INDEX_DIR = os.getenv("INDEX_DIR", "./index_store")
TOP_K = int(os.getenv("TOP_K", "3"))

QUESTIONS = [
    "How many days of paid time off do full-time employees accrue per year?",
    "How many weeks of parental leave does the primary caregiver receive?",
    "What is required for fully remote work arrangements at Nimbus Analytics?",
    "How many events per second can StreamSight handle before requiring horizontal scaling?",
    "How long is raw event data retained in hot storage?",
    "What triggers an anomaly-detection-based alert in StreamSight?",
    "What is the price of the Professional tier of StreamSight?",
    "What multi-factor authentication method does Nimbus Analytics use, and why is SMS disabled?",
    "How often are encryption keys rotated?",
    "Within how many hours must customers be notified of a data breach?",
    "What is the default rate limit for read endpoints in the StreamSight API?",
    "How is a webhook payload signed and verified in the StreamSight API?",
    "What was Nimbus Analytics' total revenue for fiscal year 2025?",
    "What was the net revenue retention (NRR) for fiscal year 2025?",
    "What happens during a new engineer's first week at Nimbus Analytics?",
    "What is the capital of France?",  # negative control - should be refused
]


def main():
    print(f"Loading embedding model '{DEFAULT_MODEL_NAME}'...")
    embedder = TextEmbedder(model_name=DEFAULT_MODEL_NAME)

    print(f"Loading vector index from '{INDEX_DIR}'...")
    store = VectorStore.load(INDEX_DIR)
    engine = SearchEngine(embedder=embedder, vector_store=store)

    print(f"Initializing LLM ({os.getenv('LLM_PROVIDER', 'anthropic')})...")
    generator = LLMGenerator()

    report_lines = []
    report_lines.append("# Evaluation Report\n")
    report_lines.append(
        f"This report documents {len(QUESTIONS)-1} in-corpus questions plus 1 negative control, "
        f"run end-to-end against the real pipeline in this repo: `SearchEngine` (FAISS retrieval, "
        f"`{DEFAULT_MODEL_NAME}` embeddings) followed by `LLMGenerator` "
        f"(provider: `{os.getenv('LLM_PROVIDER', 'anthropic')}`, model: `{os.getenv('LLM_MODEL', 'default')}`). "
        f"Every retrieved chunk, every distance score, and every generated answer below is genuine "
        f"tool output from running this exact codebase — nothing is hand-written.\n"
    )
    report_lines.append(f"*Generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                         f"by `scripts/build_evaluation_report.py`.*\n")
    report_lines.append("---\n")

    correct_citation_count = 0
    total = len(QUESTIONS)

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{i}/{total}] {question}")
        retrieved = engine.retrieve(question, k=TOP_K)

        report_lines.append(f"## Q{i}: {question}\n")
        report_lines.append("**Top-{} retrieved:**\n".format(len(retrieved)))
        for r in retrieved:
            src = r["metadata"].get("source", "unknown")
            page = r["metadata"].get("page", "?")
            preview = r["content"].replace("\n", " ").strip()
            if len(preview) > 220:
                preview = preview[:220] + "..."
            report_lines.append(
                f"{r['rank']}. `{src}` (page {page}, dist={r['distance']:.4f}) — \"{preview}\""
            )
        report_lines.append("")

        try:
            answer = generator.generate_answer(question, retrieved)
        except Exception as exc:
            answer = f"[ERROR calling LLM: {exc}]"

        report_lines.append("**Generated answer:**\n")
        report_lines.append(f"> {answer}\n")

        cited = any(
            r["metadata"].get("source", "") in answer for r in retrieved
        )
        refused = "cannot answer" in answer.lower()
        if cited or (i == total and refused):
            correct_citation_count += 1
            note = "✅ Source correctly cited." if cited else "✅ Correctly refused (negative control)."
        else:
            note = "⚠️ Review manually — no expected source name found in the answer."
        report_lines.append(f"**Citation check:** {note}\n")
        report_lines.append("---\n")

    report_lines.append("## Summary\n")
    report_lines.append(f"- Questions evaluated: {total} ({total-1} in-corpus + 1 negative control)")
    report_lines.append(f"- Answers with a correctly matched citation (or correct refusal): {correct_citation_count} / {total}")
    report_lines.append(f"- Retrieval: FAISS `IndexFlatL2`, embedding model `{DEFAULT_MODEL_NAME}`, top-k={TOP_K}")
    report_lines.append(f"- Generation: `{os.getenv('LLM_PROVIDER', 'anthropic')}` / `{os.getenv('LLM_MODEL', 'default')}`")
    report_lines.append("")
    report_lines.append(
        "Review each question above manually: check that the retrieved chunks were actually "
        "relevant, and that the generated answer's facts match the source document. The negative "
        "control (final question) should be refused with \"I cannot answer this based on the "
        "provided documents\" — if it isn't, that's a prompt-grounding issue worth flagging in your write-up."
    )

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation-report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ Wrote {out_path}")


if __name__ == "__main__":
    main()
