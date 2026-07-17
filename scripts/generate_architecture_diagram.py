"""One-off script to generate architecture-diagram.png. Not part of the
RAG pipeline itself. Requires matplotlib.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")

COLOR_INDEX = "#CEE3F6"
COLOR_QUERY = "#D9EFE1"
COLOR_STORE = "#F5DCC0"
BORDER = "#3D3D3A"


def box(x, y, w, h, text, color, fontsize=9.5, weight="bold"):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=BORDER, facecolor=color, zorder=2,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, zorder=3, wrap=True)
    return (x, y, w, h)


def arrow(b1, b2, side1="right", side2="left", label=None, color=BORDER, style="-|>"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {
        "right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
        "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1),
    }
    pts2 = {
        "right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
        "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2),
    }
    p1, p2 = pts[side1], pts2[side2]
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                         linewidth=1.3, color=color, zorder=1,
                         connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=7.8,
                 style="italic", color="#444")


# ---- Title / section headers ----
ax.text(3.6, 8.6, "INDEXING PIPELINE (run once per document collection)",
         ha="center", fontsize=11, weight="bold", color="#0C447C")
ax.text(10.6, 8.6, "RETRIEVAL & GENERATION PIPELINE (run per query)",
         ha="center", fontsize=11, weight="bold", color="#085041")

# ---- Indexing pipeline (left) ----
b_docs = box(0.4, 7.2, 2.4, 0.9, "Raw documents\n(PDF, TXT)\ndata/", COLOR_INDEX)
b_loader = box(0.4, 5.9, 2.4, 0.9, "Document Loader\nPDFLoader / TextLoader\n(document_loader.py)", COLOR_INDEX)
b_chunker = box(0.4, 4.6, 2.4, 0.9, "Text Chunker\noverlapping chunks\n(text_chunker.py)", COLOR_INDEX)
b_embed1 = box(0.4, 3.3, 2.4, 0.9, "Embedding Generator\nall-MiniLM-L6-v2\n(embedder.py)", COLOR_INDEX)
b_store = box(0.4, 2.0, 2.4, 0.9, "Vector Store\nFAISS IndexFlatL2\n(vector_store.py)", COLOR_STORE)
b_docstore = box(3.4, 2.0, 2.6, 0.9, "Document Store\nchunk_map (text + metadata)\nsaved to index_store/", COLOR_STORE)

arrow(b_docs, b_loader, "bottom", "top")
arrow(b_loader, b_chunker, "bottom", "top")
arrow(b_chunker, b_embed1, "bottom", "top")
arrow(b_embed1, b_store, "bottom", "top")
arrow(b_store, b_docstore, "right", "left", label="chunk IDs")

# ---- Retrieval & generation pipeline (right) ----
b_query = box(7.6, 7.2, 2.3, 0.9, "User Query (CLI)\nask \"...\"", COLOR_QUERY)
b_embed2 = box(7.6, 5.9, 2.3, 0.9, "Embedding Generator\n(same model as indexing)", COLOR_QUERY)
b_search = box(7.6, 4.6, 2.3, 0.9, "Search Engine\ntop-k similarity search\n(search_engine.py)", COLOR_QUERY)
b_prompt = box(7.6, 3.3, 2.3, 0.9, "Prompt Builder\ncontext + citations\n(llm_client.py)", COLOR_QUERY)
b_llm = box(7.6, 2.0, 2.3, 0.9, "LLM API\nAnthropic / OpenAI", COLOR_QUERY)
b_answer = box(7.6, 0.7, 2.3, 0.9, "Final Answer\nwith [Source: ...] citations", COLOR_QUERY)

arrow(b_query, b_embed2, "bottom", "top")
arrow(b_embed2, b_search, "bottom", "top")
arrow(b_search, b_prompt, "bottom", "top")
arrow(b_prompt, b_llm, "bottom", "top")
arrow(b_llm, b_answer, "bottom", "top")

# Cross-pipeline connection: vector store <-> search engine
b_index_ref = box(10.7, 4.6, 2.3, 0.9, "Saved Vector Index\nindex_store/\n(loaded by ask command)", COLOR_STORE)
arrow(b_store, b_index_ref, "bottom", "bottom", label="save()")
arrow(b_index_ref, b_search, "left", "right", label="load() + search()")

# Legend
ax.add_patch(FancyBboxPatch((0.4, 0.2), 0.3, 0.3, boxstyle="round,pad=0.02", facecolor=COLOR_INDEX, edgecolor=BORDER))
ax.text(0.85, 0.35, "Indexing components", fontsize=8.5, va="center")
ax.add_patch(FancyBboxPatch((3.0, 0.2), 0.3, 0.3, boxstyle="round,pad=0.02", facecolor=COLOR_QUERY, edgecolor=BORDER))
ax.text(3.45, 0.35, "Query-time components", fontsize=8.5, va="center")
ax.add_patch(FancyBboxPatch((5.9, 0.2), 0.3, 0.3, boxstyle="round,pad=0.02", facecolor=COLOR_STORE, edgecolor=BORDER))
ax.text(6.35, 0.35, "Persisted storage", fontsize=8.5, va="center")

ax.text(7, 8.95, "RAG-Powered Document Q&A System — Architecture", ha="center",
         fontsize=15, weight="bold")

plt.tight_layout()
plt.savefig("/home/claude/rag-project/architecture-diagram.png", dpi=180, bbox_inches="tight")
print("saved architecture-diagram.png")
