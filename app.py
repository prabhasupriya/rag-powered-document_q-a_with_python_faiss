"""
app.py

Streamlit UI for the RAG Document Q&A system. Wraps the existing pipeline
(src/ingestion, src/embeddings, src/storage, src/retrieval, src/generation)
with zero changes to that code -- this file only adds a browser front end.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub (include the `data/` sample corpus).
    2. On share.streamlit.io, create a new app pointing at this repo/app.py.
    3. In the app's Settings -> Secrets, add:
           LLM_PROVIDER = "groq"
           LLM_API_KEY = "gsk_..."
           LLM_MODEL = "llama-3.3-70b-versatile"
    4. Deploy. First load will build the index from data/ automatically.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.embeddings.embedder import TextEmbedder, DEFAULT_MODEL_NAME
from src.generation.llm_client import LLMGenerator
from src.ingestion.document_loader import DirectoryProcessor
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.search_engine import SearchEngine
from src.storage.vector_store import VectorStore

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = os.getenv("INDEX_DIR", "./index_store")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def _load_secrets_into_env():
    """On Streamlit Community Cloud, config lives in st.secrets, not .env.
    Copy anything found there into os.environ so the existing pipeline code
    (which reads via os.getenv) doesn't need to know the difference."""
    try:
        for key in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        # st.secrets raises if no secrets.toml exists at all (e.g. local run) -- fine, ignore.
        pass


_load_secrets_into_env()


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return TextEmbedder(model_name=DEFAULT_MODEL_NAME)


@st.cache_resource(show_spinner="Building/loading the vector index...")
def get_vector_store(_embedder, data_dir: str, index_dir: str):
    """Load a saved index if present; otherwise build one from data_dir."""
    if os.path.exists(os.path.join(index_dir, "store_meta.json")):
        return VectorStore.load(index_dir)

    processor = DirectoryProcessor()
    documents = processor.process(data_dir)
    if not documents:
        return VectorStore(embedding_dimension=_embedder.embedding_dimension)

    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = _embedder.generate_embeddings([c.content for c in chunks])

    store = VectorStore(embedding_dimension=_embedder.embedding_dimension)
    store.add_embeddings(embeddings, chunks)
    store.save(index_dir)
    return store


def rebuild_index_with_uploads(embedder, uploaded_files) -> VectorStore:
    """Build a fresh in-memory index from the baked-in data/ corpus PLUS
    any files the user uploaded through the sidebar this session."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy the baked-in sample corpus in first.
        if os.path.isdir(DATA_DIR):
            for fname in os.listdir(DATA_DIR):
                src_path = os.path.join(DATA_DIR, fname)
                if os.path.isfile(src_path):
                    with open(src_path, "rb") as f_in, open(os.path.join(tmp_dir, fname), "wb") as f_out:
                        f_out.write(f_in.read())

        # Add uploaded files.
        for uploaded in uploaded_files:
            with open(os.path.join(tmp_dir, uploaded.name), "wb") as f_out:
                f_out.write(uploaded.getbuffer())

        processor = DirectoryProcessor()
        documents = processor.process(tmp_dir)
        chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        embeddings = embedder.generate_embeddings([c.content for c in chunks])

        store = VectorStore(embedding_dimension=embedder.embedding_dimension)
        store.add_embeddings(embeddings, chunks)
        return store


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📚 RAG Document Q&A")
st.sidebar.caption("FAISS + sentence-transformers + your choice of LLM provider")

provider = os.getenv("LLM_PROVIDER", "not set")
model = os.getenv("LLM_MODEL", "not set")
st.sidebar.markdown(f"**LLM provider:** `{provider}`  \n**Model:** `{model}`")

k = st.sidebar.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=3)
show_context = st.sidebar.checkbox("Show retrieved context", value=True)

st.sidebar.divider()
st.sidebar.subheader("Add your own documents (optional)")
uploaded_files = st.sidebar.file_uploader(
    "Upload .pdf / .txt files to search alongside the sample corpus",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)
rebuild = st.sidebar.button("Rebuild index with uploads", disabled=not uploaded_files)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Ask questions about your documents")
st.caption(
    "This demo ships with a sample corpus (`data/`) describing a fictional company, "
    "Nimbus Analytics. Ask about HR policy, the product spec, security policy, API docs, "
    "the FY2025 financial report, or the engineering onboarding guide -- or upload your own "
    "files in the sidebar."
)

if not os.getenv("LLM_API_KEY"):
    st.warning(
        "No `LLM_API_KEY` found. Set it in `.env` locally, or in Settings -> Secrets "
        "if this app is deployed on Streamlit Community Cloud. Retrieval will still work "
        "without it -- only answer generation needs the key."
    )

embedder = get_embedder()

if rebuild and uploaded_files:
    with st.spinner("Indexing uploaded documents..."):
        st.session_state["vector_store"] = rebuild_index_with_uploads(embedder, uploaded_files)
    st.sidebar.success(f"Indexed {len(uploaded_files)} uploaded file(s) + sample corpus.")

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = get_vector_store(embedder, DATA_DIR, INDEX_DIR)

store = st.session_state["vector_store"]
engine = SearchEngine(embedder=embedder, vector_store=store)

question = st.text_input("Your question", placeholder="e.g. How many days of PTO do full-time employees get?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Retrieving relevant context..."):
        retrieved = engine.retrieve(question, k=k)

    if not retrieved:
        st.error("No documents are indexed yet. Add files in the sidebar or check `data/`.")
    else:
        if show_context:
            with st.expander(f"🔎 Retrieved {len(retrieved)} chunk(s)", expanded=False):
                for r in retrieved:
                    src = r["metadata"].get("source", "unknown")
                    page = r["metadata"].get("page", "?")
                    st.markdown(f"**[{r['rank']}] `{src}` (page {page}, distance={r['distance']:.4f})**")
                    st.text(r["content"])
                    st.divider()

        if not os.getenv("LLM_API_KEY"):
            st.info("Set an LLM_API_KEY to generate a cited answer from the retrieved context above.")
        else:
            try:
                with st.spinner("Generating answer..."):
                    generator = LLMGenerator()
                    answer = generator.generate_answer(question, retrieved)
                st.subheader("Answer")
                st.markdown(answer)
            except Exception as exc:
                st.error(f"LLM call failed: {exc}")
elif ask_clicked:
    st.warning("Type a question first.")
