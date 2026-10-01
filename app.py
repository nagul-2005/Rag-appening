import json
import sys
from pathlib import Path
import requests
import streamlit as st

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings
from src.graph import run_rag_pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (layout only, no content styling) ─────────────────────────────
st.markdown(
    """
    <style>
        /* Tighten the top padding */
        .block-container { padding-top: 1.5rem; }
        /* Chat input fixed width */
        .stChatInput { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper functions ──────────────────────────────────────────────────────────

def load_sample_queries():
    """Load pre-configured sample queries from sample_queries.json."""
    sample_file = Path(__file__).resolve().parent / "sample_queries.json"
    if sample_file.exists():
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def query_rag_backend(question: str) -> dict:
    """
    Query the FastAPI backend; fall back to direct pipeline if backend is down.
    """
    try:
        resp = requests.post(
            f"{settings.BACKEND_URL}/chat",
            json={"question": question},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    # Direct local pipeline fallback
    return run_rag_pipeline(question)


def score_color(score: float) -> str:
    """Return a colour hex string based on the confidence/relevance score."""
    if score >= 0.70:
        return "🟢"
    elif score >= 0.45:
        return "🟡"
    return "🔴"


def render_chunks(chunks: list):
    """Render retrieved context chunks using native Streamlit components."""
    if not chunks:
        st.info("No context chunks were retrieved for this query.")
        return

    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        page = meta.get("page", "?")
        source = meta.get("source", "Ebook-Agentic-AI.pdf")
        score = chunk.get("relevance_score", 0.0)
        content = chunk.get("content", "")

        # One card per chunk using a container + columns
        with st.container(border=True):
            col_left, col_right = st.columns([3, 1])

            with col_left:
                st.markdown(f"**Chunk {idx + 1}** &nbsp;·&nbsp; 📄 Page **{page}** &nbsp;·&nbsp; `{source}`")

            with col_right:
                indicator = score_color(score)
                st.markdown(
                    f"**Relevance** &nbsp; {indicator} &nbsp; `{score:.2f}`",
                    help="Similarity score from vector search (0.0 – 1.0)"
                )

            st.progress(min(score, 1.0))
            st.caption(content)


def render_confidence_badge(score: float):
    """Render an overall confidence metric row."""
    indicator = score_color(score)
    label = "High" if score >= 0.70 else "Medium" if score >= 0.45 else "Low"
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        st.metric(label="Confidence Score", value=f"{score:.2f}", delta=None)
    with col2:
        st.markdown(f"### {indicator}")
        st.caption(f"{label} confidence")


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.caption("Grounded on the *Agentic AI eBook*")
    st.divider()

    # Backend status
    st.subheader("🔌 Backend Status")
    try:
        health = requests.get(f"{settings.BACKEND_URL}/health", timeout=3).json()
        st.success("FastAPI Online ✅")
        st.caption(f"**Vector Store:** {health.get('vector_store', 'chroma').upper()}")
        st.caption(f"**Embeddings:** {health.get('embedding_provider', 'google')}")
        st.caption(f"**LLM Model:** {health.get('llm_model', 'gemini-2.5-flash')}")
        count = health.get("indexed_documents_count")
        if count:
            st.caption(f"**Indexed Chunks:** {count:,}")
    except Exception:
        st.info("⚡ Standalone Mode (direct pipeline)")

    st.divider()

    # Sample queries
    st.subheader("💡 Sample Queries")
    sample_queries = load_sample_queries()

    if sample_queries:
        for sq in sample_queries:
            if st.button(
                sq["query"],
                key=f"sq_{sq['id']}",
                use_container_width=True,
                help=f"Category: {sq.get('category', '')}"
            ):
                st.session_state.pending_query = sq["query"]

    st.divider()

    # Architecture info
    st.subheader("🏗️ Architecture")
    st.caption("**Pipeline:**  Retrieve → Grade → Generate")
    st.caption("**Chunking:**  800 chars / 150 overlap")
    st.caption("**Embeddings:**  gemini-embedding-001")
    st.caption("**LLM:**  gemini-2.5-flash")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🤖 Agentic AI eBook — RAG Chat Assistant")
st.caption(
    "Ask anything about the *Agentic AI eBook*. "
    "Answers are strictly grounded on retrieved document context."
)
st.divider()

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            confidence = msg.get("confidence_score", 0.0)
            chunks = msg.get("retrieved_chunks", [])

            render_confidence_badge(confidence)

            if chunks:
                with st.expander(
                    f"📚 Retrieved Context  ({len(chunks)} chunks)",
                    expanded=False
                ):
                    render_chunks(chunks)


# ── Handle new query ──────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about Agentic AI…")

# Merge sidebar button queries and chat input
active_query = st.session_state.pending_query or user_input

# Avoid re-triggering the same pending query on rerun
if st.session_state.pending_query and not user_input:
    st.session_state.pending_query = None

if active_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving context & running LangGraph pipeline…"):
            result = query_rag_backend(active_query)

        answer = result.get("answer", "No answer generated.")
        chunks = result.get("retrieved_chunks", [])

        # Answer
        st.markdown(answer)


        # Context expander
        if chunks:
            with st.expander(
                f"📚 Retrieved Context  ({len(chunks)} chunks)",
                expanded=False
            ):
                render_chunks(chunks)
        else:
            st.warning("⚠️ No relevant context found in the eBook for this query.")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "retrieved_chunks": chunks,
    })
