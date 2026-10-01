import json
import sys
from pathlib import Path
import requests
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings
from src.graph import run_rag_pipeline

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic AI eBook Assistant",
    page_icon="🤖",
    layout="wide"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_sample_queries():
    """Load sample questions from sample_queries.json."""
    file_path = Path(__file__).resolve().parent / "sample_queries.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def query_rag(question: str) -> dict:
    """Call the FastAPI /chat endpoint, with direct pipeline fallback if backend is down."""
    try:
        resp = requests.post(
            f"{settings.BACKEND_URL}/chat",
            json={"question": question},
            timeout=25
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass

    # Direct fallback if FastAPI is offline
    return run_rag_pipeline(question)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_query" not in st.session_state:
    st.session_state.selected_query = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agentic AI Bot")
    st.caption("Grounded on the Agentic AI eBook")
    st.divider()

    # Backend health check
    st.subheader("System Status")
    try:
        health = requests.get(f"{settings.BACKEND_URL}/health", timeout=2).json()
        st.success("FastAPI Online ✅")
        st.caption(f"**Model:** {health.get('llm_model')}")
        st.caption(f"**Total Chunks:** {health.get('total_chunks')}")
    except Exception:
        st.info("Direct Pipeline Mode ⚡")

    st.divider()

    # Sample query buttons
    st.subheader("💡 Sample Questions")
    for item in get_sample_queries():
        if st.button(item["query"], key=f"q_{item['id']}", use_container_width=True):
            st.session_state.selected_query = item["query"]

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_query = None
        st.rerun()


# ── Main Chat Area ────────────────────────────────────────────────────────────
st.header("Agentic AI eBook Assistant")
st.caption("Ask questions about agentic patterns, reflection, multi-agent systems, and tool usage.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display confidence and chunks for assistant replies
        if msg["role"] == "assistant":
            score = msg.get("confidence_score", 0.0)
            st.caption(f"🎯 **Confidence Score:** `{score:.2f}`")

            chunks = msg.get("retrieved_chunks", [])
            if chunks:
                with st.expander(f"📚 View Retrieved Context ({len(chunks)} chunks)"):
                    for idx, c in enumerate(chunks):
                        page = c.get("metadata", {}).get("page", "?")
                        rel = c.get("relevance_score", 0.0)
                        st.markdown(f"**Chunk {idx + 1}** (Page {page}) — Relevance: `{rel:.2f}`")
                        st.text(c.get("content", ""))


# Handle input (either from chat box or sidebar click)
user_input = st.chat_input("Ask a question about the eBook...")
prompt = st.session_state.selected_query or user_input
st.session_state.selected_query = None  # Reset selection

if prompt:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get and show assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Searching eBook & generating answer..."):
            result = query_rag(prompt)

        answer = result.get("answer", "No answer generated.")
        chunks = result.get("retrieved_chunks", [])
        confidence = result.get("confidence_score", 0.0)

        st.markdown(answer)
        st.caption(f"🎯 **Confidence Score:** `{confidence:.2f}`")

        if chunks:
            with st.expander(f"📚 View Retrieved Context ({len(chunks)} chunks)"):
                for idx, c in enumerate(chunks):
                    page = c.get("metadata", {}).get("page", "?")
                    rel = c.get("relevance_score", 0.0)
                    st.markdown(f"**Chunk {idx + 1}** (Page {page}) — Relevance: `{rel:.2f}`")
                    st.text(c.get("content", ""))

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "retrieved_chunks": chunks,
        "confidence_score": confidence
    })
