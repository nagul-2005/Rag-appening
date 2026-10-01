import logging
from typing import Any, Dict, List, TypedDict

from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.ingest import get_embedding_model, ingest_pdf
from src.utils import distance_to_score, compute_confidence

logger = logging.getLogger(__name__)


# ── Pipeline State ────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    chunks: List[Dict[str, Any]]
    confidence_score: float
    answer: str


# ── Vector DB & LLM Helpers ──────────────────────────────────────────────────

def get_vectorstore():
    """Load ChromaDB collection. Ingests the PDF if not already indexed."""
    if not settings.CHROMA_DIR.exists() or not any(settings.CHROMA_DIR.iterdir()):
        logger.info("ChromaDB not found. Running ingestion first...")
        ingest_pdf()

    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(settings.CHROMA_DIR)
    )


def call_llm(prompt: str) -> str:
    """Generate text using Gemini 2.5 Flash with zero temperature for factual accuracy."""
    if not settings.GOOGLE_API_KEY:
        return "Note: Google API key not set. Please set GOOGLE_API_KEY in .env."

    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=settings.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=settings.TEMPERATURE)
    )
    return response.text or ""


# ── LangGraph Nodes ───────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> Dict[str, Any]:
    """Node 1: Query ChromaDB for the top-k most similar chunks."""
    question = state["question"]
    vs = get_vectorstore()

    # Chroma returns list of (Document, distance)
    results = vs.similarity_search_with_score(question, k=settings.TOP_K)

    chunks = []
    for doc, dist in results:
        chunks.append({
            "content": doc.page_content.strip(),
            "metadata": doc.metadata,
            "relevance_score": distance_to_score(dist)
        })

    return {"chunks": chunks}


def grade_documents(state: RAGState) -> Dict[str, Any]:
    """Node 2: Filter out chunks below the relevance threshold and calculate confidence."""
    raw_chunks = state.get("chunks", [])

    # Keep only relevant chunks
    filtered = [c for c in raw_chunks if c["relevance_score"] >= settings.MIN_SCORE]
    scores = [c["relevance_score"] for c in filtered]
    confidence = compute_confidence(scores)

    return {
        "chunks": filtered,
        "confidence_score": confidence
    }


def generate(state: RAGState) -> Dict[str, Any]:
    """Node 3: Answer strictly using the filtered chunks. Guard against hallucination."""
    question = state["question"]
    chunks = state.get("chunks", [])

    # If no relevant chunks survived grading, stop here
    if not chunks:
        return {
            "answer": "I cannot answer this question based on the provided Agentic AI eBook context.",
            "confidence_score": 0.0
        }

    # Format the context snippets
    context = "\n\n".join([
        f"[Page {c['metadata'].get('page', '?')}] {c['content']}"
        for c in chunks
    ])

    prompt = f"""You are an assistant specialized in the Agentic AI eBook.
Answer the user's question STRICTLY using only the provided context.
Do not use external knowledge or invent facts.
If the context does not contain enough information, state:
"I cannot answer this question based on the provided Agentic AI eBook context."

Context:
{context}

Question: {question}

Answer:"""

    try:
        answer = call_llm(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = "Error generating answer from LLM."

    return {"answer": answer}


# ── Build & Compile Graph ─────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(RAGState)

    # Add the 3 nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)

    # Wire the flow: START -> retrieve -> grade_documents -> generate -> END
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("grade_documents", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Compiled LangGraph application
rag_app = build_graph()


def run_rag_pipeline(question: str) -> Dict[str, Any]:
    """Helper function to run the RAG pipeline on a question."""
    result = rag_app.invoke({
        "question": question,
        "chunks": [],
        "confidence_score": 0.0,
        "answer": ""
    })

    return {
        "answer": result.get("answer", ""),
        "retrieved_chunks": result.get("chunks", []),
        "confidence_score": result.get("confidence_score", 0.0)
    }
