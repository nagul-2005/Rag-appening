import logging
from typing import Any, Dict, List, TypedDict, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.ingest import get_embedding_function, ingest_pdf
from src.utils import calculate_confidence_score, normalize_distance_to_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """LangGraph state representation for RAG workflow."""
    question: str
    raw_retrieved: List[Dict[str, Any]]
    graded_chunks: List[Dict[str, Any]]
    answer: str
    confidence_score: float
    is_grounded: bool


def get_vectorstore_instance():
    """
    Retrieves the persisted vectorstore instance. Automatically triggers ingestion if empty.

    Returns:
        VectorStore instance ready for similarity searches.
    """
    embedding_fn = get_embedding_function()
    store_type = settings.VECTOR_STORE_TYPE.lower()

    if store_type == "pinecone" and settings.PINECONE_API_KEY:
        try:
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore(
                embedding=embedding_fn,
                index_name=settings.PINECONE_INDEX_NAME,
                pinecone_api_key=settings.PINECONE_API_KEY
            )
        except Exception as e:
            logger.warning(f"Pinecone load failed ({e}). Falling back to ChromaDB.")

    # ChromaDB load
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    persist_path = settings.CHROMA_PERSIST_DIR
    if not persist_path.exists() or not any(persist_path.iterdir()):
        logger.info(f"Vector store directory at {persist_path} is empty or missing. Triggering auto-ingestion...")
        ingest_pdf()

    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(persist_path)
    )


def get_llm_instance():
    """
    Returns a callable LLM wrapper. Tries:
      1. google.genai SDK directly (supports current API keys)
      2. langchain_google_genai (legacy)
      3. langchain_openai
      4. None (fallback extractive summarizer)
    """
    provider = settings.LLM_PROVIDER.lower()

    # ── Google via google.genai SDK (primary path for current API keys) ──
    if provider == "google" and settings.GOOGLE_API_KEY:
        try:
            import google.genai as genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            model_name = settings.LLM_MODEL  # e.g. "gemini-2.5-flash"

            class GoogleGenAILLM:
                """Thin wrapper around google.genai Client to match the LangChain .invoke() interface."""
                def invoke(self, messages):
                    # Convert LangChain messages to plain text prompt
                    parts = []
                    for msg in messages:
                        role = getattr(msg, "type", "human")
                        if role == "system":
                            parts.append(f"[SYSTEM]\n{msg.content}")
                        else:
                            parts.append(msg.content)
                    prompt = "\n\n".join(parts)

                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=settings.LLM_TEMPERATURE,
                        )
                    )
                    # Return object with .content attribute to match LangChain interface
                    class _Resp:
                        content = response.text
                    return _Resp()

            logger.info(f"Initialized Google GenAI LLM via google.genai SDK ({model_name}).")
            return GoogleGenAILLM()
        except Exception as e:
            logger.warning(f"google.genai SDK init failed: {e}. Trying langchain_google_genai...")

        # ── Fallback: langchain_google_genai ──
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = settings.LLM_MODEL
            logger.info(f"Initialized Google Gemini LLM via langchain ({model_name}).")
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=settings.LLM_TEMPERATURE
            )
        except Exception as e:
            logger.warning(f"langchain_google_genai init failed: {e}.")

    # ── OpenAI ──
    if provider == "openai" and settings.OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            logger.info(f"Initialized OpenAI LLM ({settings.LLM_MODEL}).")
            return ChatOpenAI(
                model=settings.LLM_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=settings.LLM_TEMPERATURE
            )
        except Exception as e:
            logger.warning(f"OpenAI LLM initialization failed: {e}.")

    logger.warning("No valid LLM configured. Using grounded extractive summarizer fallback.")
    return None



# --- Graph Nodes ---

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 1: Retrieve top-k relevant document chunks from the vector database.
    """
    question = state["question"]
    logger.info(f"Graph Node [retrieve]: Searching top-{settings.TOP_K} chunks for query: '{question}'")

    vectorstore = get_vectorstore_instance()
    raw_retrieved = []

    try:
        results = vectorstore.similarity_search_with_score(question, k=settings.TOP_K)
        for doc, score in results:
            norm_score = normalize_distance_to_score(score, metric="cosine")
            raw_retrieved.append({"doc": doc, "score": norm_score})
    except Exception as e:
        logger.warning(f"Vector search with score failed: {e}. Falling back to similarity search...")
        docs = vectorstore.similarity_search(question, k=settings.TOP_K)
        for idx, doc in enumerate(docs):
            raw_retrieved.append({"doc": doc, "score": round(0.85 - (idx * 0.1), 2)})

    logger.info(f"Graph Node [retrieve]: Found {len(raw_retrieved)} candidate chunks.")
    return {"raw_retrieved": raw_retrieved}



def grade_documents_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 2: Grade retrieved document chunks based on relevance threshold and assign scores.
    """
    raw_retrieved = state.get("raw_retrieved", [])
    logger.info("Graph Node [grade_documents]: Grading retrieved chunks for relevance...")

    graded_chunks = []
    scores = []

    for item in raw_retrieved:
        doc: Document = item["doc"]
        score: float = item["score"]

        # Filter chunks below relevance threshold
        if score >= settings.RELEVANCE_THRESHOLD:
            page = doc.metadata.get("page", 1)
            source = doc.metadata.get("source", settings.PDF_FILENAME)
            chunk_id = doc.metadata.get("chunk_id", f"page_{page}")

            graded_chunks.append({
                "content": doc.page_content.strip(),
                "metadata": {
                    "page": page,
                    "source": source,
                    "chunk_id": chunk_id
                },
                "relevance_score": score
            })
            scores.append(score)

    overall_confidence = calculate_confidence_score(scores)
    is_grounded = len(graded_chunks) > 0 and overall_confidence >= 0.20

    logger.info(
        f"Graph Node [grade_documents]: Kept {len(graded_chunks)}/{len(raw_retrieved)} chunks. "
        f"Overall Confidence Score: {overall_confidence:.2f}"
    )

    return {
        "graded_chunks": graded_chunks,
        "confidence_score": overall_confidence,
        "is_grounded": is_grounded
    }


def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Node 3: Formulate a grounded response using LLM or structured context synthesis.
    """
    question = state["question"]
    graded_chunks = state.get("graded_chunks", [])
    is_grounded = state.get("is_grounded", False)
    confidence = state.get("confidence_score", 0.0)

    logger.info("Graph Node [generate]: Synthesizing response...")

    if not is_grounded or not graded_chunks:
        fallback_msg = (
            "I cannot answer this question based on the provided Agentic AI eBook context. "
            "The requested topic was not found in the ingested document."
        )
        return {
            "answer": fallback_msg,
            "graded_chunks": [],
            "confidence_score": 0.0
        }

    # Format context strings
    context_str = "\n\n".join([
        f"--- CONTEXT CHUNK {idx+1} (Page {c['metadata']['page']}) [Relevance: {c['relevance_score']:.2f}] ---\n{c['content']}"
        for idx, c in enumerate(graded_chunks)
    ])

    llm = get_llm_instance()

    if llm is not None:
        system_prompt = (
            "You are an expert AI Assistant specialized in Agentic AI architectures. "
            "Your task is to answer the user's question STRICTLY based ONLY on the provided context snippets "
            "from the 'Agentic AI eBook'. Do NOT use external knowledge or invent facts.\n"
            "If the context does not contain sufficient details to answer the question, state explicitly: "
            "'I cannot answer this question based on the provided Agentic AI eBook context.'\n\n"
            "Maintain a professional, structured, and informative tone."
        )

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context Snippets:\n{context_str}\n\n"
            f"Grounded Answer:"
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            answer_text = response.content.strip()
            return {"answer": answer_text}
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}. Falling back to extractive synthesis.")

    # Extractive grounded synthesis fallback (if LLM API key not set)
    synthesized_points = []
    for c in graded_chunks[:3]:
        snippet = c['content'].replace('\n', ' ')
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        synthesized_points.append(f"• (Page {c['metadata']['page']}): {snippet}")

    fallback_answer = (
        f"Based on the Agentic AI eBook context (Confidence Score: {confidence:.2f}), "
        f"here are the key grounded findings regarding '{question}':\n\n" +
        "\n\n".join(synthesized_points) +
        "\n\n*(Note: Configure OPENAI_API_KEY or GOOGLE_API_KEY in .env for full conversational LLM generation.)*"
    )

    return {"answer": fallback_answer}


# --- LangGraph Construction ---

def build_rag_graph():
    """
    Constructs and compiles the stateful LangGraph RAG workflow.

    Workflow topology:
      START -> retrieve -> grade_documents -> generate -> END
    """
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)

    app_graph = workflow.compile()
    logger.info("LangGraph RAG pipeline compiled successfully.")
    return app_graph


# Pre-compiled Graph Application instance
rag_graph_app = build_rag_graph()


def run_rag_pipeline(question: str) -> Dict[str, Any]:
    """
    Executes the RAG pipeline end-to-end for a user question.

    Args:
        question: User query string.

    Returns:
        Dictionary containing answer, graded_chunks, and confidence_score.
    """
    initial_state: RAGState = {
        "question": question,
        "raw_retrieved": [],
        "graded_chunks": [],
        "answer": "",
        "confidence_score": 0.0,
        "is_grounded": False
    }

    final_state = rag_graph_app.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "retrieved_chunks": final_state.get("graded_chunks", []),
        "confidence_score": final_state.get("confidence_score", 0.0)
    }
