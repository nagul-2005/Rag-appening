import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.graph import run_rag_pipeline, get_vectorstore_instance
from src.ingest import ingest_pdf
from src.schemas import ChatRequest, ChatResponse, HealthResponse, RetrievedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI startup and shutdown tasks."""
    logger.info("Initializing Agentic AI eBook RAG FastAPI service...")
    # Check vector DB on startup
    try:
        if not settings.CHROMA_PERSIST_DIR.exists() or not any(settings.CHROMA_PERSIST_DIR.iterdir()):
            logger.info("ChromaDB directory empty on startup. Triggering initial ingestion...")
            ingest_pdf()
        else:
            logger.info(f"Vector store detected at {settings.CHROMA_PERSIST_DIR}.")
    except Exception as e:
        logger.error(f"Error during startup vector store verification: {e}")
    yield
    logger.info("Shutting down FastAPI service...")


app = FastAPI(
    title="Agentic AI eBook RAG API",
    description="Production-grade REST API serving a grounded LangGraph RAG pipeline over the Agentic AI eBook.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Root redirect to API documentation."""
    return {"message": "Welcome to Agentic AI RAG API. Access /docs for API documentation or /health for system status."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint returning system status, vector store configuration, and model info.
    """
    chunk_count = None
    try:
        vectorstore = get_vectorstore_instance()
        # Count documents in collection if available
        if hasattr(vectorstore, "_collection") and vectorstore._collection is not None:
            chunk_count = vectorstore._collection.count()
    except Exception as e:
        logger.warning(f"Could not fetch document count for health status: {e}")

    return HealthResponse(
        status="ok",
        vector_store=settings.VECTOR_STORE_TYPE,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        llm_model=settings.LLM_MODEL,
        indexed_documents_count=chunk_count
    )


@app.post("/chat", response_model=ChatResponse, tags=["RAG Chat"])
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Executes the grounded RAG graph workflow for a user question.

    - **question**: User query about the Agentic AI eBook.
    - **Returns**: Grounded answer, retrieved context chunks with metadata, and overall confidence score.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question parameter cannot be empty."
        )

    try:
        logger.info(f"Received API query: '{request.question}'")
        result = run_rag_pipeline(request.question)

        # Format retrieved chunks into Pydantic models
        formatted_chunks = [
            RetrievedChunk(
                content=chunk["content"],
                metadata=chunk["metadata"],
                relevance_score=chunk["relevance_score"]
            )
            for chunk in result.get("retrieved_chunks", [])
        ]

        return ChatResponse(
            answer=result.get("answer", ""),
            retrieved_chunks=formatted_chunks,
            confidence_score=result.get("confidence_score", 0.0)
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the RAG pipeline: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.FASTAPI_HOST,
        port=settings.FASTAPI_PORT,
        reload=True
    )
