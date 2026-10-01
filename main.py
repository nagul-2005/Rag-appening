import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import settings
from src.graph import run_rag_pipeline, get_vectorstore
from src.ingest import ingest_pdf
from src.schemas import ChatRequest, ChatResponse, HealthResponse, RetrievedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure vector store is ready on startup
    if not settings.CHROMA_DIR.exists() or not any(settings.CHROMA_DIR.iterdir()):
        logger.info("ChromaDB not found on startup. Indexing eBook...")
        ingest_pdf()
    yield


app = FastAPI(
    title="Agentic AI eBook RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# Allow requests from Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns the service status and number of indexed chunks."""
    count = None
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        vector_store="chroma",
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.LLM_MODEL,
        total_chunks=count
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Executes the LangGraph RAG pipeline for the given question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = run_rag_pipeline(request.question)

        chunks = [
            RetrievedChunk(
                content=c["content"],
                metadata=c["metadata"],
                relevance_score=c["relevance_score"]
            )
            for c in result.get("retrieved_chunks", [])
        ]

        return ChatResponse(
            answer=result["answer"],
            retrieved_chunks=chunks,
            confidence_score=result["confidence_score"]
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
