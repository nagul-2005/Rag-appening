import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.config import settings
from src.utils import download_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleEmbeddings:
    """Wrapper around the Google GenAI SDK for generating embeddings."""
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        import google.genai as genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            self.client.models.embed_content(model=self.model, contents=t).embeddings[0].values
            for t in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        return self.client.models.embed_content(model=self.model, contents=text).embeddings[0].values


def get_embedding_model():
    """Returns the configured embedding model."""
    if settings.GOOGLE_API_KEY:
        return GoogleEmbeddings(api_key=settings.GOOGLE_API_KEY, model=settings.EMBEDDING_MODEL)

    # Lightweight fallback for local testing without an API key
    class LocalSimpleEmbeddings:
        def embed_documents(self, texts):
            return [self.embed_query(t) for t in texts]

        def embed_query(self, text):
            import numpy as np
            vec = np.zeros(384, dtype=np.float32)
            for word in text.lower().split():
                vec[abs(hash(word)) % 384] += 1.0
            norm = np.linalg.norm(vec)
            return (vec / norm).tolist() if norm > 0 else vec.tolist()

    return LocalSimpleEmbeddings()


def load_pdf(pdf_path: Path) -> List[Document]:
    """Read the PDF and return page documents."""
    logger.info(f"Reading PDF from {pdf_path}...")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(str(pdf_path)).load()
    except Exception:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        docs = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": idx + 1, "source": settings.PDF_FILENAME}))

    # Ensure page numbers are 1-indexed
    for doc in docs:
        page = doc.metadata.get("page", 0)
        doc.metadata["page"] = page if page > 0 else page + 1
        doc.metadata["source"] = settings.PDF_FILENAME

    logger.info(f"Loaded {len(docs)} pages.")
    return docs


def split_text(docs: List[Document]) -> List[Document]:
    """Split page documents into smaller chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)

    # Add chunk IDs for traceability
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_id"] = f"page_{page}_chunk_{i}"

    logger.info(f"Created {len(chunks)} chunks from {len(docs)} pages.")
    return chunks


def ingest_pdf() -> int:
    """Download, extract, chunk, and index the eBook into ChromaDB."""
    # 1. Download
    pdf_path = download_pdf(settings.PDF_URL, settings.PDF_PATH)

    # 2. Extract text
    pages = load_pdf(pdf_path)

    # 3. Chunk
    chunks = split_text(pages)

    # 4. Save to Chroma
    logger.info(f"Saving {len(chunks)} chunks to ChromaDB at {settings.CHROMA_DIR}...")
    embeddings = get_embedding_model()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.COLLECTION_NAME,
        persist_directory=str(settings.CHROMA_DIR)
    )
    logger.info("ChromaDB index ready!")
    return len(chunks)


if __name__ == "__main__":
    ingest_pdf()
