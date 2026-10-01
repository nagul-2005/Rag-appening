import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.utils import download_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf_documents(pdf_path: Path) -> List[Document]:
    """
    Loads text content from a PDF file using PyPDFLoader with fallback to pdfplumber or pypdf.

    Args:
        pdf_path: Path to the target PDF file.

    Returns:
        List of LangChain Document objects representing pages.
    """
    logger.info(f"Loading document from {pdf_path}...")
    documents: List[Document] = []

    # Attempt 1: PyPDFLoader from langchain_community
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        logger.info(f"PyPDFLoader successfully loaded {len(documents)} pages.")
    except Exception as e:
        logger.warning(f"PyPDFLoader encountered an error: {e}. Trying pdfplumber fallback...")

    # Attempt 2: pdfplumber fallback if PyPDFLoader yields no docs
    if not documents:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": settings.PDF_FILENAME,
                                    "page": idx + 1
                                }
                            )
                        )
            logger.info(f"pdfplumber successfully loaded {len(documents)} pages.")
        except Exception as e:
            logger.warning(f"pdfplumber fallback failed: {e}. Trying pypdf raw fallback...")

    # Attempt 3: Direct pypdf fallback
    if not documents:
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": settings.PDF_FILENAME,
                                "page": idx + 1
                            }
                        )
                    )
            logger.info(f"pypdf direct reader loaded {len(documents)} pages.")
        except Exception as e:
            logger.error(f"All PDF loading methods failed for {pdf_path}: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {e}") from e

    # Normalize page numbers to 1-indexed integers
    for doc in documents:
        raw_page = doc.metadata.get("page", 0)
        # PyPDFLoader usually uses 0-indexed page numbers in metadata
        if isinstance(raw_page, int) and raw_page >= 0:
            doc.metadata["page"] = raw_page if raw_page > 0 else raw_page + 1
        doc.metadata["source"] = settings.PDF_FILENAME

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits document pages into smaller semantic chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of raw document pages.

    Returns:
        List of chunked Document objects.
    """
    logger.info(f"Chunking documents (chunk_size={settings.CHUNK_SIZE}, chunk_overlap={settings.CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Assign clean unique chunk IDs in metadata
    for idx, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_id"] = f"page_{page}_chunk_{idx}"

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} document pages.")
    return chunks


class FallbackHashEmbeddings:
    """
    Fallback deterministic bag-of-words hash embedding function.
    Guarantees the pipeline runs even without external model weights or internet access.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        import numpy as np
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            h = abs(hash(word)) % self.dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedding_function():
    """
    Instantiates and returns the configured embedding function with automatic fallback.

    Returns:
        LangChain Embeddings model instance.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "openai" and settings.OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info("Using OpenAI Embeddings (text-embedding-3-small)...")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.OPENAI_API_KEY
            )
        except Exception as e:
            logger.warning(f"Failed to load OpenAIEmbeddings ({e}). Falling back...")

    if provider == "google" and settings.GOOGLE_API_KEY:
        # Try google.genai SDK first (supports current API keys)
        try:
            import google.genai as genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            embed_model = "gemini-embedding-001"

            class GoogleGenAIEmbeddings:
                """Embedding wrapper using google.genai Client."""
                def embed_documents(self, texts):
                    result = []
                    for text in texts:
                        resp = client.models.embed_content(
                            model=embed_model,
                            contents=text,
                        )
                        result.append(resp.embeddings[0].values)
                    return result

                def embed_query(self, text):
                    resp = client.models.embed_content(
                        model=embed_model,
                        contents=text,
                    )
                    return resp.embeddings[0].values

            # Quick validation test
            test = GoogleGenAIEmbeddings()
            test.embed_query("test")
            logger.info(f"Using Google GenAI Embeddings via google.genai SDK ({embed_model})...")
            return GoogleGenAIEmbeddings()
        except Exception as e:
            logger.warning(f"google.genai embeddings failed ({e}). Trying langchain_google_genai...")

        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            logger.info("Using Google Generative AI Embeddings (models/text-embedding-004)...")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=settings.GOOGLE_API_KEY
            )
        except Exception as e:
            logger.warning(f"langchain_google_genai embeddings failed ({e}). Falling back...")


    # Attempt HuggingFace local embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Using HuggingFace Embeddings ({settings.EMBEDDING_MODEL})...")
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e1:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info(f"Using HuggingFace Embeddings via community ({settings.EMBEDDING_MODEL})...")
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e2:
            logger.warning(f"HuggingFace Embeddings load failed ({e2}). Using zero-dependency FallbackHashEmbeddings...")
            return FallbackHashEmbeddings()



def build_vector_store(chunks: List[Document], embedding_function=None):
    """
    Indexes document chunks into the configured vector database (ChromaDB or Pinecone).

    Args:
        chunks: List of chunked Document objects.
        embedding_function: Optional pre-initialized embeddings model.

    Returns:
        VectorStore instance.
    """
    if embedding_function is None:
        embedding_function = get_embedding_function()

    store_type = settings.VECTOR_STORE_TYPE.lower()

    if store_type == "pinecone" and settings.PINECONE_API_KEY:
        logger.info(f"Indexing {len(chunks)} chunks into Pinecone index: {settings.PINECONE_INDEX_NAME}...")
        try:
            from langchain_pinecone import PineconeVectorStore
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embedding_function,
                index_name=settings.PINECONE_INDEX_NAME,
                pinecone_api_key=settings.PINECONE_API_KEY
            )
            logger.info("Pinecone indexing complete.")
            return vectorstore
        except Exception as e:
            logger.warning(f"Pinecone vector store initialization failed: {e}. Falling back to ChromaDB...")

    # Default: ChromaDB persistent vector database
    logger.info(f"Indexing {len(chunks)} chunks into ChromaDB at {settings.CHROMA_PERSIST_DIR}...")
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=settings.COLLECTION_NAME,
        persist_directory=str(settings.CHROMA_PERSIST_DIR)
    )
    logger.info("ChromaDB vector store successfully built and persisted.")
    return vectorstore


def ingest_pdf(force_redownload: bool = False) -> int:
    """
    Complete ingestion pipeline: download -> load -> chunk -> vector store index.

    Args:
        force_redownload: If True, forces redownloading the PDF even if already present.

    Returns:
        Total number of document chunks indexed into the vector database.
    """
    logger.info("Starting RAG document ingestion pipeline...")

    # Step 1: Ensure PDF is present locally
    pdf_file = download_pdf(settings.PDF_URL, settings.PDF_PATH)

    # Step 2: Load document text
    documents = load_pdf_documents(pdf_file)

    # Step 3: Chunk document into semantic snippets
    chunks = chunk_documents(documents)

    # Step 4: Index into vector store
    build_vector_store(chunks)

    logger.info(f"Ingestion pipeline completed successfully! Total indexed chunks: {len(chunks)}")
    return len(chunks)


if __name__ == "__main__":
    ingest_pdf()
