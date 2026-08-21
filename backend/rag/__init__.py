"""
Retrieval-Augmented Generation (RAG) module.

Responsible for knowledge retrieval: document ingestion, embedding
generation, vector search, and context retrieval. The pipeline is the
reusable entry point for CrewAI, FastAPI, and examples; embedding and
storage backends are swappable behind narrow interfaces.
"""

from .chunker import TextChunker
from .config import RAGSettings, settings
from .corpus import BUNDLED_CORPUS_DIR, load_bundled_corpus, load_documents
from .documents import Chunk, Document, RetrievalResult
from .embedder import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
    build_embedder,
)
from .metrics import (
    RAGQualityMetrics,
    RetrievalMetrics,
    rag_quality_metrics,
    retrieval_metrics,
)
from .pipeline import RAGPipeline
from .retriever import Retriever
from .store import VectorStore, build_vector_store

__all__ = [
    "BUNDLED_CORPUS_DIR",
    "Chunk",
    "Document",
    "Embedder",
    "HashingEmbedder",
    "RAGPipeline",
    "RAGQualityMetrics",
    "RAGSettings",
    "RetrievalMetrics",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEmbedder",
    "TextChunker",
    "TfidfEmbedder",
    "VectorStore",
    "build_embedder",
    "build_vector_store",
    "load_bundled_corpus",
    "load_documents",
    "rag_quality_metrics",
    "retrieval_metrics",
    "settings",
]
