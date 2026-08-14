"""
Retrieval-Augmented Generation (RAG) module.

Responsible for knowledge retrieval: document ingestion, embedding
generation, vector search, and context retrieval. The pipeline is the
reusable entry point for CrewAI, FastAPI, and examples; embedding and
storage backends are swappable behind narrow interfaces.
"""

from .chunker import TextChunker
from .config import RAGSettings, settings
from .documents import Chunk, Document, RetrievalResult
from .embedder import Embedder, HashingEmbedder, TfidfEmbedder, build_embedder
from .metrics import RetrievalMetrics, retrieval_metrics
from .pipeline import RAGPipeline
from .retriever import Retriever
from .store import VectorStore

__all__ = [
    "Chunk",
    "Document",
    "Embedder",
    "HashingEmbedder",
    "RAGPipeline",
    "RAGSettings",
    "RetrievalMetrics",
    "RetrievalResult",
    "Retriever",
    "TextChunker",
    "TfidfEmbedder",
    "VectorStore",
    "build_embedder",
    "retrieval_metrics",
    "settings",
]
