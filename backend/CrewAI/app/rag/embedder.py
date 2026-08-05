"""Sentence Transformer embedding model for medical text."""

from sentence_transformers import SentenceTransformer
from typing import Any
import numpy as np
import logging
from ..config import settings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Wrapper for SentenceTransformer embedding model."""

    def __init__(self, model_name: str = None):
        """Initialize the embedder."""
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = np.array(self.embed_text(text1))
        emb2 = np.array(self.embed_text(text2))
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
