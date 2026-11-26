"""Semantic embedding service for hybrid RAG retrieval across scenes and user queries."""

import re
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class EmbeddingService:
    """Computes dense semantic embeddings for scene descriptions, transcripts, and natural language prompts."""

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.svd: Optional[TruncatedSVD] = None
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Fit vocabulary and latent semantic space across available scene corpus."""
        if not texts or len(texts) < 2:
            return
        clean_texts = [self._preprocess(t) for t in texts]
        tfidf = self.vectorizer.fit_transform(clean_texts)
        n_comp = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if n_comp < 2:
            n_comp = 2
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self.is_fitted = True

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text or list of texts into normalized dense semantic vectors."""
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else texts
        clean_texts = [self._preprocess(t) for t in text_list]

        if not self.is_fitted or self.svd is None:
            # Fallback uniform deterministic embedding
            dummy = np.zeros((len(text_list), self.n_components), dtype=np.float32)
            for i, t in enumerate(clean_texts):
                val = (hash(t) % 1000) / 1000.0
                dummy[i, :] = val
            return dummy[0] if single_input else dummy

        tfidf = self.vectorizer.transform(clean_texts)
        dense = self.svd.transform(tfidf)
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        normalized = dense / norms
        return normalized[0] if single_input else normalized

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two normalized vectors."""
        v1_flat = np.asarray(v1).flatten()
        v2_flat = np.asarray(v2).flatten()
        norm1 = np.linalg.norm(v1_flat) + 1e-8
        norm2 = np.linalg.norm(v2_flat) + 1e-8
        return float(np.dot(v1_flat, v2_flat) / (norm1 * norm2))

    @staticmethod
    def _preprocess(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text).lower().strip())
