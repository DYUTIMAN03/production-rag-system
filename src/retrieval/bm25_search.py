"""
BM25 Keyword Search — complements vector search for exact term matching.
Uses rank_bm25 library with tokenized document chunks.
"""

import os
import pickle
import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import SearchResult


class BM25Search:
    """BM25 keyword-based search over document chunks."""

    def __init__(self, index_path: str = "./bm25_index.pkl"):
        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_data: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def build_index(self, chunks: List[dict]):
        """
        Build BM25 index from chunk data.

        Args:
            chunks: List of dicts with 'chunk_id', 'text', 'metadata' keys
        """
        self.chunk_data = chunks
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Search for the top-k most relevant chunks using BM25 scoring."""
        if not self.bm25 or not self.chunk_data:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score (descending)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:top_k]

        results = []
        for idx, score in top_indices:
            if score <= 0:
                continue

            chunk = self.chunk_data[idx]
            meta = chunk.get("metadata", {})

            results.append(SearchResult(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                score=score,
                source=meta.get("source", ""),
                page_number=meta.get("page_number"),
                section_heading=meta.get("section_heading"),
                metadata=meta,
            ))

        return results

    def save_index(self):
        """Persist the BM25 index to disk."""
        data = {
            "chunk_data": self.chunk_data,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def load_index(self) -> bool:
        """Load a previously saved BM25 index. Returns True if successful."""
        if not os.path.exists(self.index_path):
            return False

        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)

            self.chunk_data = data["chunk_data"]
            self.tokenized_corpus = data["tokenized_corpus"]

            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)

            return True
        except Exception:
            return False
