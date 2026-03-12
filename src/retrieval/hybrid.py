"""
Hybrid Retrieval — blends BM25 keyword search and vector semantic search.
Normalizes scores and combines with configurable weights.
"""

from typing import Dict, List

from src.retrieval.bm25_search import BM25Search
from src.retrieval.vector_store import SearchResult, VectorStore


class HybridRetriever:
    """
    Combines vector search and BM25 search with weighted score fusion.

    The hybrid approach handles both:
    - Semantic similarity (vector search catches meaning/intent)
    - Exact keyword matching (BM25 catches specific terms/phrases)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_search: BM25Search,
        alpha: float = 0.6,  # Weight for vector search (1-alpha for BM25)
    ):
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.alpha = alpha

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """
        Perform hybrid retrieval: vector + BM25 with score fusion.

        Args:
            query: User's search query
            top_k: Number of results to return

        Returns:
            Merged, deduplicated results sorted by combined score
        """
        # Run both searches
        vector_results = self.vector_store.search(query, top_k=top_k)
        bm25_results = self.bm25_search.search(query, top_k=top_k)

        # Normalize scores to [0, 1] range
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)

        # Merge results with weighted scores
        merged = self._merge_results(vector_results, bm25_results)

        # Sort by combined score descending and return top-k
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Min-max normalize scores to [0, 1] range."""
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores identical — set to 1.0
            for r in results:
                r.score = 1.0
            return results

        for r in results:
            r.score = (r.score - min_score) / (max_score - min_score)

        return results

    def _merge_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Merge and deduplicate results from both search methods.
        Combines scores with configurable weights.
        """
        # Build lookup by chunk_id
        merged: Dict[str, SearchResult] = {}

        for r in vector_results:
            merged[r.chunk_id] = SearchResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=self.alpha * r.score,  # Weighted vector score
                source=r.source,
                page_number=r.page_number,
                section_heading=r.section_heading,
                metadata={**(r.metadata or {}), "vector_score": r.score},
            )

        for r in bm25_results:
            bm25_weighted = (1 - self.alpha) * r.score

            if r.chunk_id in merged:
                # Chunk found in both — add BM25 score
                existing = merged[r.chunk_id]
                existing.score += bm25_weighted
                if existing.metadata:
                    existing.metadata["bm25_score"] = r.score
            else:
                # BM25-only result
                merged[r.chunk_id] = SearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=bm25_weighted,
                    source=r.source,
                    page_number=r.page_number,
                    section_heading=r.section_heading,
                    metadata={**(r.metadata or {}), "bm25_score": r.score},
                )

        return list(merged.values())
