"""
Cross-Encoder Re-Ranker — reranks retrieved chunks using ms-marco-MiniLM-L6-v2.
Evaluates query-chunk pairs jointly for dramatically improved precision.
"""

import math
from typing import List

from sentence_transformers import CrossEncoder

from src.retrieval.vector_store import SearchResult


class Reranker:
    """
    Cross-encoder re-ranker that scores (query, chunk) pairs together.

    Unlike bi-encoder embeddings that encode query and chunks independently,
    a cross-encoder processes them as a pair — consistently producing better
    relevance scores.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Re-rank search results using the cross-encoder.

        Args:
            query: User's search query
            results: Initial search results from hybrid retrieval
            top_k: Number of top results to return after re-ranking

        Returns:
            Re-ranked results with updated scores, sorted by relevance
        """
        if not results:
            return []

        # Create (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, r.text) for r in results]

        # Score all pairs
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Convert raw logits to 0-1 probabilities via sigmoid
        # ms-marco cross-encoders output raw logits (range ~-11 to +11)
        scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

        # Update results with re-ranker scores
        reranked = []
        for result, score in zip(results, scores):
            reranked_result = SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(score),
                source=result.source,
                page_number=result.page_number,
                section_heading=result.section_heading,
                metadata={
                    **(result.metadata or {}),
                    "reranker_score": float(score),
                    "original_hybrid_score": result.score,
                },
            )
            reranked.append(reranked_result)

        # Sort by reranker score descending
        reranked.sort(key=lambda r: r.score, reverse=True)

        return reranked[:top_k]

    def get_score_stats(self, results: List[SearchResult]) -> dict:
        """Get statistics about re-ranker scores for observability."""
        if not results:
            return {"min": 0, "max": 0, "mean": 0, "count": 0}

        scores = [r.score for r in results]
        return {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(sum(scores) / len(scores), 4),
            "count": len(scores),
        }
