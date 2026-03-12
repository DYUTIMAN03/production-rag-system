"""
Metrics Collector — tracks latency (P50/P95), cost, citation coverage,
failure rate, and reranker score distribution for observability.
"""

import time
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: float
    total_latency_ms: float
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    is_grounded: bool = True
    citation_count: int = 0
    confidence_score: float = 0.0
    reranker_max_score: float = 0.0
    reranker_mean_score: float = 0.0
    is_error: bool = False
    error_message: str = ""


class MetricsCollector:
    """
    Collects and aggregates per-request metrics for the RAG pipeline.

    Tracks:
    - Latency at P50 and P95 (not averages — averages hide worst-case)
    - Cost per request in dollar terms
    - Citation coverage: % of answers grounded in evidence
    - Failure rate: errors + unsupported responses
    - Reranker score distribution
    """

    # Groq free tier — effectively $0, but we track token counts
    # for cost estimation if you switch to a paid model
    COST_PER_INPUT_TOKEN = 0.0
    COST_PER_OUTPUT_TOKEN = 0.0

    def __init__(self, max_history: int = 10000):
        self._metrics: deque = deque(maxlen=max_history)

    def record(self, rag_response) -> RequestMetrics:
        """
        Record metrics from a RAG pipeline response.

        Args:
            rag_response: RAGResponse from the pipeline

        Returns:
            RequestMetrics for this request
        """
        m = rag_response.metrics
        reranker_scores = m.get("reranker_scores", {})

        request_metrics = RequestMetrics(
            timestamp=time.time(),
            total_latency_ms=m.get("total_latency_ms", 0),
            retrieval_latency_ms=m.get("retrieval_latency_ms", 0),
            rerank_latency_ms=m.get("rerank_latency_ms", 0),
            generation_latency_ms=m.get("generation_latency_ms", 0),
            input_tokens=m.get("input_tokens", 0),
            output_tokens=m.get("output_tokens", 0),
            total_tokens=m.get("total_tokens", 0),
            is_grounded=rag_response.is_grounded,
            citation_count=len(rag_response.citations),
            confidence_score=rag_response.confidence_score,
            reranker_max_score=reranker_scores.get("max", 0),
            reranker_mean_score=reranker_scores.get("mean", 0),
        )

        self._metrics.append(request_metrics)
        return request_metrics

    def record_error(self, error_message: str):
        """Record a failed request."""
        self._metrics.append(RequestMetrics(
            timestamp=time.time(),
            total_latency_ms=0,
            is_error=True,
            error_message=error_message,
        ))

    def record_from_stream(
        self,
        confidence_score: float = 0.0,
        is_grounded: bool = False,
        chunks_used: int = 0,
        citation_count: int = 0,
        total_latency_ms: float = 0.0,
        reranker_max_score: float = 0.0,
    ):
        """Record metrics from a streaming response (no RAGResponse object available)."""
        self._metrics.append(RequestMetrics(
            timestamp=time.time(),
            total_latency_ms=total_latency_ms,
            is_grounded=is_grounded,
            citation_count=citation_count,
            confidence_score=confidence_score,
            reranker_max_score=reranker_max_score,
        ))

    def get_summary(self) -> dict:
        """
        Calculate aggregated metrics summary.

        Returns P50/P95 latencies, cost, citation coverage,
        failure rate, and reranker score distribution.
        """
        if not self._metrics:
            return self._empty_summary()

        all_metrics = list(self._metrics)
        total_requests = len(all_metrics)
        errors = [m for m in all_metrics if m.is_error]
        successful = [m for m in all_metrics if not m.is_error]

        # Latency percentiles (P50, P95)
        latencies = sorted([m.total_latency_ms for m in successful]) if successful else [0]
        retrieval_latencies = sorted([m.retrieval_latency_ms for m in successful]) if successful else [0]
        generation_latencies = sorted([m.generation_latency_ms for m in successful]) if successful else [0]

        # Citation coverage
        grounded = sum(1 for m in successful if m.is_grounded)
        citation_coverage = grounded / len(successful) if successful else 0

        # Token totals & cost
        total_input_tokens = sum(m.input_tokens for m in successful)
        total_output_tokens = sum(m.output_tokens for m in successful)
        total_cost = (
            total_input_tokens * self.COST_PER_INPUT_TOKEN +
            total_output_tokens * self.COST_PER_OUTPUT_TOKEN
        )

        # Reranker score distribution
        reranker_max_scores = [m.reranker_max_score for m in successful if m.reranker_max_score > 0]
        reranker_mean_scores = [m.reranker_mean_score for m in successful if m.reranker_mean_score > 0]

        return {
            "total_requests": total_requests,
            "successful_requests": len(successful),
            "error_count": len(errors),
            "failure_rate": round(len(errors) / total_requests, 4) if total_requests else 0,
            "latency": {
                "total": {
                    "p50_ms": round(self._percentile(latencies, 50), 2),
                    "p95_ms": round(self._percentile(latencies, 95), 2),
                    "mean_ms": round(statistics.mean(latencies), 2) if latencies else 0,
                },
                "retrieval": {
                    "p50_ms": round(self._percentile(retrieval_latencies, 50), 2),
                    "p95_ms": round(self._percentile(retrieval_latencies, 95), 2),
                },
                "generation": {
                    "p50_ms": round(self._percentile(generation_latencies, 50), 2),
                    "p95_ms": round(self._percentile(generation_latencies, 95), 2),
                },
            },
            "cost": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost_usd": round(total_cost, 6),
                "avg_cost_per_request_usd": round(
                    total_cost / len(successful), 6
                ) if successful else 0,
            },
            "quality": {
                "citation_coverage": round(citation_coverage, 4),
                "grounded_responses": grounded,
                "ungrounded_responses": len(successful) - grounded,
            },
            "reranker_scores": {
                "max_score_distribution": {
                    "mean": round(statistics.mean(reranker_max_scores), 4) if reranker_max_scores else 0,
                    "min": round(min(reranker_max_scores), 4) if reranker_max_scores else 0,
                    "max": round(max(reranker_max_scores), 4) if reranker_max_scores else 0,
                },
                "mean_score_distribution": {
                    "mean": round(statistics.mean(reranker_mean_scores), 4) if reranker_mean_scores else 0,
                },
            },
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate the given percentile from sorted data."""
        if not data:
            return 0.0
        k = (len(data) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[-1]
        return data[f] + (k - f) * (data[c] - data[f])

    def _empty_summary(self) -> dict:
        """Return empty summary when no metrics have been recorded."""
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "error_count": 0,
            "failure_rate": 0,
            "latency": {
                "total": {"p50_ms": 0, "p95_ms": 0, "mean_ms": 0},
                "retrieval": {"p50_ms": 0, "p95_ms": 0},
                "generation": {"p50_ms": 0, "p95_ms": 0},
            },
            "cost": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0,
                "avg_cost_per_request_usd": 0,
            },
            "quality": {
                "citation_coverage": 0,
                "grounded_responses": 0,
                "ungrounded_responses": 0,
            },
            "reranker_scores": {
                "max_score_distribution": {"mean": 0, "min": 0, "max": 0},
                "mean_score_distribution": {"mean": 0},
            },
        }
