"""Tests for the RAG pipeline — end-to-end integration tests."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.prompt_manager import PromptManager
from src.observability.metrics import MetricsCollector, RequestMetrics


class TestPromptManager:
    """Tests for prompt versioning and management."""

    def setup_method(self):
        self.pm = PromptManager()

    def test_load_prompts(self):
        """Test that prompts load from YAML config."""
        prompt = self.pm.get_prompt("rag_system")
        assert "version" in prompt
        assert "template" in prompt

    def test_get_template(self):
        """Test retrieving a prompt template."""
        template = self.pm.get_template("rag_system")
        assert isinstance(template, str)
        assert len(template) > 10

    def test_format_prompt(self):
        """Test rendering a prompt with variables."""
        formatted = self.pm.format_prompt(
            "rag_query",
            context="Test context here",
            question="What is RAG?"
        )
        assert "Test context here" in formatted
        assert "What is RAG?" in formatted

    def test_get_version(self):
        """Test retrieving prompt version."""
        version = self.pm.get_version("rag_system")
        assert version == "1.0.0"

    def test_get_all_versions(self):
        """Test retrieving all prompt versions."""
        versions = self.pm.get_all_versions()
        assert "rag_system" in versions
        assert "rag_query" in versions
        assert "rag_refusal" in versions

    def test_invalid_prompt_name(self):
        """Test error on invalid prompt name."""
        try:
            self.pm.get_prompt("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestMetricsCollector:
    """Tests for the metrics collection and aggregation."""

    def setup_method(self):
        self.collector = MetricsCollector()

    def test_empty_summary(self):
        """Test summary with no recorded metrics."""
        summary = self.collector.get_summary()
        assert summary["total_requests"] == 0
        assert summary["failure_rate"] == 0

    def test_record_and_summarize(self):
        """Test recording metrics and getting a summary."""
        # Create a mock RAG response
        class MockResponse:
            is_grounded = True
            confidence_score = 0.85
            citations = [None, None]  # 2 citations
            metrics = {
                "total_latency_ms": 1500,
                "retrieval_latency_ms": 200,
                "rerank_latency_ms": 300,
                "generation_latency_ms": 1000,
                "input_tokens": 500,
                "output_tokens": 200,
                "total_tokens": 700,
                "reranker_scores": {"max": 0.9, "mean": 0.65, "min": 0.3},
            }

        self.collector.record(MockResponse())
        summary = self.collector.get_summary()

        assert summary["total_requests"] == 1
        assert summary["successful_requests"] == 1
        assert summary["error_count"] == 0
        assert summary["latency"]["total"]["p50_ms"] > 0
        assert summary["quality"]["citation_coverage"] == 1.0

    def test_failure_rate(self):
        """Test failure rate calculation."""
        self.collector.record_error("Test error")
        self.collector.record_error("Another error")

        summary = self.collector.get_summary()
        assert summary["total_requests"] == 2
        assert summary["error_count"] == 2
        assert summary["failure_rate"] == 1.0

    def test_percentile_calculation(self):
        """Test P50/P95 latency calculations."""
        class MockResponse:
            is_grounded = True
            confidence_score = 0.8
            citations = []
            metrics = {
                "total_latency_ms": 0,
                "retrieval_latency_ms": 0,
                "rerank_latency_ms": 0,
                "generation_latency_ms": 0,
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "reranker_scores": {"max": 0.8, "mean": 0.5, "min": 0.2},
            }

        # Record requests with varying latencies
        for i in range(100):
            mock = MockResponse()
            mock.metrics = dict(mock.metrics)
            mock.metrics["total_latency_ms"] = (i + 1) * 10  # 10ms to 1000ms
            self.collector.record(mock)

        summary = self.collector.get_summary()
        assert summary["latency"]["total"]["p50_ms"] > 0
        assert summary["latency"]["total"]["p95_ms"] > summary["latency"]["total"]["p50_ms"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
