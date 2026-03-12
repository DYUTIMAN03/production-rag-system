"""
Langfuse Tracer — instruments the RAG pipeline for full observability.
Traces every request: retrieved chunks, prompts sent, responses, token usage.
"""

import os
import functools
from typing import Any, Callable, Optional

from dotenv import load_dotenv

load_dotenv()


class LangfuseTracer:
    """
    Langfuse integration for tracing RAG pipeline requests.

    From Day 1, every query is traceable:
    - Which chunks were retrieved
    - What prompt was sent to the LLM
    - What the response was
    - How many tokens were consumed
    """

    def __init__(self):
        self.enabled = False
        self.langfuse = None
        self._init_langfuse()

    def _init_langfuse(self):
        """Initialize Langfuse client if credentials are available."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

        if public_key and secret_key:
            try:
                from langfuse import Langfuse
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                self.langfuse.auth_check()
                self.enabled = True
                print(f"  [+] Langfuse tracing enabled (host: {host})")
            except Exception as e:
                print(f"  [!] Langfuse initialization failed: {e}. Tracing disabled.")
                self.enabled = False
        else:
            print("  [!] Langfuse credentials not found. Tracing disabled. "
                  "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")

    def create_trace(self, name: str, metadata: dict = None, user_id: str = None):
        """Create a new trace for a pipeline execution."""
        if not self.enabled:
            return DummyTrace()

        try:
            return self.langfuse.trace(
                name=name,
                metadata=metadata or {},
                user_id=user_id,
            )
        except Exception as e:
            print(f"  [!] Failed to create trace: {e}")
            return DummyTrace()

    def trace_retrieval(self, trace, query: str, results: list, search_type: str = "hybrid"):
        """Log retrieval step to a trace."""
        if not self.enabled or isinstance(trace, DummyTrace):
            return

        try:
            trace.span(
                name=f"retrieval_{search_type}",
                input={"query": query},
                output={
                    "num_results": len(results),
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "source": r.source,
                            "score": round(r.score, 4),
                            "text_preview": r.text[:150] + "...",
                        }
                        for r in results[:10]  # Log top 10
                    ]
                },
                metadata={"search_type": search_type},
            )
        except Exception as e:
            print(f"  [!] Trace retrieval failed: {e}")

    def trace_reranking(self, trace, query: str, results: list, score_stats: dict):
        """Log reranking step to a trace."""
        if not self.enabled or isinstance(trace, DummyTrace):
            return

        try:
            trace.span(
                name="reranking",
                input={"query": query, "num_candidates": len(results)},
                output={
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "reranker_score": round(r.score, 4),
                            "source": r.source,
                        }
                        for r in results
                    ],
                    "score_stats": score_stats,
                },
            )
        except Exception as e:
            print(f"  [!] Trace reranking failed: {e}")

    def trace_generation(
        self,
        trace,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        token_usage: dict,
        prompt_versions: dict,
    ):
        """Log LLM generation step to a trace."""
        if not self.enabled or isinstance(trace, DummyTrace):
            return

        try:
            trace.generation(
                name="llm_generation",
                input=user_prompt,
                output=response_text,
                model=token_usage.get("model", "unknown"),
                usage={
                    "input": token_usage.get("input_tokens", 0),
                    "output": token_usage.get("output_tokens", 0),
                    "total": token_usage.get("total_tokens", 0),
                },
                metadata={
                    "system_prompt": system_prompt[:200] + "...",
                    "prompt_versions": prompt_versions,
                },
            )
        except Exception as e:
            print(f"  [!] Trace generation failed: {e}")

    def trace_pipeline_result(self, trace, rag_response):
        """Log the final pipeline result with scores."""
        if not self.enabled or isinstance(trace, DummyTrace):
            return

        try:
            # Record quality scores on the trace
            self.langfuse.score(
                trace_id=trace.id,
                name="confidence",
                value=rag_response.confidence_score,
            )
            self.langfuse.score(
                trace_id=trace.id,
                name="is_grounded",
                value=1.0 if rag_response.is_grounded else 0.0,
            )
            self.langfuse.score(
                trace_id=trace.id,
                name="citation_count",
                value=float(len(rag_response.citations)),
            )

            # Record latency metrics
            metrics = rag_response.metrics
            if "total_latency_ms" in metrics:
                self.langfuse.score(
                    trace_id=trace.id,
                    name="total_latency_ms",
                    value=metrics["total_latency_ms"],
                )
        except Exception as e:
            print(f"  [!] Trace pipeline result failed: {e}")

    def flush(self):
        """Flush any pending traces to Langfuse."""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception:
                pass


class DummyTrace:
    """No-op trace when Langfuse is disabled."""

    id = "dummy"

    def span(self, **kwargs):
        return self

    def generation(self, **kwargs):
        return self

    def score(self, **kwargs):
        return self

    def update(self, **kwargs):
        return self

    def end(self, **kwargs):
        return self
