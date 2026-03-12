"""
RAG Pipeline — orchestrates the full retrieval-augmented generation flow:
query rewrite → hybrid retrieve → rerank → citation enforcement → generate.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional

from src.generation.llm import GroqLLM, LLMResponse
from src.generation.prompt_manager import PromptManager
from src.generation.query_rewriter import QueryRewriter
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import SearchResult


@dataclass
class Citation:
    """A citation linking an answer claim to source material."""
    source: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_text: str = ""
    relevance_score: float = 0.0


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    citations: List[Citation]
    chunks_used: int
    confidence_score: float
    is_grounded: bool  # True if answer is supported by evidence
    metrics: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Full RAG pipeline with query rewriting, hybrid retrieval,
    cross-encoder reranking, and citation enforcement.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        llm: GroqLLM,
        prompt_manager: PromptManager,
        query_rewriter: Optional[QueryRewriter] = None,
        initial_top_k: int = 20,
        final_top_k: int = 5,
        reranker_threshold: float = 0.3,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.query_rewriter = query_rewriter
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        self.reranker_threshold = reranker_threshold

    def query(self, question: str) -> RAGResponse:
        """
        Process a user query through the full RAG pipeline.

        Steps:
        0. Query rewriting (LLM-based)
        1. Hybrid retrieval (vector + BM25)
        2. Cross-encoder re-ranking
        3. Citation enforcement check
        4. LLM generation with context

        Args:
            question: User's natural language question

        Returns:
            RAGResponse with answer, citations, and metrics
        """
        total_start = time.time()
        metrics = {}

        # Step 0: Query Rewriting
        search_query = question
        if self.query_rewriter:
            rewrite_result = self.query_rewriter.rewrite(question)
            search_query = rewrite_result.rewritten_query
            metrics["query_rewrite_latency_ms"] = rewrite_result.latency_ms
            metrics["original_query"] = rewrite_result.original_query
            metrics["rewritten_query"] = rewrite_result.rewritten_query

        # Step 1: Hybrid Retrieval
        retrieval_start = time.time()
        initial_results = self.hybrid_retriever.search(
            query=search_query,
            top_k=self.initial_top_k,
        )
        metrics["retrieval_latency_ms"] = round((time.time() - retrieval_start) * 1000, 2)
        metrics["initial_candidates"] = len(initial_results)

        # Step 2: Re-Ranking
        rerank_start = time.time()
        reranked_results = self.reranker.rerank(
            query=question,  # Use original question for reranking
            results=initial_results,
            top_k=self.final_top_k,
        )
        metrics["rerank_latency_ms"] = round((time.time() - rerank_start) * 1000, 2)
        metrics["reranked_count"] = len(reranked_results)

        # Get reranker score stats for observability
        score_stats = self.reranker.get_score_stats(reranked_results)
        metrics["reranker_scores"] = score_stats

        # Step 3: Citation Enforcement
        top_score = reranked_results[0].score if reranked_results else 0
        is_grounded = top_score >= self.reranker_threshold

        if not is_grounded:
            # Evidence insufficient — refuse to answer
            refusal = self.prompt_manager.format_prompt(
                "rag_refusal",
                threshold=self.reranker_threshold,
            )
            metrics["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)
            metrics["citation_grounded"] = False

            return RAGResponse(
                answer=refusal,
                citations=[],
                chunks_used=0,
                confidence_score=top_score,
                is_grounded=False,
                metrics=metrics,
            )

        # Step 4: Build context and generate answer
        context = self._build_context(reranked_results)
        citations = self._build_citations(reranked_results)

        # Get prompts
        system_prompt = self.prompt_manager.get_template("rag_system")
        user_prompt = self.prompt_manager.format_prompt(
            "rag_query",
            context=context,
            question=question,
        )

        # Generate
        generation_start = time.time()
        llm_response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        metrics["generation_latency_ms"] = round((time.time() - generation_start) * 1000, 2)
        metrics["input_tokens"] = llm_response.input_tokens
        metrics["output_tokens"] = llm_response.output_tokens
        metrics["total_tokens"] = llm_response.total_tokens
        metrics["llm_model"] = llm_response.model
        metrics["citation_grounded"] = True
        metrics["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)

        # Record prompt versions used
        metrics["prompt_versions"] = self.prompt_manager.get_all_versions()

        return RAGResponse(
            answer=llm_response.text,
            citations=citations,
            chunks_used=len(reranked_results),
            confidence_score=top_score,
            is_grounded=True,
            metrics=metrics,
        )

    def query_stream(self, question: str) -> Generator[str, None, None]:
        """
        Stream a RAG response — retrieval + reranking runs normally,
        then LLM generation is streamed token-by-token as SSE events.

        Yields:
            SSE-formatted strings: metadata event, token chunks, done event
        """
        total_start = time.time()
        metrics = {}

        # Step 0: Query Rewriting
        search_query = question
        if self.query_rewriter:
            rewrite_result = self.query_rewriter.rewrite(question)
            search_query = rewrite_result.rewritten_query
            metrics["rewritten_query"] = rewrite_result.rewritten_query

        # Step 1: Hybrid Retrieval
        initial_results = self.hybrid_retriever.search(
            query=search_query,
            top_k=self.initial_top_k,
        )

        # Step 2: Re-Ranking
        reranked_results = self.reranker.rerank(
            query=question,
            results=initial_results,
            top_k=self.final_top_k,
        )

        # Step 3: Citation Enforcement
        top_score = reranked_results[0].score if reranked_results else 0
        is_grounded = top_score >= self.reranker_threshold
        citations = self._build_citations(reranked_results) if is_grounded else []

        # Send metadata event first
        metadata = {
            "type": "metadata",
            "confidence_score": top_score,
            "is_grounded": is_grounded,
            "chunks_used": len(reranked_results) if is_grounded else 0,
            "citations": [
                {
                    "source": c.source,
                    "page_number": c.page_number,
                    "section_heading": c.section_heading,
                    "chunk_text": c.chunk_text,
                    "relevance_score": c.relevance_score,
                }
                for c in citations
            ],
            "rewritten_query": metrics.get("rewritten_query", question),
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        if not is_grounded:
            refusal = self.prompt_manager.format_prompt(
                "rag_refusal", threshold=self.reranker_threshold,
            )
            yield f"data: {json.dumps({'type': 'token', 'content': refusal})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Step 4: Stream generation
        context = self._build_context(reranked_results)
        system_prompt = self.prompt_manager.get_template("rag_system")
        user_prompt = self.prompt_manager.format_prompt(
            "rag_query", context=context, question=question,
        )

        for token in self.llm.generate_stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Done
        total_ms = round((time.time() - total_start) * 1000, 2)
        yield f"data: {json.dumps({'type': 'done', 'total_latency_ms': total_ms})}\n\n"

    def _build_context(self, results: List[SearchResult]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        context_parts = []
        for i, result in enumerate(results, 1):
            source_info = f"[Source: {result.source}"
            if result.page_number and result.page_number > 0:
                source_info += f", Page {result.page_number}"
            if result.section_heading:
                source_info += f", Section: {result.section_heading}"
            source_info += "]"

            context_parts.append(
                f"Passage {i} {source_info}:\n{result.text}"
            )

        return "\n\n".join(context_parts)

    def _build_citations(self, results: List[SearchResult]) -> List[Citation]:
        """Build citation objects from reranked results."""
        return [
            Citation(
                source=r.source,
                page_number=r.page_number if r.page_number and r.page_number > 0 else None,
                section_heading=r.section_heading,
                chunk_text=r.text[:200] + "..." if len(r.text) > 200 else r.text,
                relevance_score=r.score,
            )
            for r in results
        ]
