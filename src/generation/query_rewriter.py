"""
Query Rewriter — uses the LLM to rewrite vague user queries
into optimized retrieval queries for better document matching.
"""

from dataclasses import dataclass
from typing import Optional

from src.generation.llm import GroqLLM, LLMResponse
from src.generation.prompt_manager import PromptManager


@dataclass
class RewrittenQuery:
    """Result of query rewriting."""
    original_query: str
    rewritten_query: str
    latency_ms: float


class QueryRewriter:
    """
    Rewrites user queries for better retrieval performance.

    User queries are often casual/vague:
      "how does attention work?"
    A rewritten query is optimized for document retrieval:
      "multi-head self-attention mechanism in transformer architecture"

    This bridges the vocabulary gap between user language and document language.
    """

    def __init__(
        self,
        llm: GroqLLM,
        prompt_manager: PromptManager,
        enabled: bool = True,
    ):
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.enabled = enabled

    def rewrite(self, query: str) -> RewrittenQuery:
        """
        Rewrite a user query into an optimized retrieval query.

        Args:
            query: The user's original natural language question

        Returns:
            RewrittenQuery with both original and rewritten versions
        """
        if not self.enabled:
            return RewrittenQuery(
                original_query=query,
                rewritten_query=query,
                latency_ms=0.0,
            )

        try:
            system_prompt = self.prompt_manager.get_template("query_rewrite_system")
            user_prompt = self.prompt_manager.format_prompt(
                "query_rewrite",
                question=query,
            )

            response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            # Clean up the rewritten query — strip quotes, whitespace
            rewritten = response.text.strip().strip('"').strip("'").strip()

            # Fallback: if rewriting produced empty or very short result, use original
            if len(rewritten) < 5:
                rewritten = query

            return RewrittenQuery(
                original_query=query,
                rewritten_query=rewritten,
                latency_ms=response.latency_ms,
            )

        except Exception as e:
            # Graceful degradation — if rewriting fails, use the original query
            print(f"[!] Query rewriting failed: {e}. Using original query.")
            return RewrittenQuery(
                original_query=query,
                rewritten_query=query,
                latency_ms=0.0,
            )
