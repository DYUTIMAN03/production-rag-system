"""
LLM Client — Groq integration with Llama 3.3 70B.
Handles generation, streaming, token counting, rate limiting, and retries.
"""

import os
import time
from dataclasses import dataclass
from typing import Generator, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    latency_ms: float


class GroqLLM:
    """
    Groq LLM client — ultra-fast inference (free tier).

    Free tier limits:
    - 30 requests per minute
    - 14,400 requests per day
    - 6,000 tokens per minute
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in .env or pass directly. "
                "Get a free key at: https://console.groq.com/keys"
            )

        self.client = Groq(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """
        Generate a response using Groq.

        Args:
            system_prompt: System instructions for the model
            user_prompt: The user's query with context

        Returns:
            LLMResponse with text, token counts, and latency
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    stream=False,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract token usage
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                return LLMResponse(
                    text=response.choices[0].message.content or "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    model=self.model,
                    latency_ms=round(latency_ms, 2),
                )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff for rate limiting
                    wait_time = 2 ** (attempt + 1)
                    print(f"LLM request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"LLM generation failed after {self.max_retries} attempts: {e}")

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Generator[str, None, LLMResponse]:
        """
        Stream a response from Groq token-by-token.

        Yields:
            Token chunks as strings

        Returns:
            LLMResponse with final token counts (accessible after iteration)
        """
        start_time = time.time()

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_text += token
                yield token

        latency_ms = (time.time() - start_time) * 1000

        # Store final response data as an attribute for the caller to access
        self._last_stream_response = LLMResponse(
            text=full_text,
            input_tokens=0,  # Not available in streaming mode
            output_tokens=0,
            total_tokens=0,
            model=self.model,
            latency_ms=round(latency_ms, 2),
        )
