"""
Token-Aware Chunker — splits documents into 500-800 token chunks with ~100 token overlap.
Respects sentence boundaries to avoid splitting mid-sentence.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken

from src.ingestion.loader import Document


@dataclass
class Chunk:
    """Represents a chunk of a document with metadata."""
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


class TokenAwareChunker:
    """
    Splits documents into token-aware chunks with overlap.
    Respects sentence boundaries for clean splits.
    """

    def __init__(
        self,
        target_tokens: int = 600,
        max_tokens: int = 800,
        min_tokens: int = 100,
        overlap_tokens: int = 100,
        tokenizer_name: str = "cl100k_base",
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Split on sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
        # Filter out empty strings
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document into overlapping chunks respecting sentence boundaries."""
        sentences = self._split_into_sentences(document.text)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If a single sentence exceeds max_tokens, force-split it
            if sentence_tokens > self.max_tokens:
                # Flush current buffer first
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(self._create_chunk(
                        text=chunk_text,
                        document=document,
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
                    current_sentences = []
                    current_tokens = 0

                # Force-split the long sentence by tokens
                forced_chunks = self._force_split(sentence, document, chunk_index)
                chunks.extend(forced_chunks)
                chunk_index += len(forced_chunks)
                continue

            # Would adding this sentence exceed target?
            if current_tokens + sentence_tokens > self.target_tokens and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    document=document,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

                # Create overlap: keep last sentences up to overlap_tokens
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens > self.overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if self.count_tokens(chunk_text) >= self.min_tokens or not chunks:
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    document=document,
                    chunk_index=chunk_index,
                ))
            elif chunks:
                # Append short remainder to last chunk
                last_chunk = chunks[-1]
                combined = last_chunk.text + " " + chunk_text
                if self.count_tokens(combined) <= self.max_tokens:
                    chunks[-1] = self._create_chunk(
                        text=combined,
                        document=document,
                        chunk_index=last_chunk.chunk_index,
                    )

        return chunks

    def _create_chunk(self, text: str, document: Document, chunk_index: int) -> Chunk:
        """Create a Chunk object with metadata from the parent document."""
        source_base = document.source.replace(" ", "_").replace(".", "_")
        chunk_id = f"{source_base}_chunk_{chunk_index}"

        return Chunk(
            chunk_id=chunk_id,
            text=text,
            source=document.source,
            chunk_index=chunk_index,
            page_number=document.page_number,
            section_heading=document.section_heading,
            token_count=self.count_tokens(text),
            metadata={
                **document.metadata,
                "chunk_index": chunk_index,
            }
        )

    def _force_split(self, text: str, document: Document, start_index: int) -> List[Chunk]:
        """Force-split a very long text into target-sized token chunks."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        idx = start_index

        for i in range(0, len(tokens), self.target_tokens - self.overlap_tokens):
            chunk_tokens = tokens[i:i + self.target_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(self._create_chunk(
                text=chunk_text,
                document=document,
                chunk_index=idx,
            ))
            idx += 1

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents, ensuring unique chunk IDs per source."""
        all_chunks = []
        # Track chunk index per source file to avoid duplicate IDs
        # (MarkdownLoader creates multiple Documents from one file)
        source_counters: dict = {}
        for doc in documents:
            chunks = self.chunk_document(doc)
            # Re-index chunks with global counter per source
            source_key = doc.source
            if source_key not in source_counters:
                source_counters[source_key] = 0
            for chunk in chunks:
                chunk.chunk_index = source_counters[source_key]
                source_base = doc.source.replace(" ", "_").replace(".", "_")
                chunk.chunk_id = f"{source_base}_chunk_{chunk.chunk_index}"
                source_counters[source_key] += 1
            all_chunks.extend(chunks)
        return all_chunks
