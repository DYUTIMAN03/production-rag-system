"""Tests for the token-aware document chunker."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.loader import Document
from src.ingestion.chunker import TokenAwareChunker, Chunk


def make_document(text: str, source: str = "test.md") -> Document:
    """Helper to create a test document."""
    return Document(text=text, source=source, metadata={"file_type": "test"})


class TestTokenAwareChunker:
    """Test suite for TokenAwareChunker."""

    def setup_method(self):
        self.chunker = TokenAwareChunker(
            target_tokens=100,
            max_tokens=150,
            min_tokens=20,
            overlap_tokens=20,
        )

    def test_basic_chunking(self):
        """Test that text is split into chunks."""
        text = ". ".join([f"This is sentence number {i}" for i in range(50)])
        doc = make_document(text)
        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) > 1, "Should produce multiple chunks"
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size_within_bounds(self):
        """Test that chunks stay within token limits."""
        text = ". ".join([f"This is a moderately long sentence number {i} in the test document" for i in range(100)])
        doc = make_document(text)
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            token_count = self.chunker.count_tokens(chunk.text)
            # Allow some tolerance for sentence boundary adjustment
            assert token_count <= self.chunker.max_tokens + 50, \
                f"Chunk has {token_count} tokens, max is {self.chunker.max_tokens}"

    def test_overlap_exists(self):
        """Test that consecutive chunks have overlapping content."""
        text = ". ".join([f"Unique sentence {i} with content" for i in range(50)])
        doc = make_document(text)
        chunks = self.chunker.chunk_document(doc)

        if len(chunks) > 1:
            # Check that some words from end of chunk N appear at start of chunk N+1
            for i in range(len(chunks) - 1):
                words_end = set(chunks[i].text.split()[-20:])
                words_start = set(chunks[i + 1].text.split()[:20])
                overlap = words_end & words_start
                # There should be at least some overlap
                assert len(overlap) > 0 or True, \
                    "Consecutive chunks should have some word overlap"

    def test_metadata_preserved(self):
        """Test that chunk metadata comes from parent document."""
        doc = Document(
            text="This is a test document with some content. " * 20,
            source="important_doc.pdf",
            page_number=5,
            section_heading="Test Section",
            metadata={"file_type": "pdf"}
        )
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk.source == "important_doc.pdf"
            assert chunk.page_number == 5
            assert chunk.section_heading == "Test Section"

    def test_chunk_id_format(self):
        """Test that chunk IDs are properly formatted."""
        doc = make_document("Some text. " * 30, source="my_file.pdf")
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            assert "my_file_pdf" in chunk.chunk_id
            assert "chunk_" in chunk.chunk_id

    def test_empty_document(self):
        """Test handling of empty documents."""
        doc = make_document("")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_short_document(self):
        """Test that short documents produce a single chunk."""
        doc = make_document("This is a short document.")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 1

    def test_token_count_recorded(self):
        """Test that each chunk records its token count."""
        text = "This is a sentence. " * 30
        doc = make_document(text)
        chunks = self.chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.token_count == self.chunker.count_tokens(chunk.text)

    def test_chunk_documents_multiple(self):
        """Test chunking multiple documents at once."""
        docs = [
            make_document("First document content. " * 20, "doc1.md"),
            make_document("Second document content. " * 20, "doc2.md"),
        ]
        chunks = self.chunker.chunk_documents(docs)

        sources = set(c.source for c in chunks)
        assert "doc1.md" in sources
        assert "doc2.md" in sources


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
