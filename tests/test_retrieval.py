"""Tests for retrieval components — vector store, BM25, hybrid, and reranker."""

import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search


def make_test_chunks():
    """Create test chunks for retrieval testing."""
    return [
        Chunk(
            chunk_id="chunk_0",
            text="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            source="ml_guide.md",
            chunk_index=0,
            token_count=20,
        ),
        Chunk(
            chunk_id="chunk_1",
            text="Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            source="dl_guide.md",
            chunk_index=0,
            token_count=18,
        ),
        Chunk(
            chunk_id="chunk_2",
            text="Natural language processing enables computers to understand, interpret, and generate human language.",
            source="nlp_guide.md",
            chunk_index=0,
            token_count=16,
        ),
        Chunk(
            chunk_id="chunk_3",
            text="Vector databases store high-dimensional embeddings for efficient similarity search operations.",
            source="vector_db.md",
            chunk_index=0,
            token_count=14,
        ),
        Chunk(
            chunk_id="chunk_4",
            text="RAG combines retrieval with generation to produce grounded answers from document collections.",
            source="rag_guide.md",
            chunk_index=0,
            token_count=16,
        ),
    ]


class TestVectorStore:
    """Tests for ChromaDB vector store."""

    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.store = VectorStore(
            persist_directory=self.test_dir,
            collection_name="test_collection",
        )

    def teardown_method(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_and_count(self):
        """Test adding chunks and counting."""
        chunks = make_test_chunks()
        added = self.store.add_chunks(chunks)
        assert added == 5
        assert self.store.count() == 5

    def test_search_returns_results(self):
        """Test that search returns relevant results."""
        self.store.add_chunks(make_test_chunks())
        results = self.store.search("What is machine learning?", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert results[0].text  # Should have text content
        assert results[0].source  # Should have source metadata

    def test_search_scores(self):
        """Test that search results have scores."""
        self.store.add_chunks(make_test_chunks())
        results = self.store.search("neural networks deep learning", top_k=3)

        for r in results:
            assert isinstance(r.score, float)

    def test_empty_search(self):
        """Test searching an empty store."""
        results = self.store.search("anything")
        assert results == []

    def test_get_all_chunks(self):
        """Test retrieving all stored chunks."""
        self.store.add_chunks(make_test_chunks())
        all_chunks = self.store.get_all_chunks()
        assert len(all_chunks) == 5

    def test_clear(self):
        """Test clearing the store."""
        self.store.add_chunks(make_test_chunks())
        assert self.store.count() == 5
        self.store.clear()
        assert self.store.count() == 0


class TestBM25Search:
    """Tests for BM25 keyword search."""

    def setup_method(self):
        self.bm25 = BM25Search()
        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": {"source": c.source},
            }
            for c in make_test_chunks()
        ]
        self.bm25.build_index(chunks_data)

    def test_search_returns_results(self):
        """Test that BM25 search returns results."""
        results = self.bm25.search("machine learning artificial intelligence")
        assert len(results) > 0

    def test_keyword_matching(self):
        """Test that BM25 matches specific keywords."""
        results = self.bm25.search("vector databases embeddings")
        assert len(results) > 0
        # The vector database chunk should be ranked highly
        sources = [r.source for r in results[:3]]
        assert "vector_db.md" in sources

    def test_score_ordering(self):
        """Test that results are ordered by score descending."""
        results = self.bm25.search("deep learning neural networks")
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_empty_query(self):
        """Test with empty/stop-word-only query."""
        results = self.bm25.search("")
        assert len(results) == 0

    def test_index_persistence(self):
        """Test saving and loading BM25 index."""
        import tempfile
        idx_path = os.path.join(tempfile.gettempdir(), "test_bm25.pkl")

        self.bm25.index_path = idx_path
        self.bm25.save_index()

        new_bm25 = BM25Search(index_path=idx_path)
        assert new_bm25.load_index() is True

        results = new_bm25.search("machine learning")
        assert len(results) > 0

        os.remove(idx_path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
