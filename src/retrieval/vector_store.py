"""
ChromaDB Vector Store — stores and retrieves document chunk embeddings.
Uses sentence-transformers all-MiniLM-L6-v2 for embedding.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

from src.ingestion.chunker import Chunk


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk_id: str
    text: str
    score: float  # Lower distance = more similar
    source: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    metadata: dict = None


class VectorStore:
    """ChromaDB-backed vector store with sentence-transformer embeddings."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add document chunks to the vector store. Returns count added."""
        if not chunks:
            return 0

        # Deduplicate chunks by ID (keep last occurrence)
        seen = {}
        for chunk in chunks:
            seen[chunk.chunk_id] = chunk
        unique_chunks = list(seen.values())

        ids = [chunk.chunk_id for chunk in unique_chunks]
        documents = [chunk.text for chunk in unique_chunks]
        metadatas = [
            {
                "source": chunk.source,
                "page_number": chunk.page_number or -1,
                "section_heading": chunk.section_heading or "",
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
            }
            for chunk in unique_chunks
        ]

        # Upsert in batches of 100 (ChromaDB limit)
        batch_size = 100
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            total_added += len(batch_ids)

        return total_added

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Search for the top-k most similar chunks to the query."""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return []

        search_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Convert cosine distance to similarity score (0-1, higher=better)
            similarity = 1.0 - distance

            meta = results["metadatas"][0][i]
            search_results.append(SearchResult(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                score=similarity,
                source=meta.get("source", ""),
                page_number=meta.get("page_number"),
                section_heading=meta.get("section_heading"),
                metadata=meta,
            ))

        return search_results

    def get_all_chunks(self) -> List[dict]:
        """Retrieve all chunks from the collection (for BM25 index building)."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            include=["documents", "metadatas"],
        )

        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "chunk_id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        return chunks

    def count(self) -> int:
        """Return the total number of chunks stored."""
        return self.collection.count()

    def clear(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
