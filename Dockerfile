# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for compiling some python packages like chroma/sqllite)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and models
ENV HF_HOME=/app/models
RUN mkdir -p /app/data/documents /app/models /app/chroma_db

# Copy project files
COPY . .

# Run ingestion script to pre-bake ChromaDB and BM25 index into the Docker image
# This ensures that ephemeral hosting (like free Render) starts with the data ready.
RUN python -c "from src.ingestion.loader import load_documents; \
from src.ingestion.chunker import TokenAwareChunker; \
from src.retrieval.vector_store import VectorStore; \
from src.retrieval.bm25_search import BM25Search; \
docs = load_documents('./data/documents'); \
chunker = TokenAwareChunker(); \
chunks = chunker.chunk_documents(docs); \
store = VectorStore(); \
store.add_chunks(chunks); \
bm25 = BM25Search(); \
bm25.build_index(store.get_all_chunks()); \
bm25.save_index(); \
print(f'Ingested {len(docs)} documents -> {len(chunks)} chunks')"

# Expose the API port
EXPOSE 8000

# Set environment variables for production
ENV HOST=0.0.0.0
ENV PORT=8000

# Run FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
