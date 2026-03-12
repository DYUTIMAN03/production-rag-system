# 🔍 Production RAG System with Full Observability

A domain-specific **"Ask My Docs"** system with hybrid retrieval (BM25 + vector search), cross-encoder reranking, citation enforcement, Langfuse observability, RAGAS evaluation, and CI-gated regression testing.

**100% free and open-source — no payment required.**

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

---

## Architecture

```
Query → Hybrid Retrieval (Vector + BM25) → Cross-Encoder Re-Ranking → Citation Enforcement → LLM Generation
           ↓                                      ↓                         ↓                      ↓
      Langfuse Trace                         Score Stats              Grounded?              Token Usage
                                                                    ↓ No → Refuse
                                                                    ↓ Yes → Cited Answer
```

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| LLM | Google Gemini `gemini-2.0-flash` | Free |
| Embeddings | `all-MiniLM-L6-v2` (local) | Free |
| Vector Store | ChromaDB (persistent) | Free |
| Keyword Search | `rank-bm25` | Free |
| Re-Ranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | Free |
| Tracing | Langfuse (self-hosted Docker) | Free |
| Evaluation | RAGAS-style metrics | Free |
| API | FastAPI | Free |
| CI | GitHub Actions | Free |

## Quick Start

### 1. Clone & Setup
```bash
cd RAG
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env with your GOOGLE_API_KEY
# Get a free key at: https://aistudio.google.com/apikey
```

### 3. Start Langfuse (Optional)
```bash
docker compose -f docker-compose.langfuse.yml up -d
# Visit http://localhost:3000 to create an account
# Generate API keys and add to .env
```

### 4. Ingest Documents
```bash
# Via API after starting the server, or manually:
python -c "
from src.ingestion.loader import load_documents
from src.ingestion.chunker import TokenAwareChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search

docs = load_documents('./data/documents')
chunker = TokenAwareChunker()
chunks = chunker.chunk_documents(docs)

store = VectorStore()
store.add_chunks(chunks)

bm25 = BM25Search()
bm25.build_index(store.get_all_chunks())
bm25.save_index()

print(f'Ingested {len(docs)} documents → {len(chunks)} chunks')
"
```

### 5. Run the Server
```bash
python -m src.api.main
# Visit http://localhost:8000
```

## Features

### Hybrid Retrieval
Combines BM25 keyword search with vector semantic search. Vector search captures meaning; BM25 captures exact terms. Configurable weight blending (default: 60/40).

### Cross-Encoder Re-Ranking
After initial retrieval, a cross-encoder evaluates (query, chunk) pairs jointly for dramatically improved precision. Reduces 20 candidates to top 5.

### Citation Enforcement
**Hard rule, not a soft guideline.** If the re-ranker scores fall below the confidence threshold, the system explicitly declines to answer. No hallucination.

### Prompt Versioning
All prompts stored in `config/prompts.yaml` with version numbers. Every response is traceable to the exact prompt version that generated it.

### Observability (Langfuse)
Every request traces: chunks retrieved, prompt sent, response generated, tokens consumed. P50/P95 latency, cost tracking, citation coverage, re-ranker score distribution.

### CI-Gated Evaluation
GitHub Actions runs RAGAS evaluation on every PR. If faithfulness or other quality metrics drop below thresholds, the build fails.

## Project Structure

```
RAG/
├── config/
│   ├── settings.yaml        # All tunable parameters
│   └── prompts.yaml         # Versioned prompt templates
├── src/
│   ├── ingestion/           # Document loading & chunking
│   ├── retrieval/           # Vector, BM25, hybrid, reranker
│   ├── generation/          # LLM, prompts, RAG pipeline
│   ├── observability/       # Langfuse tracing, metrics
│   ├── evaluation/          # RAGAS evaluation & golden dataset
│   ├── api/                 # FastAPI endpoints
│   └── web/                 # Frontend UI
├── tests/                   # Unit & integration tests
├── data/documents/          # Source document corpus
├── docker-compose.langfuse.yml
└── .github/workflows/       # CI evaluation pipeline
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/query` | Ask a question (returns cited answer) |
| `POST` | `/api/ingest` | Ingest documents from a path |
| `GET` | `/api/metrics` | Pipeline metrics (P50/P95, cost, quality) |
| `GET` | `/api/health` | Health check with chunk count |

## Testing
```bash
python -m pytest tests/ -v
```

## Evaluation
```bash
python -m src.evaluation.evaluate --output evaluation_report.json
```

## License

MIT
