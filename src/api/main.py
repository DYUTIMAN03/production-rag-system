"""
FastAPI Application — Production RAG System with Full Observability.
"""

import os
import sys
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from src.api.feedback import FeedbackStore
from src.api.routes import router, set_dependencies
from src.generation.llm import GroqLLM
from src.generation.prompt_manager import PromptManager
from src.generation.query_rewriter import QueryRewriter
from src.generation.rag_pipeline import RAGPipeline
from src.ingestion.chunker import TokenAwareChunker
from src.observability.metrics import MetricsCollector
from src.observability.tracer import LangfuseTracer
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


def load_config():
    """Load application configuration from YAML."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "settings.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown application resources."""
    config = load_config()
    print("\n[*] Initializing Production RAG System...")

    # Vector Store
    vs_config = config.get("vector_store", {})
    vector_store = VectorStore(
        persist_directory=vs_config.get("persist_directory", "./chroma_db"),
        collection_name=vs_config.get("collection_name", "rag_documents"),
        embedding_model=config.get("embedding", {}).get("model", "all-MiniLM-L6-v2"),
    )
    print(f"  [+] Vector store initialized ({vector_store.count()} chunks)")

    # BM25 Search
    bm25 = BM25Search()
    if bm25.load_index():
        print(f"  [+] BM25 index loaded ({len(bm25.chunk_data)} chunks)")
    else:
        chunks = vector_store.get_all_chunks()
        if chunks:
            bm25.build_index(chunks)
            print(f"  [+] BM25 index built ({len(chunks)} chunks)")
        else:
            print("  [!] BM25 index empty -- ingest documents first")

    # Hybrid Retriever
    ret_config = config.get("retrieval", {})
    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_search=bm25,
        alpha=ret_config.get("hybrid_alpha", 0.6),
    )

    # Reranker
    reranker = Reranker(
        model_name=config.get("reranker", {}).get("model", "cross-encoder/ms-marco-MiniLM-L6-v2"),
    )
    print("  [+] Cross-encoder reranker loaded")

    # LLM
    llm_config = config.get("llm", {})
    llm = GroqLLM(
        model=llm_config.get("model", "llama-3.3-70b-versatile"),
        temperature=llm_config.get("temperature", 0.1),
        max_output_tokens=llm_config.get("max_output_tokens", 1024),
    )
    print(f"  [+] LLM client initialized ({llm_config.get('model', 'llama-3.3-70b-versatile')})")

    # Prompt Manager
    prompt_manager = PromptManager()
    versions = prompt_manager.get_all_versions()
    print(f"  [+] Prompts loaded (versions: {versions})")

    # Query Rewriter
    query_rewriter = QueryRewriter(llm=llm, prompt_manager=prompt_manager)
    print("  [+] Query rewriter initialized")

    # RAG Pipeline
    pipeline = RAGPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        llm=llm,
        prompt_manager=prompt_manager,
        query_rewriter=query_rewriter,
        initial_top_k=ret_config.get("initial_top_k", 20),
        final_top_k=ret_config.get("final_top_k", 5),
        reranker_threshold=ret_config.get("reranker_threshold", 0.3),
    )

    # Observability
    tracer = LangfuseTracer()
    metrics_collector = MetricsCollector()

    # Chunker
    chunk_config = config.get("chunking", {})
    chunker = TokenAwareChunker(
        target_tokens=chunk_config.get("target_tokens", 600),
        max_tokens=chunk_config.get("max_tokens", 800),
        min_tokens=chunk_config.get("min_tokens", 100),
        overlap_tokens=chunk_config.get("overlap_tokens", 100),
    )

    # Feedback Store
    feedback_store = FeedbackStore()
    print("  [+] Feedback store initialized")

    # Set dependencies for routes
    set_dependencies(pipeline, vector_store, bm25, metrics_collector, tracer, chunker, feedback_store)

    print("\n[OK] RAG System ready!\n")

    yield  # Application runs here

    # Cleanup
    if tracer.enabled:
        tracer.flush()
    print("\n[x] RAG System shutting down.")


# Create FastAPI app
app = FastAPI(
    title="Production RAG System",
    description="Domain-specific document Q&A with hybrid retrieval, "
                "cross-encoder reranking, citation enforcement, and full observability.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve static files (web frontend)
web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
async def serve_landing():
    """Serve the cinematic landing page."""
    landing_path = os.path.join(web_dir, "landing.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path)
    return {"message": "Production RAG System API. Visit /docs for API documentation."}


@app.get("/app")
async def serve_frontend():
    """Serve the RAG system web frontend."""
    index_path = os.path.join(web_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Production RAG System API. Visit /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
