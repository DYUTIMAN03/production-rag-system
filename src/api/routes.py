"""
API Routes — FastAPI endpoints for the RAG system.
"""

import os
import sys

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import List

from src.api.models import (
    CitationResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummaryResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()

# These will be set by main.py during app initialization
_pipeline = None
_vector_store = None
_bm25_search = None
_metrics_collector = None
_tracer = None
_chunker = None
_feedback_store = None


def set_dependencies(pipeline, vector_store, bm25_search, metrics_collector, tracer, chunker, feedback_store=None):
    """Set shared dependencies from main.py."""
    global _pipeline, _vector_store, _bm25_search, _metrics_collector, _tracer, _chunker, _feedback_store
    _pipeline = pipeline
    _vector_store = vector_store
    _bm25_search = bm25_search
    _metrics_collector = metrics_collector
    _tracer = tracer
    _chunker = chunker
    _feedback_store = feedback_store


@router.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main RAG query endpoint.
    Performs query rewriting → hybrid retrieval → re-ranking → citation enforcement → generation.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Create Langfuse trace for this request
        trace = _tracer.create_trace(
            name="rag_query",
            metadata={"question": request.question, "top_k": request.top_k},
        )

        # Run the pipeline
        response = _pipeline.query(request.question)

        # Record Langfuse traces
        _tracer.trace_pipeline_result(trace, response)

        # Record metrics
        _metrics_collector.record(response)

        # Flush traces
        _tracer.flush()

        return QueryResponse(
            answer=response.answer,
            citations=[
                CitationResponse(
                    source=c.source,
                    page_number=c.page_number,
                    section_heading=c.section_heading,
                    chunk_text=c.chunk_text,
                    relevance_score=c.relevance_score,
                )
                for c in response.citations
            ],
            chunks_used=response.chunks_used,
            confidence_score=response.confidence_score,
            is_grounded=response.is_grounded,
            metrics=response.metrics,
        )

    except Exception as e:
        if _metrics_collector:
            _metrics_collector.record_error(str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.post("/api/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streaming RAG query endpoint via Server-Sent Events.
    Returns metadata first, then streams tokens, then sends done signal.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    def event_stream():
        try:
            # Create Langfuse trace for this streaming request
            trace = _tracer.create_trace(
                name="rag_query_stream",
                metadata={"question": request.question, "top_k": request.top_k},
            ) if _tracer else None

            collected_answer = []
            stream_metadata = {}
            stream_done = {}

            for event in _pipeline.query_stream(request.question):
                # Parse SSE events for tracing and metrics
                if event.startswith("data: "):
                    import json as _json
                    try:
                        data = _json.loads(event[6:].strip())
                        if data.get("type") == "metadata":
                            stream_metadata = data
                        elif data.get("type") == "token":
                            collected_answer.append(data.get("content", ""))
                        elif data.get("type") == "done":
                            stream_done = data
                    except Exception:
                        pass
                yield event

            # Record metrics after streaming completes
            if _metrics_collector and stream_metadata:
                _metrics_collector.record_from_stream(
                    confidence_score=stream_metadata.get("confidence_score", 0),
                    is_grounded=stream_metadata.get("is_grounded", False),
                    chunks_used=stream_metadata.get("chunks_used", 0),
                    citation_count=len(stream_metadata.get("citations", [])),
                    total_latency_ms=stream_done.get("total_latency_ms", 0),
                    reranker_max_score=stream_metadata.get("confidence_score", 0),
                )

            # Record Langfuse trace
            if trace and _tracer and _tracer.enabled:
                trace.update(
                    output={"answer": "".join(collected_answer)},
                )
                _tracer.flush()

        except Exception as e:
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    """
    Ingest documents from a given file or directory path.
    """
    if not _vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        from src.ingestion.loader import load_documents

        # Load documents
        documents = load_documents(request.path)

        if not documents:
            return IngestResponse(
                documents_loaded=0,
                chunks_created=0,
                chunks_stored=0,
                message=f"No documents found at: {request.path}",
            )

        # Chunk documents
        chunks = _chunker.chunk_documents(documents)

        # Store in vector DB
        stored = _vector_store.add_chunks(chunks)

        # Rebuild BM25 index
        all_chunks = _vector_store.get_all_chunks()
        _bm25_search.build_index(all_chunks)
        _bm25_search.save_index()

        return IngestResponse(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            chunks_stored=stored,
            message=f"Successfully ingested {len(documents)} documents into {stored} chunks",
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "documents")


@router.post("/api/upload")
async def upload_endpoint(files: List[UploadFile] = File(...)):
    """
    Upload documents from the user's PC and ingest them into the RAG system.
    Accepts PDF, Markdown, and text files.
    """
    if not _vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_files = []
    rejected_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            rejected_files.append(f"{file.filename} (unsupported type: {ext})")
            continue

        # Save the file
        safe_name = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_DIR, safe_name)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved_files.append(file_path)

    if not saved_files:
        detail = "No supported files uploaded."
        if rejected_files:
            detail += f" Rejected: {', '.join(rejected_files)}"
        raise HTTPException(status_code=400, detail=detail)

    # Ingest uploaded files
    try:
        from src.ingestion.loader import load_documents

        all_docs = []
        for fp in saved_files:
            docs = load_documents(fp)
            all_docs.extend(docs)

        if not all_docs:
            return {
                "files_saved": len(saved_files),
                "documents_loaded": 0,
                "chunks_created": 0,
                "chunks_stored": 0,
                "message": "Files saved but no text could be extracted.",
            }

        chunks = _chunker.chunk_documents(all_docs)
        stored = _vector_store.add_chunks(chunks)

        # Rebuild BM25 index
        all_chunks = _vector_store.get_all_chunks()
        _bm25_search.build_index(all_chunks)
        _bm25_search.save_index()

        msg_parts = [f"Uploaded {len(saved_files)} file(s), created {stored} chunks."]
        if rejected_files:
            msg_parts.append(f"Rejected: {', '.join(rejected_files)}")

        return {
            "files_saved": len(saved_files),
            "documents_loaded": len(all_docs),
            "chunks_created": len(chunks),
            "chunks_stored": stored,
            "rejected": rejected_files,
            "message": " ".join(msg_parts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload ingestion error: {str(e)}")



@router.get("/api/metrics", response_model=MetricsResponse)
async def metrics_endpoint():
    """Return aggregated pipeline metrics."""
    if not _metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not initialized")

    summary = _metrics_collector.get_summary()
    return MetricsResponse(**summary)


@router.get("/api/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        vector_store_count=_vector_store.count() if _vector_store else 0,
        langfuse_enabled=_tracer.enabled if _tracer else False,
    )


@router.get("/api/documents")
async def documents_endpoint():
    """Return unique document sources with chunk counts for the Corpus Panel."""
    if not _vector_store:
        return {"documents": []}

    try:
        chunks = _vector_store.get_all_chunks()
        source_counts = {}
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            # Clean up source path to just filename
            source_name = source.replace("\\", "/").split("/")[-1] if source else "unknown"
            source_counts[source_name] = source_counts.get(source_name, 0) + 1

        documents = [
            {"name": name, "chunks": count}
            for name, count in sorted(source_counts.items())
        ]
        return {"documents": documents, "total_chunks": len(chunks)}
    except Exception:
        return {"documents": [], "total_chunks": 0}


@router.post("/api/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(request: FeedbackRequest):
    """Save user feedback (thumbs up/down) for a query-response pair."""
    if not _feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")

    try:
        feedback_id = _feedback_store.save_feedback(
            question=request.question,
            answer=request.answer,
            is_positive=request.is_positive,
            citations=request.citations,
            comment=request.comment,
            confidence_score=request.confidence_score,
            chunks_used=request.chunks_used,
            is_grounded=request.is_grounded,
        )

        emoji = "[+]" if request.is_positive else "[-]"
        return FeedbackResponse(
            feedback_id=feedback_id,
            message=f"{emoji} Feedback recorded. Thank you!",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


@router.get("/api/feedback/summary", response_model=FeedbackSummaryResponse)
async def feedback_summary_endpoint():
    """Get aggregated feedback statistics."""
    if not _feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")

    summary = _feedback_store.get_summary()
    return FeedbackSummaryResponse(**summary)
