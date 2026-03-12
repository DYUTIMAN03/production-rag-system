# Retrieval-Augmented Generation (RAG): A Technical Guide

## Introduction

Retrieval-Augmented Generation (RAG) is a technique that enhances large language model (LLM) outputs by grounding them in external knowledge retrieved from a document corpus. Instead of relying solely on the model's training data, RAG systems retrieve relevant passages from a knowledge base and use them as context for generating more accurate, factual, and up-to-date responses.

RAG was introduced by Lewis et al. in 2020 and has since become the dominant architecture for enterprise AI applications that need to answer questions about proprietary documents, internal knowledge bases, and domain-specific content.

## Document Chunking

Document chunking is the process of splitting large documents into smaller, semantically meaningful pieces called chunks. This is a critical step because embedding models have token limits, and retrieving entire documents would dilute the signal-to-noise ratio.

### Chunk Size and Overlap

The optimal chunk size is typically between 500 and 800 tokens. Chunks that are too small lose context, while chunks that are too large include irrelevant information that reduces retrieval precision.

Overlap between consecutive chunks is essential — typically 100 tokens. Without overlap, you risk slicing important sentences at boundaries, causing the retriever to miss critical context that spans two chunks. This overlap ensures that boundary information is represented in at least one complete chunk.

### Sentence-Aware Chunking

Naive chunking that splits purely by token count can break sentences in the middle, creating incoherent passages. Sentence-aware chunking respects sentence boundaries, ensuring each chunk starts and ends at natural sentence breaks while staying within the target token range.

## Embedding Models

Embedding models convert text into dense vector representations that capture semantic meaning. These vectors are stored in a vector database and used for similarity search during retrieval.

### Bi-Encoder vs Cross-Encoder

Bi-encoders (like sentence-transformers) encode the query and documents independently, producing fixed-size embeddings that can be compared via cosine similarity. They are fast but may miss nuanced query-document interactions.

Cross-encoders process the query and document together as a single input, allowing the model to attend across both texts simultaneously. This produces more accurate relevance scores but is computationally expensive and cannot be used for indexing — only for re-ranking a small set of candidates.

### Popular Models

Common embedding models include all-MiniLM-L6-v2 (384 dimensions, fast and lightweight), all-mpnet-base-v2 (768 dimensions, higher quality), and OpenAI's text-embedding models. For cross-encoder re-ranking, the ms-marco-MiniLM-L6-v2 model trained on the MS MARCO passage ranking dataset is widely used.

## Retrieval Strategies

### Vector Search

Vector search (also called semantic search or dense retrieval) uses cosine similarity between query and document embeddings to find the most relevant passages. It excels at understanding meaning, intent, and paraphrased queries.

However, vector search can struggle with exact keyword matches, rare technical terms, and specific identifiers that don't have strong semantic signal in the embedding space.

### BM25 Keyword Search

BM25 (Best Matching 25) is a classical information retrieval algorithm based on term frequency and inverse document frequency (TF-IDF). It excels at finding documents containing specific keywords and phrases.

BM25 handles exact term matching far better than vector search. When a user searches for a specific API name, error code, or technical identifier, BM25 will reliably surface documents containing those exact terms.

### Hybrid Search

Hybrid search combines vector search and BM25 to leverage the strengths of both approaches. The scores from each method are normalized to a common scale and combined with configurable weights.

A typical weighting is 60% vector search and 40% BM25, though the optimal ratio depends on the document corpus and query patterns. Score normalization (min-max scaling) is important to prevent one method from dominating the combined scores.

## Cross-Encoder Re-Ranking

After the initial retrieval step returns a broad set of candidate chunks (typically 20-50), a cross-encoder re-ranker evaluates each (query, chunk) pair jointly to produce more precise relevance scores.

The cross-encoder processes the query and chunk text together through a transformer model, allowing deep token-level attention between them. This consistently and dramatically improves precision compared to bi-encoder scores alone.

The re-ranker typically reduces the candidate set from 20 chunks to the top 5 most relevant, ensuring that only the highest-quality context is sent to the LLM for generation.

## Citation Enforcement

Citation enforcement is a critical quality control mechanism in production RAG systems. The system must explicitly decline to answer if the retrieved chunks don't actually support a response. No hallucinating plausible-sounding answers.

When the re-ranker scores for all retrieved chunks fall below a confidence threshold (typically 0.3), the system should return a refusal response stating that it doesn't have enough evidence to answer the question. This prevents the LLM from confabulating an answer that sounds plausible but isn't grounded in the source material.

Citation enforcement is what separates a demo from a production system. Anyone can build a RAG system that generates answers — the engineering challenge is ensuring those answers are trustworthy and traceable back to source evidence.

## Observability and Tracing

Production RAG systems require comprehensive observability to debug issues, track quality, and identify regressions.

### What to Trace

Every request should capture:
- The user's query
- Which chunks were retrieved and their scores
- What prompt was sent to the LLM
- The LLM's response
- Token consumption (input, output, total)
- Latency breakdown (retrieval, re-ranking, generation)

### Key Metrics

Important metrics to track include:
- Latency at P50 and P95 (not averages — averages hide worst-case performance)
- Cost per request in dollar terms
- Citation coverage: what percentage of answers are grounded in evidence
- Failure rate: how often the system errors or produces unsupported responses
- Re-ranker score distribution: are top chunks actually relevant?

### Tools

Langfuse is an open-source, self-hostable observability platform designed for LLM applications. It provides tracing, prompt management, and quality scoring out of the box.

## Evaluation with RAGAS

RAGAS (Retrieval-Augmented Generation Assessment Suite) is a framework specifically designed for evaluating RAG pipelines. It measures four key metrics:

### Faithfulness
Are the claims in the generated answer actually supported by the retrieved chunks? This is the most important metric — it directly measures whether the system is hallucinating or producing grounded responses.

### Answer Relevancy
Does the answer actually address the question asked? A perfectly faithful answer that doesn't address the user's question is still a bad answer.

### Context Precision
Are the retrieved chunks actually relevant to the question? High precision means less noise in the context, which leads to better LLM generation.

### Context Recall
Is the retriever finding all the relevant information from the corpus? Low recall means the system is missing important context that could improve the answer.

## CI-Gated Quality

In production AI teams, every change — code, prompts, or configuration — is automatically evaluated before being merged. A CI pipeline runs the evaluation script against a golden dataset of manually verified question-answer pairs.

If any quality metric drops below defined thresholds, the build fails and the change doesn't get merged. This discipline prevents quality regressions from shipping silently and ensures that the system maintains a consistent quality bar over time.

The golden dataset should contain 50-200 question-answer pairs that have been manually verified for correctness against the document corpus. Building this dataset takes time, but it becomes the foundation for all quality assurance.

## Prompt Versioning

Prompts are part of the system architecture — a prompt change can affect behavior just as dramatically as a code change. Storing prompts in versioned configuration files (YAML or JSON) with explicit version numbers, descriptions, and modification dates enables tracking which prompt version produced each response.

This versioning is critical for debugging quality regressions. When evaluation metrics drop, you can trace back to see if a prompt change caused the regression and compare outputs between prompt versions.
