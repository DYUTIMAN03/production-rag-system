# Embeddings and Representation Learning

Embeddings are dense vector representations that capture semantic meaning of data (text, images, code, etc.) in a continuous vector space. They are fundamental to modern AI — powering search, recommendation, clustering, and retrieval-augmented generation.

## Text Embeddings

### Word2Vec

Word2Vec (Mikolov et al., 2013) learns word embeddings by predicting context from words or words from context:
- **CBOW (Continuous Bag of Words)**: Predicts target word from surrounding context words
- **Skip-gram**: Predicts surrounding context words from target word

Key properties: captures analogies (king - man + woman ≈ queen), words in similar contexts have similar embeddings. Typical dimensions: 100-300.

Limitations: each word has exactly one embedding regardless of context ("bank" has the same vector in "river bank" and "bank account").

### GloVe (Global Vectors)

GloVe learns embeddings by factorizing the word co-occurrence matrix. It captures both local context (like Word2Vec) and global corpus statistics. Often produces slightly better embeddings than Word2Vec for downstream tasks.

### FastText

Facebook's extension of Word2Vec that represents words as bags of character n-grams. This enables:
- Embeddings for out-of-vocabulary words (by composing subword embeddings)
- Better representations for morphologically rich languages
- Robust handling of typos and rare words

### Contextual Embeddings (BERT, GPT)

Modern language models produce context-dependent embeddings — the same word gets different vectors based on surrounding text. BERT uses bidirectional attention, producing rich representations that capture nuanced meaning.

To get a sentence embedding from BERT:
- **[CLS] token**: Use the embedding of the special classification token
- **Mean pooling**: Average all token embeddings (usually better)
- **Max pooling**: Take the element-wise maximum across token embeddings

## Sentence and Document Embeddings

### Sentence-Transformers

The sentence-transformers library fine-tunes transformer models specifically for producing high-quality sentence embeddings. It uses a siamese/triplet network architecture trained on semantic similarity datasets.

Popular models:
- **all-MiniLM-L6-v2**: Good balance of quality and speed (384 dimensions, 80MB)
- **all-mpnet-base-v2**: Higher quality, slower (768 dimensions, 420MB)
- **bge-large-en-v1.5**: BAAI's high-quality model (1024 dimensions)
- **gte-large**: Alibaba's general text embedding model
- **e5-large-v2**: Microsoft's general-purpose embedding model

### OpenAI Embeddings

OpenAI's text-embedding-3-small and text-embedding-3-large models via API. Support Matryoshka representation learning — embeddings can be truncated to smaller dimensions with graceful quality degradation.

### Cohere Embed

Cohere's embedding models support multiple input types (search_document, search_query, classification, clustering) and produce embeddings optimized for each use case.

## Similarity Metrics

### Cosine Similarity

Measures the cosine of the angle between two vectors: cos(θ) = (A·B) / (||A|| × ||B||). Range: [-1, 1]. Most common metric for text embeddings because it's invariant to vector magnitude.

### Euclidean Distance (L2)

Straight-line distance between two points in vector space: d = √(Σ(a_i - b_i)²). Sensitive to vector magnitude. Used when absolute differences matter.

### Dot Product (Inner Product)

Simple dot product: A·B = Σ(a_i × b_i). Equivalent to cosine similarity when vectors are normalized. Faster to compute than cosine similarity.

### Manhattan Distance (L1)

Sum of absolute differences: d = Σ|a_i - b_i|. More robust to outliers than Euclidean distance.

## Approximate Nearest Neighbor (ANN) Search

Exact nearest neighbor search is O(n) per query — too slow for millions of embeddings. ANN methods trade small accuracy losses for dramatic speed improvements.

### HNSW (Hierarchical Navigable Small World)

Builds a multi-layer graph structure for fast navigation. The top layers contain fewer nodes with long-range connections for quick coarse search; lower layers have more nodes with short-range connections for fine-grained search.

Performance: ~95-99% recall with 10-100x speedup over brute force. Used by ChromaDB, Qdrant, Weaviate, and pgvector.

### IVF (Inverted File Index)

Clusters vectors into Voronoi cells using k-means. At search time, only searches the nprobe closest cells. Combined with product quantization (IVF-PQ) for memory efficiency.

Used by FAISS (Facebook AI Similarity Search).

### ScaNN (Scalable Nearest Neighbors)

Google's ANN library using anisotropic vector quantization. Optimized for maximum inner product search, which is the natural metric for retrieval tasks.

## Multimodal Embeddings

### CLIP (Contrastive Language-Image Pretraining)

CLIP jointly trains an image encoder and text encoder to map images and text into a shared embedding space. Trained on 400M image-text pairs from the internet.

Applications: zero-shot image classification, image-text search, image generation guidance (used by DALL-E and Stable Diffusion).

### ImageBind

Meta's model that extends CLIP-style training to 6 modalities: images, text, audio, depth, thermal, and IMU data. All modalities are mapped to a shared embedding space.

## Embedding Techniques

### Matryoshka Representation Learning (MRL)

Trains embeddings such that any prefix of the embedding vector is also a useful representation. This allows flexible dimensionality — use 256 dims for fast approximate search and full 1024 dims for precise re-ranking.

### Instruction-Following Embeddings

Recent models like Instructor and e5-mistral accept natural language instructions that describe how to embed the input. For example: "Represent this query for retrieving relevant scientific papers:" produces different embeddings than "Represent this sentence for clustering by topic:".

### Late Interaction (ColBERT)

Instead of compressing each document to a single vector, ColBERT stores per-token embeddings and computes relevance using MaxSim (maximum similarity between query and document tokens). This captures fine-grained interactions while keeping retrieval efficient through pre-computed document embeddings.

## Applications

- **Semantic search**: Find documents by meaning rather than keyword matching
- **RAG retrieval**: Retrieve relevant context for LLM generation
- **Recommendation**: Find similar items (products, movies, music)
- **Clustering**: Group similar documents, images, or users
- **Anomaly detection**: Identify outliers in embedding space
- **Classification**: Use embeddings as features for downstream classifiers
- **Deduplication**: Find near-duplicate content using embedding similarity
