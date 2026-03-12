# Attention Mechanisms and Transformers Deep Dive

Attention mechanisms are the fundamental building block of modern deep learning, enabling models to dynamically focus on relevant parts of the input. The transformer architecture, built entirely on attention, has become the dominant architecture for NLP, vision, and multimodal AI.

## The Attention Mechanism

### Intuition

Attention answers the question: "Given a query, which parts of the input should I focus on?" It computes a weighted sum of values, where the weights are determined by the compatibility between a query and corresponding keys.

### Scaled Dot-Product Attention

The core attention operation:
1. Compute compatibility scores: score = Q * K^T / √d_k
2. Apply softmax to get attention weights: weights = softmax(score)
3. Compute weighted sum of values: output = weights * V

Where Q (queries), K (keys), V (values) are linear projections of the input, and d_k is the key dimension. The scaling factor √d_k prevents softmax saturation when dimensions are large.

### Multi-Head Attention (MHA)

Instead of a single attention function, multi-head attention runs h parallel attention heads with different learned projections:
1. Project Q, K, V into h different subspaces
2. Compute attention independently in each head
3. Concatenate and project the outputs

Multiple heads allow the model to attend to different types of relationships simultaneously — one head might capture syntactic relationships while another captures semantic ones. Typical configurations use 8-128 heads.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

MQA shares key and value projections across all heads, keeping only separate query projections. This dramatically reduces the KV-cache memory during inference (important for serving).

GQA is a middle ground: groups of query heads share the same KV projections. For example, 32 query heads might share 8 KV heads (groups of 4). GQA achieves near-MHA quality with near-MQA efficiency.

Llama 2 70B uses GQA; Llama 3 uses GQA across all sizes.

## The Transformer Architecture

### Original Architecture ("Attention Is All You Need", 2017)

The original transformer has an encoder-decoder structure:

**Encoder** (6 layers, each containing):
- Multi-head self-attention (each position attends to all positions)
- Position-wise feed-forward network (two linear layers with ReLU)
- Residual connections and layer normalization around each sub-layer

**Decoder** (6 layers, each containing):
- Masked multi-head self-attention (each position can only attend to earlier positions)
- Multi-head cross-attention (attends to encoder output)
- Position-wise feed-forward network
- Residual connections and layer normalization

### Pre-Norm vs Post-Norm

The original transformer uses Post-Norm (apply layer norm after the residual connection). Modern transformers use Pre-Norm (apply layer norm before the attention/FFN), which provides more stable training gradients and enables training deeper models without warmup.

### RMSNorm

Root Mean Square LayerNorm, used by Llama and other modern LLMs. Removes the mean-centering step of standard LayerNorm:
RMSNorm(x) = x / √(mean(x²) + ε) * γ

RMSNorm is faster than standard LayerNorm and produces equivalent results.

## Positional Encoding

Since transformers process all tokens simultaneously (no inherent position information), position must be explicitly encoded.

### Sinusoidal Positional Encoding

The original transformer approach uses fixed sinusoidal functions of different frequencies:
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

This allows the model to generalize to sequence lengths not seen during training.

### Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating the query and key vectors in 2D subspaces. The rotation angle depends on the token position. Key advantage: the attention score between two tokens depends only on their relative position, enabling better length generalization.

Used by Llama 1/2/3, Mistral, Qwen, and most modern LLMs.

### ALiBi (Attention with Linear Biases)

ALiBi adds a position-dependent linear bias to attention scores instead of modifying embeddings. The bias penalizes attention to distant tokens. Enables training on short sequences and extrapolating to longer ones.

## Architecture Variants

### Encoder-Only (e.g., BERT)

Uses only the encoder with bidirectional self-attention (each token attends to all tokens). Best for understanding tasks: classification, named entity recognition, extractive QA, sentence embeddings.

Pretraining objective: Masked Language Modeling (MLM) — randomly mask 15% of tokens and predict them.

### Decoder-Only (e.g., GPT)

Uses only the decoder with causal (left-to-right) self-attention. Each token can only attend to preceding tokens. Best for generation tasks: text generation, code completion, instruction following.

Pretraining objective: Next token prediction — predict each token given all previous tokens.

This is the dominant architecture for modern LLMs (GPT-4, Claude, Llama, Mistral, Gemini).

### Encoder-Decoder (e.g., T5, BART)

Uses both encoder and decoder with cross-attention. Best for sequence-to-sequence tasks: translation, summarization, generative QA.

T5 frames all NLP tasks as text-to-text and was trained with span corruption.

## Efficient Attention Variants

Standard attention has O(n²) complexity in sequence length, making long sequences expensive.

### Flash Attention

Flash Attention is an IO-aware attention algorithm that significantly speeds up computation by:
1. Tiling the attention computation to exploit GPU SRAM (fast memory)
2. Avoiding materializing the full n×n attention matrix in GPU HBM (slow memory)
3. Recomputing attention during backward pass instead of storing it

Flash Attention 2 achieves 2-4x speedup over standard attention and is now the default in most frameworks. Flash Attention 3 adds further optimizations for Hopper GPUs.

### Sparse Attention

Only compute attention for a subset of token pairs:
- **Local attention**: Each token attends only to nearby tokens (window attention)
- **Strided attention**: Attend to every k-th token
- **BigBird**: Combines random, window, and global attention patterns
- **Longformer**: Window attention + global attention for special tokens

### Linear Attention

Approximate softmax attention with kernel functions, reducing complexity from O(n²) to O(n):
- **Performer**: Uses FAVOR+ (Fast Attention Via Orthogonal Random Features)
- **Linear Transformers**: Replace softmax with other kernel functions

### Sliding Window Attention

Used by Mistral and related models. Each layer uses a fixed window size W, and information propagates across layers. With L layers and window W, the effective receptive field is L × W. This achieves linear complexity while maintaining good performance.

## KV-Cache and Inference Optimization

During autoregressive generation, the KV-cache stores previously computed key and value vectors to avoid recomputation:
- Without KV-cache: O(n²) per token, where n is sequence length
- With KV-cache: O(n) per token (only compute attention for the new token)

KV-cache memory grows linearly with sequence length and batch size, often becoming the bottleneck for LLM serving. Management strategies:
- **PagedAttention (vLLM)**: Manages KV-cache like virtual memory with pages, eliminating fragmentation
- **GQA/MQA**: Reduces KV-cache size by sharing key-value heads
- **Quantized KV-cache**: Store cached values in INT8 or INT4
- **Sliding window**: Only cache recent tokens

## Mixture of Experts (MoE)

MoE replaces the dense FFN layer with multiple "expert" FFN networks and a gating network that routes each token to the top-k experts (typically k=2). This allows scaling model parameters dramatically while keeping computation constant per token.

Examples: Mixtral 8x7B (8 experts, 2 active, 47B total params, 13B active per token), GPT-4 (rumored MoE), Grok.

Benefits: More parameters → more knowledge, without proportionally more compute.
Challenges: Load balancing across experts, all-to-all communication in distributed training, memory for all expert weights.
