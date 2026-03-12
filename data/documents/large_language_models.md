# Large Language Models (LLMs)

## What are Large Language Models?

Large Language Models (LLMs) are deep learning models trained on massive text corpora to understand and generate human language. They are based on the transformer architecture and contain billions (or even trillions) of parameters. LLMs have demonstrated emergent abilities — capabilities that appear only at sufficient scale, such as few-shot learning, chain-of-thought reasoning, and instruction following.

The term "large" refers both to the model size (parameter count) and the training data volume. GPT-3 has 175 billion parameters and was trained on 300 billion tokens. LLaMA 2 has up to 70 billion parameters trained on 2 trillion tokens. GPT-4 is estimated to be significantly larger, possibly using a mixture-of-experts architecture.

## The Transformer Architecture

The transformer, introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al., replaced recurrent architectures with a self-attention mechanism that processes all tokens in parallel. This parallelization enabled training on much larger datasets and model sizes.

### Self-Attention Mechanism

Self-attention allows each token in a sequence to attend to every other token, computing a weighted sum based on relevance. The mechanism works through three learned projections:

1. **Query (Q)**: What the current token is looking for
2. **Key (K)**: What each token offers to attend to
3. **Value (V)**: The actual information from each token

The attention score is computed as: Attention(Q, K, V) = softmax(QK^T / √d_k) × V

The division by √d_k (the dimension of the key vectors) prevents the dot products from becoming too large, which would push the softmax into regions with extremely small gradients.

### Multi-Head Attention

Instead of computing a single attention function, multi-head attention runs multiple attention operations in parallel (each with its own learned Q, K, V projections), allowing the model to attend to different types of relationships simultaneously. Different heads can capture syntax, semantics, coreference, positional patterns, and other linguistic phenomena.

### Positional Encoding

Since transformers process all tokens simultaneously (unlike RNNs which inherently capture order), positional information must be explicitly injected. The original transformer used sinusoidal positional encodings. Modern LLMs use:

**Rotary Position Embeddings (RoPE)**: Used in LLaMA, Mistral, and many modern models. Encodes position by rotating the query and key vectors in 2D subspaces. Supports extrapolation to longer sequences than seen during training.

**ALiBi (Attention with Linear Biases)**: Adds a linear bias to attention scores based on distance between tokens. Provides better length extrapolation without learnable parameters.

### Feed-Forward Network

Each transformer layer contains a feed-forward network (FFN) applied independently to each position. Modern LLMs typically use a gated FFN variant (SwiGLU activations), which has been shown to improve performance.

### Layer Normalization

Layer normalization stabilizes training by normalizing the activations across the feature dimension. Modern LLMs use:
- **Pre-LayerNorm**: Applying normalization before (not after) each sublayer. This improves training stability, especially for large models.
- **RMSNorm**: A simplified version of layer normalization that only normalizes by the root mean square, omitting the mean centering. Used in LLaMA and many modern models.

## Pretraining Objectives

### Causal Language Modeling (CLM)

The model predicts the next token given all previous tokens. This is the pretraining objective for GPT-series models, LLaMA, Mistral, and most modern LLMs. The model processes text left-to-right, masking future tokens with a causal attention mask.

### Masked Language Modeling (MLM)

The model predicts randomly masked tokens given the surrounding context (bidirectional). This is BERT's pretraining objective. Typically, 15% of tokens are selected for prediction: 80% are replaced with [MASK], 10% with a random token, and 10% kept unchanged.

### Prefix Language Modeling

A hybrid where a prefix is processed bidirectionally (like BERT) and the remainder is generated autoregressively (like GPT). Used by models like T5 and UL2.

## Scaling Laws

Research from Kaplan et al. (2020) and Chinchilla (Hoffmann et al., 2022) established that LLM performance follows predictable power-law scaling relationships:

**Kaplan Scaling Laws**: Performance improves as a power law with model size, dataset size, and compute budget. Larger models are more sample-efficient — they achieve the same loss with less data per parameter.

**Chinchilla Optimal Scaling**: For a given compute budget, model size and training tokens should be scaled equally. This suggested that many models (like GPT-3) were undertrained — a smaller model trained on more data could achieve better performance with the same compute. Chinchilla (70B parameters, 1.4T tokens) outperformed the larger Gopher (280B parameters, 300B tokens).

**Implications**: These scaling laws guide resource allocation for training LLMs. They predict performance before training begins, enabling teams to estimate whether a proposed model will meet requirements. However, scaling laws don't account for emergent abilities that appear at certain scale thresholds.

## Key LLM Families

### GPT Series (OpenAI)

**GPT-1 (2018)**: Demonstrated that unsupervised pretraining followed by supervised fine-tuning is highly effective. 117M parameters.

**GPT-2 (2019)**: Showed that language models can perform downstream tasks without any fine-tuning (zero-shot). 1.5B parameters. Initially withheld from public release due to concerns about misuse.

**GPT-3 (2020)**: Demonstrated remarkable few-shot learning — providing a few examples in the prompt enables the model to perform new tasks without any gradient updates. 175B parameters trained on 300B tokens. Pioneered the in-context learning paradigm.

**GPT-4 (2024)**: Multimodal model accepting both text and images. Significantly improved reasoning, factuality, and instruction following. Rumored to use a mixture-of-experts architecture.

### LLaMA Series (Meta)

**LLaMA (2023)**: Open-weights model family (7B, 13B, 33B, 65B parameters) that demonstrated competitive performance with closed-source models. Popularized open-source LLM development.

**LLaMA 2 (2023)**: Improved version with expanded training data (2T tokens), longer context length (4096 tokens), and Grouped-Query Attention (GQA). Released with a commercial license. LLaMA 2 Chat variants were fine-tuned with RLHF.

**LLaMA 3 (2024)**: Significant improvements with 8B and 70B variants. Trained on 15T tokens with expanded vocabulary (128K tokens). Achieved state-of-the-art performance among open models.

### Mistral and Mixtral

**Mistral 7B (2023)**: Outperformed LLaMA 2 13B on all benchmarks despite being half the size. Introduced Sliding Window Attention (SWA) for efficient long-context processing and used Grouped-Query Attention (GQA).

**Mixtral 8x7B (2024)**: A Sparse Mixture of Experts (SMoE) model with 47B total parameters but only 13B active per token. Each token is routed to the top-2 experts out of 8. Matched GPT-3.5 performance while being much faster at inference.

### BERT and Encoder Models

**BERT (2019)**: Bidirectional Encoder Representations from Transformers. Pretrained with masked language modeling and next sentence prediction. Revolutionized NLP benchmarks and remains widely used for classification, NER, and embedding tasks. Available in Base (110M) and Large (340M) sizes.

**RoBERTa (2019)**: Improved BERT training with more data, longer training, larger batches, dynamic masking, and removal of next sentence prediction. Showed that BERT was significantly undertrained.

### Claude, Gemini, and Others

**Claude (Anthropic)**: Known for being helpful, harmless, and honest. Trained using Constitutional AI (RLAIF) where the model critiques and revises its own outputs according to a set of principles. Strong at long-form analysis and coding.

**Gemini (Google DeepMind)**: Natively multimodal models trained on text, code, images, audio, and video from the ground up. Available in Ultra, Pro, Flash, and Nano sizes.

## Training LLMs

### Pretraining Data

LLMs are pretrained on massive text corpora sourced from the internet, books, code repositories, and other text sources. Key datasets include:

- **Common Crawl**: Petabytes of web data collected over years of web crawling. Heavily filtered and deduplicated before use.
- **The Pile**: A curated 800GB dataset combining 22 diverse sources including Wikipedia, ArXiv papers, GitHub code, Stack Exchange, and books.
- **RedPajama**: An open-source recreation of the LLaMA training dataset, containing 1.2 trillion tokens.

Data quality is paramount — filtering and deduplication significantly impact model quality. Common filtering steps include language identification, quality scoring, deduplication (exact, near-duplicate, and fuzzy), toxic content removal, and personally identifiable information (PII) scrubbing.

### Tokenization

LLMs don't process raw text — they first convert text into tokens using a tokenizer. Modern tokenizers use subword tokenization algorithms:

**Byte-Pair Encoding (BPE)**: Iteratively merges the most frequent pairs of characters/subwords. Used by GPT models. Builds vocabulary bottom-up from individual characters.

**SentencePiece**: A language-independent tokenizer that treats the input as a raw byte stream. Used by LLaMA, T5, and many multilingual models.

**WordPiece**: Similar to BPE but uses likelihood rather than frequency for merge decisions. Used by BERT.

Vocabulary size is a key design choice. Larger vocabularies (32K-128K tokens) reduce sequence length but increase embedding table size. Smaller vocabularies are more parameter-efficient but create longer sequences.

### Training Infrastructure

Training LLMs requires massive computational resources:
- **GPUs**: NVIDIA A100 (80GB) and H100 (80GB) are the most commonly used. H100 provides ~3x the performance of A100 for transformer training.
- **Training Time**: GPT-3 (175B) was estimated to require ~3,600 petaflop/s-days. LLaMA 2 70B was trained on 2048 A100 GPUs for ~35 days.
- **Cost**: Training a model like GPT-4 is estimated to cost $50-100 million in compute alone.
- **Distributed Training**: Uses combinations of data, model, pipeline, and tensor parallelism, coordinated by frameworks like Megatron-LM, DeepSpeed, and FSDP.

## Fine-Tuning and Alignment

### Supervised Fine-Tuning (SFT)

After pretraining, models are fine-tuned on high-quality (instruction, response) pairs to follow instructions and engage in dialogue. The quality of this data matters far more than quantity — LIMA showed that just 1,000 carefully curated examples can produce a competitive chatbot.

### Reinforcement Learning from Human Feedback (RLHF)

RLHF aligns LLMs with human preferences through a three-step process:
1. **Collect comparisons**: Human annotators rank model outputs for the same prompt.
2. **Train a reward model**: A neural network learns to predict human preference scores.
3. **Optimize with RL**: The language model is fine-tuned using PPO (Proximal Policy Optimization) to maximize the reward model's score while staying close to the SFT model (using KL divergence penalty).

RLHF was instrumental in making ChatGPT significantly more helpful and less harmful than base GPT-3.

### Direct Preference Optimization (DPO)

DPO simplifies RLHF by eliminating the need for a separate reward model and RL optimization. Instead, it directly optimizes the language model using preference pairs, treating the problem as a classification task. DPO is simpler to implement, more stable to train, and achieves comparable or better results than RLHF.

### Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning of large models is prohibitively expensive. PEFT methods adapt models by training only a small number of additional parameters:

**LoRA (Low-Rank Adaptation)**: Decomposes weight updates into low-rank matrices (A × B), training only these small matrices while freezing the original weights. Typically adds only 0.1-1% additional parameters. QLoRA extends this by quantizing the base model to 4-bit, enabling fine-tuning of 65B models on a single 48GB GPU.

**Adapter Layers**: Inserts small trainable modules between existing layers. The adapter consists of a down-projection, nonlinearity, and up-projection, adding ~3-8% parameters.

**Prompt Tuning**: Learns a small set of continuous vectors (soft prompts) prepended to the input embedding, while the model weights remain frozen.

## Inference Optimization

### Quantization

Reducing the precision of model weights from 32-bit or 16-bit floating point to lower precision (8-bit, 4-bit, or even 2-bit integers). This reduces memory usage and speeds up inference with minimal quality loss.

**GPTQ**: Post-training quantization to 4-bit with ~1% accuracy loss. Processes weights in a specific order to minimize quantization error.

**AWQ (Activation-Aware Weight Quantization)**: Preserves precision for the most important weights (those activated by the most important tokens), achieving better quality than uniform quantization.

**GGUF**: A quantization format designed for CPU inference, used extensively by llama.cpp. Supports various quantization levels (Q4_0, Q4_K_M, Q5_K_M, Q8_0) with different quality-size tradeoffs.

### KV Cache

During autoregressive generation, the key and value tensors from previous tokens are cached to avoid redundant computation. The KV cache grows linearly with sequence length and batch size, often becoming the memory bottleneck during inference.

**Multi-Query Attention (MQA)**: Uses a single key and value head shared across all query heads, dramatically reducing KV cache size. Used by Falcon and PaLM.

**Grouped-Query Attention (GQA)**: A middle ground where groups of query heads share key-value heads. LLaMA 2 70B uses 8 KV heads for 64 query heads, reducing the KV cache by 8x while maintaining quality.

### Speculative Decoding

Uses a small, fast "draft" model to generate several candidate tokens, which are then verified in parallel by the large target model. This can speed up inference by 2-3x because verifying multiple tokens in parallel is cheaper than generating them one by one.

### Continuous Batching

Instead of waiting for all sequences in a batch to finish, continuous batching removes completed sequences and adds new ones dynamically. This significantly improves GPU utilization and throughput. Implemented by vLLM, TGI (Text Generation Inference), and other serving frameworks.

## Prompt Engineering

### Zero-Shot Prompting

Asking the model to perform a task without providing any examples. The model relies entirely on its pretraining knowledge. Works well for simple tasks and capable models.

### Few-Shot Prompting (In-Context Learning)

Providing a few examples of the desired input-output behavior in the prompt. The model learns the pattern from the examples and applies it to the new input. This is one of the most remarkable emergent abilities of large language models.

### Chain-of-Thought (CoT) Prompting

Instructing the model to break down its reasoning into explicit intermediate steps before arriving at an answer. This dramatically improves performance on arithmetic, commonsense reasoning, and symbolic reasoning tasks. Can be elicited by adding "Let's think step by step" or by providing examples with detailed reasoning chains.

### ReAct (Reasoning + Action)

Combines reasoning traces with task-specific actions. The model alternates between thinking (generating reasoning traces) and acting (interacting with external tools or APIs). This is the foundation of modern AI agent architectures.

### Retrieval-Augmented Generation (RAG)

Instead of relying solely on the model's parametric knowledge, RAG retrieves relevant documents from an external knowledge base and includes them in the prompt context. This:
- Grounds the model's responses in factual evidence
- Enables the model to access up-to-date information
- Reduces hallucination
- Allows knowledge updates without retraining

The RAG pipeline typically involves: query encoding → document retrieval → context construction → generation with citation.

## Evaluation and Benchmarks

### Common Benchmarks

**MMLU (Massive Multitask Language Understanding)**: 57 subjects spanning STEM, humanities, social sciences, and more. Tests broad knowledge and reasoning.

**HumanEval**: Evaluates code generation ability by measuring functional correctness. The model generates Python functions given a docstring.

**HellaSwag**: Tests commonsense reasoning through sentence completion. Adversarially constructed to be easy for humans but challenging for models.

**TruthfulQA**: Measures the truthfulness of model responses on questions where humans commonly hold incorrect beliefs.

**GPQA (Graduate-Level Google-Proof Q&A)**: Expert-level questions in physics, chemistry, and biology that require deep domain knowledge.

**MT-Bench**: Measures multi-turn conversation quality using GPT-4 as a judge. Tests instruction following, reasoning, and coherence across extended dialogues.

### Challenges in LLM Evaluation

Evaluating LLMs is notoriously difficult because:
- **Benchmark contamination**: Training data may contain benchmark test sets, inflating scores.
- **Metric limitations**: Automated metrics (BLEU, ROUGE) correlate poorly with human judgments of quality.
- **Task diversity**: No single benchmark captures the full range of LLM capabilities.
- **Instruction sensitivity**: Small changes in prompt wording can dramatically affect performance.
- **Human evaluation cost**: Human evaluation is the gold standard but expensive and time-consuming.

## Safety and Alignment

### Hallucination

LLMs can generate plausible-sounding but factually incorrect information. This is a fundamental challenge because the model generates text by predicting likely next tokens, not by reasoning about truth. Mitigation strategies include RAG (grounding in retrieved evidence), chain-of-thought prompting, training on factual data, and citation enforcement.

### Bias and Fairness

LLMs inherit biases from their training data, which reflects societal biases present in internet text. These biases can manifest as stereotypes, unfair treatment of demographic groups, and skewed representations. Addressing bias requires careful data curation, evaluation across demographic groups, and targeted fine-tuning.

### Constitutional AI (CAI)

Anthropic's approach to AI alignment where the model is trained to follow a set of principles (a "constitution"). The model critiques and revises its own outputs according to these principles, then is trained using RLAIF (RL from AI Feedback) where the AI's own judgments serve as the reward signal. This reduces reliance on human labelers and scales better.

### Red Teaming

Systematic adversarial testing to discover model vulnerabilities, biases, and harmful outputs. Red teams attempt prompt injection, jailbreaking, and other attacks to identify weaknesses before deployment. This is an essential practice for responsible AI deployment.
