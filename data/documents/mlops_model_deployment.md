# MLOps and Model Deployment

MLOps (Machine Learning Operations) applies DevOps principles to machine learning systems, addressing the unique challenges of developing, deploying, and maintaining ML models in production.

## The ML Lifecycle

A production ML system involves much more than model training:

1. **Data collection and labeling**
2. **Data validation and preprocessing**
3. **Feature engineering and storage**
4. **Model training and experimentation**
5. **Model evaluation and validation**
6. **Model packaging and deployment**
7. **Monitoring and observability**
8. **Feedback loops and retraining**

Google's famous paper "Hidden Technical Debt in Machine Learning Systems" showed that model code is typically less than 5% of a production ML system. The remaining 95% is infrastructure: data pipelines, serving systems, monitoring, and configuration management.

## Experiment Tracking

Tracking experiments is essential for reproducibility and comparison:
- **MLflow**: Open-source platform for tracking experiments, packaging models, and deploying. Logs parameters, metrics, artifacts, and model versions.
- **Weights & Biases (W&B)**: Cloud-based experiment tracking with rich visualization, hyperparameter sweeps, and team collaboration.
- **Langfuse**: Observability platform specifically designed for LLM applications. Traces prompts, completions, token usage, and latency.
- **Neptune.ai**: Experiment tracking focused on metadata management and comparison.

Key items to track: hyperparameters, training metrics (loss, accuracy per epoch), evaluation metrics, model artifacts, data versions, code versions, and hardware configuration.

## Model Serving

### Serving Patterns

**Online (real-time) serving**: Model responds to individual requests with low latency (< 100ms). Used for interactive applications like chatbots, search, and recommendations.

**Batch serving**: Model processes large datasets periodically (hourly, daily). Used for precomputing recommendations, generating reports, or scoring customer databases.

**Streaming serving**: Model processes continuous data streams. Used for fraud detection, anomaly detection, and real-time event processing.

### Serving Frameworks

**TorchServe**: PyTorch's official serving framework. Supports model versioning, A/B testing, multi-model serving, and metrics.

**TensorFlow Serving**: High-performance serving for TensorFlow models. gRPC and REST APIs, batching, and model versioning.

**Triton Inference Server**: NVIDIA's universal inference server supporting PyTorch, TensorFlow, ONNX, and custom backends. Dynamic batching, model ensembles, and GPU optimization.

**vLLM**: High-throughput LLM serving with PagedAttention for efficient KV-cache memory management. Supports continuous batching and tensor parallelism. 2-4x higher throughput than HuggingFace's text-generation-inference.

**Ollama**: Simple tool for running LLMs locally. Handles model downloading, quantization, and serving with a simple API.

**FastAPI**: Lightweight Python web framework commonly used for custom ML serving. Easy to set up but lacks production features like batching and model management.

## Model Optimization for Inference

### Quantization

Reducing model precision from FP32 to lower bit-widths:
- **FP16/BF16**: 2x memory reduction with minimal quality loss. Standard for GPU inference.
- **INT8**: 4x memory reduction. Post-training quantization (PTQ) or quantization-aware training (QAT).
- **INT4/NF4**: 8x memory reduction. Used with QLoRA, GPTQ, AWQ. Some quality degradation on complex tasks.
- **GGUF**: Format for CPU-optimized quantized models used by llama.cpp. Supports 2-8 bit quantization.

### Pruning

Removing unnecessary weights or neurons from a model:
- **Unstructured pruning**: Zero out individual weights (achieves high sparsity but needs sparse hardware)
- **Structured pruning**: Remove entire neurons, heads, or layers (directly reduces model size)
- **Lottery Ticket Hypothesis**: Dense networks contain sparse subnetworks that can match full model performance

### Knowledge Distillation

Training a smaller "student" model to mimic a larger "teacher" model's behavior. The student learns from the teacher's soft output probabilities (which contain more information than hard labels). Commonly used to deploy smaller models in latency-sensitive or resource-constrained environments.

### ONNX Export

ONNX (Open Neural Network Exchange) is an interchangeable model format. Exporting to ONNX enables:
- Cross-framework compatibility (PyTorch → TensorFlow)
- Runtime optimization with ONNX Runtime (graph optimizations, operator fusion)
- Deployment on edge devices and specialized hardware

## Containerization and Orchestration

### Docker

Standard for packaging ML applications:
- Reproducible environments across dev, test, and production
- Include model weights, code, and dependencies in a single image
- Multi-stage builds to minimize image size

### Kubernetes

Container orchestration for ML workloads:
- **Horizontal Pod Autoscaler**: Scale serving pods based on request rate or GPU utilization
- **GPU scheduling**: Allocate GPU resources to pods
- **Rolling updates**: Zero-downtime model version updates
- **KServe**: Kubernetes-native ML serving with autoscaling, canary deployments, and multi-framework support

## CI/CD for ML

Continuous Integration and Deployment adapted for ML:
- **Code tests**: Unit tests for data preprocessing, feature engineering, and model code
- **Data validation**: Schema checks, distribution drift detection, missing value monitoring
- **Model validation**: Performance thresholds on evaluation datasets (e.g., accuracy > 0.9)
- **Shadow deployment**: Run new model alongside production model, compare outputs without serving to users
- **Canary deployment**: Gradually route traffic to new model (1% → 10% → 50% → 100%)
- **A/B testing**: Route traffic to different models and compare business metrics

## Monitoring and Observability

### Data Drift

Monitor changes in input data distribution over time:
- **Feature drift**: Statistical properties of input features change
- **Label drift**: Distribution of target variable changes
- Tools: Evidently AI, WhyLabs, Great Expectations

### Model Performance Degradation

Production model accuracy tends to degrade over time as the world changes:
- Track prediction confidence distributions
- Compare online metrics against offline evaluation baselines
- Set up alerts for significant drops in key metrics
- Implement automated retraining triggers

### LLM-Specific Monitoring

For LLM applications, monitor:
- **Token usage and cost**: Track per-request and aggregate costs
- **Latency percentiles**: P50, P95, P99 response times
- **Hallucination rate**: Compare responses against known ground truth
- **Prompt version performance**: A/B test different prompt versions
- **User feedback signals**: Thumbs up/down, regeneration rate, follow-up questions

## Feature Stores

Feature stores manage features used across training and serving:
- **Feast**: Open-source feature store supporting batch and real-time features
- **Tecton**: Managed feature platform with real-time feature engineering
- **Hopsworks**: Feature store with versioning and data lineage

Key capabilities: feature versioning, point-in-time-correct joins (avoiding data leakage), online/offline serving parity, feature sharing across teams.
