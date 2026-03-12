# Deep Learning and Neural Networks

## What is Deep Learning?

Deep learning is a subset of machine learning based on artificial neural networks with multiple layers (hence "deep"). While traditional machine learning requires manual feature engineering, deep learning automatically discovers the representations needed for detection or classification directly from raw data. This ability to learn hierarchical features — from low-level patterns (edges, textures) to high-level concepts (faces, objects, words) — is what makes deep learning so powerful.

Deep learning has driven breakthroughs in computer vision, natural language processing, speech recognition, game playing, protein structure prediction (AlphaFold), and generative AI. The three key factors behind the deep learning revolution are: massive datasets, increased computational power (GPUs/TPUs), and algorithmic innovations.

## Neural Network Architecture

### Neurons and Layers

An artificial neuron receives inputs, applies weights to them, sums the weighted inputs with a bias term, and passes the result through an activation function. Mathematically: output = activation(Σ(weights × inputs) + bias).

A neural network is organized into layers:
- **Input Layer**: Receives the raw input data. The number of neurons equals the number of input features.
- **Hidden Layers**: Intermediate layers that learn increasingly abstract representations. The depth (number of hidden layers) and width (neurons per layer) are key architectural choices.
- **Output Layer**: Produces the final prediction. For classification, it typically has one neuron per class with a softmax activation. For regression, it usually has a single neuron with linear activation.

### Activation Functions

Activation functions introduce non-linearity into the network, enabling it to learn complex patterns. Without activation functions, a deep network would be equivalent to a single linear transformation.

**ReLU (Rectified Linear Unit)**: f(x) = max(0, x). The most widely used activation function due to its simplicity and effectiveness. It helps mitigate the vanishing gradient problem but can cause "dying ReLU" where neurons permanently output zero.

**Leaky ReLU**: f(x) = x if x > 0, else αx (where α is small, like 0.01). Solves the dying ReLU problem by allowing a small gradient for negative inputs.

**GELU (Gaussian Error Linear Unit)**: Used extensively in transformer architectures (BERT, GPT). It provides a smooth approximation to ReLU and empirically performs better in many NLP tasks.

**Sigmoid**: f(x) = 1/(1 + e^(-x)). Outputs values between 0 and 1. Used in the output layer for binary classification but rarely in hidden layers due to vanishing gradients and non-zero-centered outputs.

**Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)). Outputs values between -1 and 1. Zero-centered, making it preferred over sigmoid in hidden layers, but still suffers from vanishing gradients.

**Swish**: f(x) = x × sigmoid(x). Self-gated activation that often outperforms ReLU in deep networks. Used in EfficientNet and other modern architectures.

### Backpropagation and Gradient Descent

Backpropagation is the algorithm used to compute gradients of the loss function with respect to each weight in the network. It works by applying the chain rule of calculus, propagating the error backward from the output layer through the hidden layers.

**Stochastic Gradient Descent (SGD)** updates weights using gradients computed on a single sample or small batch. It's noisy but can escape local minima.

**Mini-batch Gradient Descent** computes gradients on small batches (typically 32-256 samples), balancing computational efficiency with gradient accuracy.

**Adam (Adaptive Moment Estimation)** combines momentum (using exponential moving averages of gradients) with adaptive learning rates (using exponential moving averages of squared gradients). It's the most popular optimizer due to its robust performance across a wide range of tasks and hyperparameter settings. The default hyperparameters (learning rate = 0.001, β1 = 0.9, β2 = 0.999) work well in most cases.

**AdamW** decouples weight decay from the gradient update, providing better regularization. It's the standard optimizer for training transformer models and large language models.

**Learning Rate Scheduling** adjusts the learning rate during training. Common strategies include step decay, cosine annealing, warmup followed by decay, and one-cycle policy. Learning rate warmup is especially important for training transformers.

## Convolutional Neural Networks (CNNs)

CNNs are specialized architectures for processing grid-structured data, primarily images. They exploit spatial locality and translation invariance through convolutional operations.

### Core Operations

**Convolution**: A learnable filter (kernel) slides across the input, computing the dot product at each position to produce a feature map. Each filter detects a specific pattern (edge, texture, shape). Stacking convolutional layers allows the network to learn a hierarchy of features.

**Pooling**: Reduces the spatial dimensions of feature maps, providing translation invariance and reducing computational cost. Max pooling takes the maximum value in each pooling window. Average pooling takes the mean. Modern architectures increasingly use strided convolutions instead of explicit pooling.

**Batch Normalization**: Normalizes the inputs to each layer, stabilizing and accelerating training. It reduces internal covariate shift and acts as a mild regularizer.

### Landmark CNN Architectures

**LeNet-5 (1998)**: Yann LeCun's pioneering architecture for handwritten digit recognition. Five layers with convolutional and pooling operations.

**AlexNet (2012)**: Won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) by a significant margin, catalyzing the deep learning revolution. Key innovations: ReLU activations, dropout regularization, GPU training, and data augmentation.

**VGGNet (2014)**: Demonstrated that depth matters by using very small (3×3) convolution filters in a deep architecture (16-19 layers). Showed that deeper networks with smaller filters outperform shallower networks with larger filters.

**ResNet (2015)**: Introduced residual connections (skip connections) that allow gradients to flow directly through the network, enabling training of extremely deep networks (50, 101, 152+ layers). The residual learning framework solved the degradation problem where deeper networks performed worse than shallow ones.

**EfficientNet (2019)**: Introduced compound scaling — simultaneously scaling network depth, width, and resolution using a fixed ratio. Achieved state-of-the-art accuracy with significantly fewer parameters than previous architectures.

### Transfer Learning

Transfer learning leverages knowledge from a model trained on a large dataset (like ImageNet with 14 million images) and applies it to a different but related task, even with limited data. This is one of the most practically important techniques in deep learning.

Common approaches include:
- **Feature Extraction**: Use the pretrained model as a fixed feature extractor by removing the classification head and training a new classifier on top.
- **Fine-tuning**: Unfreeze some or all layers of the pretrained model and train them with a very low learning rate on the new task. Typically, earlier layers (which learn generic features like edges and textures) are frozen while later layers are fine-tuned.

## Recurrent Neural Networks (RNNs)

RNNs are designed for sequential data where the order of elements matters, such as text, time series, and speech. They maintain a hidden state that captures information from previous time steps.

### Vanilla RNNs

The basic RNN updates its hidden state at each time step using the current input and the previous hidden state. However, vanilla RNNs suffer from the vanishing gradient problem — gradients diminish exponentially as they're propagated back through many time steps, making it impossible to learn long-range dependencies.

### Long Short-Term Memory (LSTM)

LSTMs solve the vanishing gradient problem by introducing a cell state and gating mechanisms. Three gates control information flow:
- **Forget Gate**: Decides what information to discard from the cell state.
- **Input Gate**: Decides what new information to store in the cell state.
- **Output Gate**: Decides what to output based on the cell state.

LSTMs can learn dependencies spanning hundreds of time steps and were the dominant architecture for NLP before transformers.

### Gated Recurrent Unit (GRU)

GRUs are a simplified variant of LSTMs with two gates (update and reset) instead of three. They have fewer parameters and are faster to train, with comparable performance to LSTMs on many tasks.

## Regularization Techniques

### Dropout

Dropout randomly sets a fraction (typically 20-50%) of neurons to zero during training. This prevents co-adaptation between neurons, forcing the network to learn redundant representations. At inference time, all neurons are used but their outputs are scaled accordingly. Dropout is one of the most effective regularization techniques for deep networks.

### Weight Decay (L2 Regularization)

Weight decay adds a penalty proportional to the squared magnitude of weights to the loss function. This encourages smaller weights and simpler models, reducing overfitting. In AdamW, weight decay is applied directly to the weights rather than through the gradient.

### Data Augmentation

Artificially increasing the training set size by applying transformations to existing data. For images: random cropping, horizontal flipping, rotation, color jittering, cutout, and mixup. For text: synonym replacement, random insertion, random swap, and back-translation. Data augmentation is one of the most effective ways to improve model generalization.

### Early Stopping

Monitoring the validation loss during training and stopping when it begins to increase (while training loss continues to decrease). This simple technique effectively prevents overfitting by selecting the model checkpoint with the best generalization performance.

### Label Smoothing

Instead of using hard target labels (0 or 1), label smoothing replaces them with soft targets (0.1 and 0.9, for example). This prevents the model from becoming overconfident and improves calibration and generalization.

## Modern Deep Learning Practices

### Mixed Precision Training

Using 16-bit floating point (FP16 or BF16) for most computations while keeping critical operations in 32-bit (FP32). This approximately doubles training speed and halves memory usage with minimal accuracy impact. BFloat16 (BF16) is particularly popular for training large models because it has the same exponent range as FP32, avoiding overflow/underflow issues.

### Gradient Accumulation

When the desired batch size is too large to fit in GPU memory, gradient accumulation simulates larger batches by accumulating gradients over multiple forward-backward passes before updating weights. Effective batch size = micro-batch size × accumulation steps × number of GPUs.

### Distributed Training

Training models across multiple GPUs or machines. **Data Parallelism** splits the batch across GPUs, with each GPU processing a subset. **Model Parallelism** splits the model across GPUs when it's too large to fit on a single device. **Pipeline Parallelism** splits the model into sequential stages across GPUs. **Tensor Parallelism** splits individual layers across GPUs.

### Knowledge Distillation

Training a smaller "student" model to mimic a larger "teacher" model. The student learns from the teacher's soft probability outputs (which contain more information than hard labels) and can achieve comparable performance with far fewer parameters. This is widely used for model compression and deployment on edge devices.
