# Generative AI: Diffusion Models, GANs, and VAEs

Generative AI encompasses models that create new content — images, text, audio, video, and code — by learning the underlying distribution of training data. This document covers the three major architectures for generative modeling beyond language models.

## Generative Adversarial Networks (GANs)

GANs, introduced by Ian Goodfellow in 2014, use two neural networks competing in a minimax game:
- **Generator (G)**: Creates fake samples from random noise
- **Discriminator (D)**: Classifies samples as real or fake

Training alternates between updating D to better distinguish real from fake, and updating G to better fool D. At equilibrium, the generator produces samples indistinguishable from real data.

### GAN Variants

**DCGAN (Deep Convolutional GAN)**: Uses convolutional layers, batch normalization, and specific architectural guidelines for stable training. Established best practices for GAN architecture design.

**StyleGAN (1, 2, 3)**: Progressive growing architecture that generates photorealistic human faces. Introduced the style-based generator with an adaptive instance normalization (AdaIN) mechanism that controls high-level attributes (pose, identity) and fine details (freckles, hair texture) separately. StyleGAN3 eliminates aliasing artifacts.

**CycleGAN**: Unpaired image-to-image translation (e.g., horse→zebra, summer→winter) using cycle consistency loss: translating A→B→A should reconstruct the original.

**Pix2Pix**: Paired image-to-image translation using conditional GANs. Requires matched input-output pairs (e.g., sketch→photo).

**WGAN (Wasserstein GAN)**: Replaces the discriminator with a critic that estimates the Wasserstein distance between distributions, providing smoother gradients and more stable training.

### GAN Challenges

- **Mode collapse**: Generator produces limited variety, ignoring parts of the data distribution
- **Training instability**: Sensitive to hyperparameters, prone to oscillation
- **Evaluation difficulty**: No single metric captures generation quality; FID and IS are commonly used
- **No density estimation**: Cannot compute the probability of a given sample

## Variational Autoencoders (VAEs)

VAEs are probabilistic generative models that learn a compressed latent representation of data. Unlike regular autoencoders, VAEs impose a prior distribution (typically Gaussian) on the latent space, enabling generation of new samples.

### Architecture

- **Encoder (q(z|x))**: Maps input x to a distribution in latent space (outputs mean μ and variance σ²)
- **Decoder (p(x|z))**: Reconstructs input from a latent sample z ~ N(μ, σ²)

The loss function combines:
- **Reconstruction loss**: How well the decoder recreates the input
- **KL divergence**: How close the latent distribution is to the prior N(0, 1)

L = -E[log p(x|z)] + KL(q(z|x) || p(z))

### VAE Variants

**β-VAE**: Introduces a weight β > 1 on the KL term to encourage disentangled latent representations, where individual latent dimensions correspond to meaningful factors of variation.

**VQ-VAE (Vector Quantized VAE)**: Uses discrete latent codes from a learned codebook instead of continuous Gaussians. Produces sharper outputs and forms the basis for models like DALL-E's image tokenizer.

**CVAE (Conditional VAE)**: Conditions generation on additional information (class labels, attributes), enabling controlled generation.

### Comparison with GANs

| Aspect | GAN | VAE |
|---|---|---|
| Sample quality | Higher (sharper) | Lower (blurrier) |
| Training stability | Unstable | Stable |
| Latent space | Unstructured | Structured, continuous |
| Density estimation | No | Yes (ELBO) |
| Mode coverage | Poor (mode collapse) | Good |

## Diffusion Models

Diffusion models are the current state-of-the-art for image generation. They work by learning to reverse a gradual noising process.

### Forward Process (Noising)

Starting from a clean image x₀, progressively add Gaussian noise over T timesteps until the image becomes pure noise: x_t = √(ᾱ_t) * x₀ + √(1-ᾱ_t) * ε, where ε ~ N(0, I) and ᾱ_t follows a noise schedule.

### Reverse Process (Denoising)

A neural network (typically a U-Net) learns to predict the noise added at each step, enabling iterative denoising from pure noise back to a clean image. The model is trained with a simple L2 loss: ||ε - ε_θ(x_t, t)||².

### Key Models

**DDPM (Denoising Diffusion Probabilistic Models)**: The foundational diffusion model paper. Uses 1000 timesteps, slow generation.

**DDIM (Denoising Diffusion Implicit Models)**: Enables deterministic sampling and fewer steps (50-100) with minimal quality loss.

**Stable Diffusion (Latent Diffusion Models)**: Performs diffusion in a lower-dimensional latent space (encoded by a VQ-VAE) rather than pixel space, dramatically reducing compute. Uses a U-Net with cross-attention for text conditioning.

**DALL-E 2/3**: OpenAI's text-to-image models. DALL-E 2 uses CLIP embeddings + diffusion; DALL-E 3 uses improved captioning and diffusion.

**Midjourney**: Commercial text-to-image model known for artistic, aesthetically pleasing outputs.

**Imagen**: Google's text-to-image model using a large language model (T5-XXL) as the text encoder.

### Conditioning Mechanisms

**Classifier-Free Guidance (CFG)**: During training, randomly drop conditioning information. At inference, interpolate between conditional and unconditional predictions: ε = ε_unconditional + w * (ε_conditional - ε_unconditional), where w is the guidance scale (typically 7-15).

Higher guidance scale produces outputs more aligned with the prompt but reduces diversity. Values of 7-9 balance quality and diversity.

**ControlNet**: Adds spatial conditioning (edge maps, depth maps, pose skeletons) to pretrained diffusion models without fine-tuning the base model.

**IP-Adapter**: Enables image prompt conditioning alongside text prompts for style transfer and reference-based generation.

## Evaluation Metrics

**FID (Fréchet Inception Distance)**: Measures the distance between feature distributions of real and generated images. Lower is better. The standard metric for image generation quality.

**IS (Inception Score)**: Measures quality and diversity of generated images using a pretrained classifier. Higher is better. Less reliable than FID.

**CLIP Score**: Measures alignment between generated images and text prompts using CLIP embeddings. Used for text-to-image models.

**LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual similarity between images using deep features. Used for image reconstruction tasks.

## Applications

- **Text-to-image generation**: Creating images from natural language descriptions
- **Image editing**: Inpainting, outpainting, style transfer
- **Super-resolution**: Upscaling low-resolution images
- **Video generation**: Sora, Runway Gen-2, Stable Video Diffusion
- **Audio generation**: Music creation, text-to-speech
- **3D generation**: Point clouds, NeRFs, 3D meshes from text or images
- **Drug discovery**: Generating molecular structures with desired properties
- **Protein design**: Designing novel protein structures (RFdiffusion)
