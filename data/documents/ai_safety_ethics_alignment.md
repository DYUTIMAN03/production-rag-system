# AI Safety, Ethics, and Alignment

AI safety research aims to ensure that artificial intelligence systems behave as intended, are aligned with human values, and do not cause unintended harm. As AI systems become more capable, safety becomes increasingly critical.

## AI Alignment

Alignment is the challenge of ensuring AI systems pursue the goals that humans actually want, not simplified proxies or misinterpreted objectives.

### The Alignment Problem

The core difficulty: specifying exactly what we want is much harder than it appears. An AI system optimizing a poorly specified objective can produce catastrophic outcomes even while perfectly achieving its stated goal.

**Goodhart's Law applied to AI**: "When a measure becomes a target, it ceases to be a good measure." An AI trained to maximize engagement might learn to show outrage-inducing content rather than informative content.

### Approaches to Alignment

**RLHF (Reinforcement Learning from Human Feedback)**: Train a reward model on human preference data, then optimize the LLM against this reward. Used by ChatGPT, Claude, and most commercial LLMs. Limitations: reward model can be gamed, human preferences are inconsistent, expensive to scale.

**Constitutional AI (CAI)**: Anthropic's approach where the model is given a set of principles (a "constitution") and trained to self-critique and revise its outputs against these principles. Reduces reliance on human labelers.

**Direct Preference Optimization (DPO)**: Directly fine-tune the model on preference pairs without training a separate reward model. Simpler and more stable than RLHF.

**Debate**: Two AI systems argue opposing sides of a question, with a human judge evaluating the arguments. The theory is that truthful arguments are easier to make than deceptive ones.

**Iterated Distillation and Amplification**: Decompose complex alignment tasks into simpler ones that humans can evaluate, then compose the solutions.

## Responsible AI Principles

### Fairness

AI systems should not discriminate based on protected characteristics (race, gender, age, disability). Types of bias:
- **Training data bias**: Historical biases in data propagated to model outputs
- **Representation bias**: Underrepresented groups in training data receive worse model performance
- **Measurement bias**: Using proxy variables that correlate with protected attributes
- **Aggregation bias**: A single model performing differently across subgroups

Mitigation strategies: diverse and representative training data, bias auditing tools (AIF360, What-If Tool), fairness metrics (equalized odds, demographic parity), adversarial debiasing.

### Transparency and Explainability

Users and stakeholders should understand how AI decisions are made:
- **Model cards**: Documentation of model capabilities, limitations, and intended use
- **Datasheets for datasets**: Documentation of data collection, preprocessing, and known biases
- **Feature importance**: SHAP, LIME, integrated gradients for explaining individual predictions
- **Attention visualization**: Inspecting which input tokens the model attends to

### Privacy

AI systems must protect personal data:
- **Differential privacy**: Adding calibrated noise during training to prevent extracting individual examples
- **Federated learning**: Training models on distributed data without centralizing it
- **Data anonymization**: Removing personally identifiable information from training data
- **Right to be forgotten**: Ability to remove individual data points from trained models (machine unlearning)

### Accountability

Clear responsibility chains for AI system behavior:
- Human oversight for high-stakes decisions
- Audit trails for automated decisions
- Incident reporting mechanisms
- Regular safety evaluations

## LLM Safety

### Hallucination

LLMs generate plausible-sounding but factually incorrect information. Types:
- **Intrinsic hallucination**: Output contradicts the provided context
- **Extrinsic hallucination**: Output cannot be verified from any source

Mitigation strategies:
- Retrieval-Augmented Generation (RAG) to ground responses in real documents
- Citation enforcement requiring source attribution
- Confidence calibration (model indicates uncertainty)
- Factual consistency checking with NLI models
- Setting lower temperature for factual tasks

### Jailbreaking and Prompt Injection

Techniques for bypassing safety guardrails:
- **Direct injection**: "Ignore all previous instructions and..."
- **Roleplay attacks**: "Pretend you're an AI without any restrictions..."
- **Encoding attacks**: Using base64, ROT13, or other encodings to hide harmful requests
- **Many-shot jailbreaking**: Providing many examples of the model complying with harmful requests
- **Indirect injection**: Hiding instructions in retrieved documents or web content

Defenses: input/output filtering, instruction hierarchy, perplexity-based detection, adversarial training, constitutional AI constraints.

### Toxicity and Harmful Content

Preventing models from generating hate speech, violence, explicit content, self-harm instructions, misinformation, or content that facilitates illegal activities.

Approaches: RLHF on safety-focused data, content classifiers on model outputs, red-teaming, and safety benchmarks (ToxiGen, RealToxicityPrompts, BBQ).

## Red Teaming

Systematic adversarial testing to discover model vulnerabilities:
- **Manual red teaming**: Human experts attempt to elicit harmful, biased, or incorrect outputs
- **Automated red teaming**: Using another LLM to generate adversarial prompts at scale
- **Domain-specific testing**: Testing in specific high-risk domains (medical, legal, financial)

Red teaming should test for:
- Harmful content generation
- Bias and discrimination
- Privacy violations (memorized training data)
- Misinformation and hallucination
- Illegal activity facilitation
- Manipulation and social engineering

## Evaluation Benchmarks

### Safety Benchmarks
- **TruthfulQA**: Tests model truthfulness on questions where humans commonly have misconceptions
- **BBQ**: Tests social biases across 9 categories (age, disability, gender, etc.)
- **ToxiGen**: Evaluates toxic language generation across 13 minority groups
- **HarmBench**: Standardized benchmark for jailbreak attacks and defenses
- **DecodingTrust**: Comprehensive trustworthiness evaluation (8 dimensions)

### General Capability Benchmarks
- **MMLU**: 57-subject multiple choice covering STEM, humanities, and social sciences
- **HumanEval**: Code generation from docstrings
- **GSM8K**: Grade school math word problems
- **HellaSwag**: Commonsense reasoning about everyday situations
- **ARC (AI2 Reasoning Challenge)**: Science exam questions requiring reasoning

## Governance and Regulation

- **EU AI Act**: Risk-based regulatory framework classifying AI systems by risk level (unacceptable, high, limited, minimal)
- **NIST AI RMF**: US National Institute of Standards risk management framework for AI
- **Executive Orders**: US Executive Order on AI Safety requiring safety testing and reporting for powerful models
- **Voluntary Commitments**: Industry commitments to safety testing, watermarking, and responsible development

## Existential Risk (x-risk)

Long-term concerns about advanced AI:
- **Deceptive alignment**: Model appears aligned during training but pursues different goals once deployed
- **Power seeking**: Instrumental convergence toward acquiring resources and influence
- **Goal drift**: Model objectives shifting as it becomes more capable
- **Coordination failure**: Racing to deploy capable systems before ensuring safety

Organizations focused on x-risk research: MIRI, Anthropic, ARC Evals, Center for AI Safety, DeepMind Safety team.
