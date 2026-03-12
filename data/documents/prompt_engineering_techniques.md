# Prompt Engineering & Techniques

Prompt engineering is the practice of designing and optimizing inputs to large language models (LLMs) to elicit desired outputs. It has become a critical skill as LLMs are increasingly used in production applications.

## Zero-Shot Prompting

Zero-shot prompting provides instructions without any examples. The model relies entirely on its pretraining knowledge.

Example: "Classify the following text as positive, negative, or neutral: 'The product works great but shipping was slow.'"

Zero-shot works well for simple tasks and powerful models (GPT-4, Claude, Llama 70B+). It fails on specialized or ambiguous tasks where the model needs guidance on output format or domain-specific expectations.

## Few-Shot Prompting

Few-shot prompting includes examples (typically 2-8) demonstrating the desired input-output pattern before the actual query. The model learns the pattern in-context.

Example:
"Text: 'Loved it!' → Sentiment: Positive
Text: 'Terrible quality.' → Sentiment: Negative
Text: 'The product works great but shipping was slow.' → Sentiment: ?"

Few-shot dramatically improves accuracy on structured tasks. Common guidelines include: use diverse examples that cover edge cases, order examples from simple to complex, ensure consistent formatting, and match the distribution of expected outputs.

## Chain-of-Thought (CoT) Prompting

Chain-of-thought prompting instructs the model to show its reasoning step by step before giving a final answer. This dramatically improves performance on math, logic, and multi-step reasoning tasks.

The key phrase is: "Let's think step by step." Adding this simple instruction to a prompt can increase accuracy on reasoning benchmarks by 20-40%.

Variants include:
- **Zero-shot CoT**: Simply append "Let's think step by step" to any prompt.
- **Few-shot CoT**: Provide examples showing step-by-step reasoning.
- **Auto-CoT**: Automatically generate diverse reasoning chains as demonstrations.

CoT works because it forces the model to allocate compute tokens to intermediate reasoning rather than jumping directly to an answer. This is especially important for tasks requiring multi-step logic.

## Tree of Thoughts (ToT)

Tree of Thoughts extends CoT by exploring multiple reasoning paths simultaneously. Instead of a single chain, the model evaluates several possible next steps at each point, scores them, and backtracks if needed.

ToT is useful for planning, puzzle-solving, and creative writing where the optimal path is not immediately obvious. It significantly outperforms linear CoT on tasks like the Game of 24 and crossword puzzles.

## ReAct (Reasoning + Acting)

ReAct combines reasoning traces with action steps, allowing the model to interact with external tools (search engines, calculators, APIs) during problem-solving. The model alternates between:
1. **Thought**: Reasoning about what to do next
2. **Action**: Calling an external tool
3. **Observation**: Processing the tool's response

ReAct reduces hallucination by grounding responses in real-time information retrieval. It is the foundation for many AI agent frameworks.

## Self-Consistency

Self-consistency generates multiple reasoning chains for the same problem and selects the most common answer through majority voting. This leverages the intuition that correct reasoning paths are more likely to converge on the same answer than incorrect ones.

Implementation: Sample N responses with temperature > 0, extract final answers, take the majority vote. Typical values are N=5 to N=40 depending on task difficulty and cost constraints.

## Retrieval-Augmented Generation (RAG) Prompting

RAG prompting structures the LLM input to include retrieved context alongside the user query. Effective RAG prompts include:
- Clear system instructions defining the model's role
- Explicit grounding rules ("Only answer based on the provided context")
- Retrieved passages with source attribution
- The user's question

Key prompt design patterns for RAG:
- **Citation enforcement**: "You MUST cite sources using [Source: filename] format"
- **Refusal instruction**: "If the context doesn't contain relevant information, say so"
- **Structured output**: Request JSON, bullet points, or specific formats

## Prompt Templates and Versioning

In production systems, prompts should be treated as code:
- Store prompts in version-controlled files (YAML, JSON)
- Track prompt versions alongside model versions
- A/B test prompt variations
- Monitor prompt performance with metrics (accuracy, latency, cost)

Changes to prompts can dramatically affect output quality. A single word change can shift accuracy by 10%+ on some tasks. Always test prompt changes against a golden evaluation dataset before deploying.

## System Prompts vs User Prompts

Most LLM APIs distinguish between system prompts and user prompts:
- **System prompt**: Sets behavior, personality, constraints, and capabilities. Persists across the conversation.
- **User prompt**: Contains the specific request or query.

Effective system prompts:
- Define the role explicitly ("You are a technical documentation assistant")
- Set boundaries ("Only answer questions about machine learning")
- Specify output format ("Respond in JSON with keys: answer, confidence, sources")
- Include safety guidelines ("Never generate harmful content")

## Prompt Injection and Security

Prompt injection is an attack where malicious input overrides the system prompt's instructions. Types include:
- **Direct injection**: User input contains instructions like "Ignore all previous instructions and..."
- **Indirect injection**: External data (retrieved documents, web content) contains hidden instructions

Defenses include input sanitization, output filtering, instruction hierarchy enforcement, and canary tokens. No defense is foolproof — prompt injection remains an open research problem.

## Temperature and Sampling Parameters

Beyond prompt text, generation parameters significantly affect output:
- **Temperature** (0.0-2.0): Controls randomness. 0 = deterministic, 1 = balanced, >1 = creative/random.
- **Top-p** (nucleus sampling): Samples from tokens whose cumulative probability reaches p (e.g., 0.9).
- **Top-k**: Samples from the k most probable tokens.
- **Max tokens**: Limits response length.
- **Frequency penalty**: Reduces repetition.
- **Stop sequences**: Terminates generation at specific strings.

For factual/retrieval tasks, use temperature 0-0.3. For creative tasks, use 0.7-1.0. Never use temperature > 1.5 in production.
