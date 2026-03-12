# Reinforcement Learning

Reinforcement Learning (RL) is a paradigm of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative reward. Unlike supervised learning, the agent is not told which actions to take but must discover which actions yield the most reward through trial and error.

## Core Concepts

### The RL Framework

The standard RL setup consists of:
- **Agent**: The learner and decision maker
- **Environment**: Everything the agent interacts with
- **State (s)**: The current situation as observed by the agent
- **Action (a)**: A choice made by the agent
- **Reward (r)**: A scalar feedback signal from the environment
- **Policy (π)**: The agent's strategy — a mapping from states to actions
- **Value function V(s)**: Expected cumulative reward from a state following the current policy
- **Q-function Q(s,a)**: Expected cumulative reward from taking action a in state s

The interaction follows the Markov Decision Process (MDP): at each timestep, the agent observes state s, takes action a, receives reward r, and transitions to new state s'. The goal is to find a policy that maximizes the expected cumulative discounted reward: E[Σ γ^t * r_t], where γ (gamma) is the discount factor (typically 0.99).

### Exploration vs Exploitation

The fundamental dilemma in RL: should the agent exploit known good actions or explore unknown actions that might be better? Strategies include:
- **ε-greedy**: With probability ε, take a random action; otherwise take the best known action. Decay ε over time.
- **Upper Confidence Bound (UCB)**: Favor actions with high uncertainty, balancing optimism and data.
- **Thompson Sampling**: Sample from posterior distributions over action values.
- **Entropy regularization**: Add an entropy bonus to encourage diverse behavior (used in SAC and PPO).

## Value-Based Methods

### Q-Learning

Q-learning learns the optimal action-value function Q*(s,a) using the Bellman equation: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]. It is off-policy (can learn from any data, not just the current policy) and guaranteed to converge to optimal values in tabular settings.

### Deep Q-Networks (DQN)

DQN uses a neural network to approximate Q(s,a), enabling Q-learning in high-dimensional state spaces (like Atari game pixels). Key innovations:
- **Experience replay**: Store transitions in a buffer and sample randomly for training, breaking temporal correlations
- **Target network**: Use a separate, periodically updated network for computing target values, improving stability
- **Reward clipping**: Clip rewards to [-1, 1] to standardize learning across different games

DQN achieved superhuman performance on many Atari games (2015), launching the modern deep RL era.

### Double DQN

Standard DQN overestimates Q-values because it uses the max operator for both action selection and evaluation. Double DQN decouples these by using the online network to select actions and the target network to evaluate them, significantly reducing overestimation.

### Dueling DQN

Dueling DQN separates the Q-value into state value V(s) and advantage A(s,a): Q(s,a) = V(s) + A(s,a). This helps the network learn which states are valuable independently from which actions are valuable in each state.

### Rainbow DQN

Rainbow combines all major DQN improvements: Double DQN, Dueling, Prioritized Experience Replay, Multi-step returns, Distributional RL (C51), and Noisy Networks. It significantly outperforms any individual improvement.

## Policy Gradient Methods

### REINFORCE

REINFORCE directly optimizes the policy by estimating policy gradients from sampled trajectories. The gradient is: ∇J = E[Σ ∇log π(a|s) * G], where G is the return (cumulative discounted reward). It is simple but has high variance and requires complete episodes.

### Actor-Critic Methods

Actor-critic methods combine:
- **Actor**: A policy network that selects actions
- **Critic**: A value network that evaluates states/actions

The critic reduces variance compared to pure policy gradients by providing a baseline. The advantage function A(s,a) = Q(s,a) - V(s) is used instead of raw returns.

### A3C and A2C

Asynchronous Advantage Actor-Critic (A3C) runs multiple agents in parallel, each in its own environment copy, asynchronously updating shared parameters. A2C is the synchronous version that waits for all workers to finish before updating, and often performs equally well with simpler implementation.

### PPO (Proximal Policy Optimization)

PPO is the most widely used RL algorithm in practice. It constrains policy updates to prevent destructive large changes using a clipped surrogate objective:
L = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

where r(θ) is the probability ratio between new and old policies, and ε is typically 0.2.

PPO is used for:
- RLHF in LLM training (ChatGPT, Claude)
- Game AI (OpenAI Five for Dota 2, OpenAI's hide-and-seek)
- Robotics control
- Recommendation systems

### SAC (Soft Actor-Critic)

SAC maximizes both expected reward and entropy (randomness) of the policy: J = E[Σ r_t + α * H(π(·|s_t))]. The entropy bonus encourages exploration and makes training more robust. SAC is the standard choice for continuous control tasks (robotics, simulation).

## Model-Based RL

Model-based methods learn a model of the environment (transition dynamics and reward function) and use it for planning, rather than learning purely from trial and error.

### World Models

World Models learn a compressed representation of the environment and a predictive model of future states. The agent can then "imagine" trajectories and plan without interacting with the real environment. Applications include autonomous driving simulation and video game AI.

### MuZero

DeepMind's MuZero learns a model, value function, and policy end-to-end, achieving superhuman performance on board games (Go, Chess) and Atari without knowing the rules. It plans using Monte Carlo Tree Search (MCTS) with the learned model.

## Multi-Agent RL

Multi-agent RL studies settings with multiple interacting agents, introducing challenges like non-stationarity (other agents are learning simultaneously) and coordination.

Key paradigms:
- **Cooperative**: Agents share a common reward (e.g., robot teams)
- **Competitive**: Agents have opposing rewards (e.g., game playing)
- **Mixed**: Both cooperative and competitive elements (e.g., traffic, economics)

## RL for LLMs

Reinforcement learning is increasingly used to improve large language models:
- **RLHF**: Aligning model outputs with human preferences (PPO on reward model scores)
- **RLAIF**: Using AI feedback instead of human feedback for the reward model
- **Constitutional AI**: Self-improvement through self-critique and revision
- **Process Reward Models**: Rewarding intermediate reasoning steps rather than just final answers

## Challenges in RL

- **Sample efficiency**: RL typically requires millions of environment interactions
- **Sparse rewards**: Many environments only provide reward at the end of long episodes
- **Sim-to-real gap**: Policies trained in simulation often fail in the real world
- **Reward hacking**: Agents find unintended ways to maximize reward without solving the intended task
- **Credit assignment**: Determining which actions were responsible for delayed rewards
