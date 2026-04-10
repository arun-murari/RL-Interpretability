# RL Policy Interpretability with Sparse Autoencoders

Investigating whether reinforcement learning policies exhibit superposition and polysemanticity, analogous to LLM internals.

## Overview

Trained a PPO agent on HalfCheetah-v4, extracted hidden layer activations, and applied sparse autoencoder (SAE) analysis to decompose learned representations into interpretable features. Found evidence of both monosemantic features (encoding specific behaviors) and polysemantic features (encoding combinations of behaviors).

## Key Findings

| Feature | Interpretation | Evidence |
|---------|----------------|----------|
| 668 | "Sprinting" — fast + low posture | x-vel = 3.33 (std 0.08), z-pos = -0.54 |
| 404 | "Low posture" — crouched, any speed | z-pos = -0.53 (std 0.08), x-vel varies |
| 333, 361, 442 | Leg extended | joint2 ≈ -0.1 (std < 0.2) |
| 938 | Leg contracted | joint2 ≈ 0.31 |

Feature 668 is notably monosemantic — it fires in a very specific behavioral context (low std across multiple state dimensions). Feature 404 is more polysemantic, firing across varied speeds.

## Method

1. **Train PPO** on HalfCheetah-v4 (continuous control, 17-dim state, 6-dim action)
2. **Collect activations** from the 256-dim hidden layer across 20k+ timesteps
3. **Train SAE** (256 → 1024 dims, BatchTopK k=32) to reconstruct activations through sparse bottleneck
4. **Analyze features** by finding top-activating samples for each of 1024 SAE features and checking correlations with state dimensions

## Project Structure

```
├── agents/
│   ├── ppo.py              # PPO implementation + activation collection
│   ├── activations.pt      # Saved activations (20480 x 256)
│   └── states.pt           # Corresponding states (20480 x 17)
├── interpretability/
│   ├── sparseautoencoder.py    # SAE architecture + training
│   └── sae_trained.pt          # Trained SAE weights
├── experiments/
│   └── analysis.py         # Feature interpretation analysis
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install torch gymnasium mujoco

# Train PPO and collect activations
python agents/ppo.py

# Train SAE
python interpretability/sparseautoencoder.py

# Analyze features
python experiments/analysis.py
```

## References

- [Polysemanticity and Capacity in Neural Networks](https://arxiv.org/abs/2210.01892) — Scherlis, Sachan et al.
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — Elhage et al. (Anthropic)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — Anthropic
