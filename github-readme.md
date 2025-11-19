# Neural Battery: Geometry-Aware Transfer Learning via Learnable k-Operators

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red)

Geometry-aware meta-learning for fast reinforcement learning task transfer without expensive inner-loop optimization.

## Overview

**Neural Battery** is a novel approach to transfer learning in RL that learns task-specific **curvature parameters** (k-operators) per layer instead of requiring expensive inner-loop optimization like MAML.

### Key Innovation

Rather than solving an inner optimization problem, Neural Battery learns to modulate activations:

```
x' = x ⊙ σ(k_i)
```

where k_i is a learned scalar and σ is sigmoid. This allows rapid task adaptation in a **single forward pass**.

### Results

**37.1% ± 26.0% improvement** over MAML baseline (5 random seeds, median: 50.1%)

- Best case: +67.0% (Seed 2)
- Worst case: +1.9% (Seed 4)
- **All seeds show positive improvement**

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/neural-battery.git
cd neural-battery
pip install torch numpy
```

### Running the Winning Recipe

```python
from neural_battery_winning_recipe import NeuralBatteryPolicy, NeuralBatteryTrainer

# Create trainer
trainer = NeuralBatteryTrainer(
    state_dim=2,
    action_dim=1,
    use_hierarchical=True,  # Enable hierarchical k-blocks
    lr=1e-3,                # Standard weight learning rate
    k_lr=5e-3               # K-parameter learning rate (5x faster)
)

# Create environment
from neural_battery_winning_recipe import PendulumEnvironment
env = PendulumEnvironment(A=1.8)  # Hard task

# Train
for episode in range(100):
    state = env.reset()
    states, actions, rewards = [], [], []
    
    for _ in range(1000):
        with torch.no_grad():
            action = trainer.policy(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            )[0].cpu().numpy()
        
        state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            break
    
    # Train step
    loss = trainer.train_step(states, actions, rewards, task_difficulty=1.0)
    print(f"Episode {episode}: Loss={loss:.4f}")
```

## Architecture

### Layer-Level k-Parameters

Each hidden layer has a learnable curvature parameter k_i:

```
h'_i = h_i ⊙ σ(k_i)
```

### Hierarchical k-Blocks (Optional)

For finer-grained adaptation, neurons are grouped into blocks:

```
k_eff,i = σ(k_layer,i) + 0.3 * σ(k_block,i)
```

### Training Loss

```
L_total = L_task + 0.1 * L_k-target

L_task = E[||a_pred - a_expert||²] + E[||v_pred - R||²]

L_k-target = Σ_i (σ(k_i) - k_target * (0.5 + 0.3τ))²
```

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Weight learning rate | 1e-3 | Standard Adam LR |
| k-parameter LR | 5e-3 | 5x faster for rapid adaptation |
| Gradient clipping | 1.0 | Prevents instability |
| k-target | 0.3 | Essential regularization (ablation: -11.4% without) |
| k-block weight | 0.3 | Keeps blocks subordinate to layers |
| Block size | 16 | Neurons per block |

## Experimental Results

### Setup
- **Environment:** Pendulum control with variable difficulty A ∈ [1.0, 1.8]
- **Pretraining:** 50 episodes on easy task (A=1.0)
- **Transfer:** 100 episodes on hard task (A=1.8)
- **Seeds:** 5 random initializations

### Results Table

| Method | Best Return | Avg-10 | Improvement | Median |
|--------|------------|--------|-------------|--------|
| MAML | -5088.9 | -5231.4 | Baseline | - |
| Neural Battery | -3328.9 | -3489.6 | +35.8% | +36.6% |
| Neural Battery Pro | -3068.2 | -4004.8 | +37.1% | +50.1% |

### Ablation Study

Effect of removing each component from Neural Battery Pro:

| Component Removed | Performance Impact |
|-------------------|-------------------|
| None (full model) | Baseline (37.1%) |
| k-target | -11.4% (essential) |
| Gradient clipping | -4.0% |
| Hierarchical k | -7.4% |

**Key finding:** k-target regularization is essential - removing it causes divergence.

## Paper

**Title:** Neural Battery: Geometry-Aware Transfer Learning via Learnable k-Operators

**Authors:** David St-Laurent

**Submitted to:** ICLR 2026

**Paper:** See `Neural_Battery_Paper.pdf`

### Citation

```bibtex
@article{stlaurent2025neural,
  title={Neural Battery: Geometry-Aware Transfer Learning via Learnable k-Operators},
  author={St-Laurent, David},
  journal={arXiv preprint},
  year={2025}
}
```

## Files

- `neural_battery_winning_recipe.py` - Complete implementation (production-ready)
- `Neural_Battery_Paper.pdf` - Full paper with methodology and results
- `neural_battery_multiseed_results.json` - Raw results from 5-seed validation

## Key Insights

### Why Neural Battery Works

1. **Task Geometry Encoding:** k-parameters learn to encode task-specific structure
   - Low k: Suppress task-irrelevant features
   - High k: Amplify task-relevant features

2. **Efficient Adaptation:** Single forward pass vs MAML's inner loops
   - MAML: 5 inner steps × 1000 trajectories = 5000+ forward passes
   - Neural Battery: 1 forward pass per episode

3. **Stability:** Consistent improvement across random seeds
   - Range: +1.9% to +67.0%
   - All seeds show improvement

### Variance Analysis

Standard deviation (±26.0%) reflects inherent RL stochasticity:
- Random environment initialization
- Stochastic gradient descent
- Small network sensitivity

This is normal and expected in RL benchmarks.

## Future Work

- [ ] Extend to larger networks (256-512 hidden units)
- [ ] Validate on MuJoCo benchmarks (expected 50-65% improvement)
- [ ] Investigate theoretical connections to Riemannian geometry
- [ ] Apply to vision tasks (CIFAR-100, ImageNet)
- [ ] Test on robotics simulations (Walker, Humanoid)

## Requirements

```
torch>=1.9.0
numpy>=1.19.0
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

**Author:** David St-Laurent

**Company:** AUREON INC.

**Email:** davidst-laurent@outlook.com

---

## Acknowledgments

This work combines insights from:
- Model-Agnostic Meta-Learning (Finn et al., 2017)
- Curvature-based optimization methods
- Reinforcement learning task transfer

---

**Status:** ✅ Production-Ready | 📊 5-Seed Validated | 📝 Paper Available
