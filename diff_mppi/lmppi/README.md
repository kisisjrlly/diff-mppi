# Latent Space Model Predictive Path Integral (LMPPI) Control

## Overview

This module implements Latent Space Model Predictive Path Integral (LMPPI) control, a novel approach that dramatically reduces the computational complexity of MPPI by performing optimization in a learned low-dimensional latent space rather than the high-dimensional control sequence space.

## Key Benefits

- **Computational Efficiency**: Reduces sampling dimension from `H × M` to `d` where `d << H × M`
- **Trajectory Feasibility**: VAE decoder ensures generated trajectories are dynamically feasible
- **Sample Efficiency**: Better exploration through learned trajectory manifold
- **Scalability**: Performance scales better with horizon length and control dimensions

## Architecture

### Core Components

1. **TrajectoryVAE**: Variational autoencoder for learning trajectory representations
2. **LMPPIController**: Main controller using latent space optimization
3. **LMPPITrainer**: Training utilities for VAE models
4. **TrajectoryDataset**: Data handling for trajectory collections

### Workflow

#### Offline Training Phase
1. Collect diverse, feasible trajectory data
2. Train VAE to learn compact latent representation
3. Validate reconstruction quality

#### Online Control Phase
1. Encode reference trajectory to latent space
2. Sample noise in low-dimensional latent space
3. Decode samples to full trajectories (replaces forward integration)
4. Evaluate trajectory costs
5. Update latent representation using MPPI weights

## Quick Start

```python
import torch
from diff_mppi.lmppi import *

# 1. Create trajectory dataset
trajectories = create_synthetic_trajectories(
    num_trajectories=1000,
    horizon=20,
    state_dim=4,
    control_dim=2
)

# 2. Train VAE model
dataset = TrajectoryDataset(trajectories, state_dim=4, control_dim=2)
train_data, val_data = dataset.split()

vae_model = TrajectoryVAE(
    input_dim=20 * 6,  # horizon * (state_dim + control_dim)
    latent_dim=8,
    architecture="mlp"
)

trainer = LMPPITrainer(vae_model, train_data, val_data)
trainer.train(num_epochs=50)

# 3. Use for control
controller = LMPPIController(
    vae_model=vae_model,
    state_dim=4,
    control_dim=2,
    cost_fn=your_cost_function,
    horizon=20
)

# Online control
initial_state = torch.randn(1, 4)
control_action = controller.step(initial_state)
```

## Detailed API Reference

### TrajectoryVAE

The core neural network model for learning latent trajectory representations.

```python
TrajectoryVAE(
    input_dim: int,           # Total trajectory dimension
    latent_dim: int,          # Latent space dimension (typically 8-16)
    hidden_dims: List[int],   # Hidden layer dimensions
    architecture: str,        # "mlp", "lstm", or "cnn"
    beta: float = 1.0        # KL divergence weight (β-VAE)
)
```

**Supported Architectures:**
- **MLP**: Simple feedforward, good for general use
- **LSTM**: Preserves temporal structure, good for dynamic sequences
- **CNN**: Captures local patterns, good for spatial trajectories

### LMPPIController

Main controller implementing latent space MPPI optimization.

```python
LMPPIController(
    vae_model: TrajectoryVAE,     # Pre-trained VAE model
    state_dim: int,               # State space dimension
    control_dim: int,             # Control space dimension
    cost_fn: Callable,            # Cost function g(state, control)
    horizon: int = 20,            # Planning horizon
    num_samples: int = 100,       # Number of trajectory samples
    temperature: float = 1.0,     # MPPI temperature parameter
    latent_noise_scale: float = 1.0  # Latent space noise scaling
)
```

**Key Methods:**
- `solve(initial_state)`: Solve optimization problem
- `step(state)`: MPC-style single control action
- `set_reference_trajectory(traj)`: Set warm-start trajectory
- `rollout(initial_state)`: Generate full trajectory

### LMPPITrainer

Comprehensive training utilities for VAE models.

```python
LMPPITrainer(
    model: TrajectoryVAE,
    train_dataset: TrajectoryDataset,
    val_dataset: TrajectoryDataset,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu"
)
```

**Features:**
- Automatic checkpointing and early stopping
- Validation monitoring and learning rate scheduling
- Reconstruction quality evaluation
- Training curve visualization

## Example Applications

### 1. Quadrotor Navigation

```python
# Define quadrotor dynamics and cost
def quadrotor_dynamics(state, control):
    # Implement quadrotor dynamics
    pass

def navigation_cost(state, control):
    # Waypoint following + obstacle avoidance
    pass

# Train VAE on demonstration data
demo_data = load_quadrotor_demonstrations()
vae_model = train_navigation_vae(demo_data)

# Online control
controller = LMPPIController(vae_model, ...)
```

### 2. Robotic Manipulation

```python
# 7-DOF arm trajectory optimization
def arm_dynamics(joint_state, joint_torques):
    # Robot arm dynamics
    pass

def manipulation_cost(state, control):
    # End-effector goal + collision avoidance
    pass

# Use LSTM architecture for temporal dependencies
vae_model = TrajectoryVAE(
    architecture="lstm",
    latent_dim=12  # Higher dimensional for complex motions
)
```

## Performance Guidelines

### Hyperparameter Selection

**Latent Dimension (`latent_dim`)**:
- Start with 8-16 for most problems
- Increase for more complex trajectory diversity
- Monitor reconstruction quality vs. compression

**Number of Samples (`num_samples`)**:
- LMPPI typically needs fewer samples than standard MPPI
- Start with 50-100, adjust based on performance

**Architecture Choice**:
- **MLP**: Default choice, works well for most cases
- **LSTM**: Use when temporal dependencies are important
- **CNN**: Use for spatially-structured trajectories

**Training Data**:
- Collect diverse, feasible trajectories
- Quality matters more than optimality
- Include various scenarios and initial conditions

### Computational Complexity

| Method | Sampling Dim | Forward Calls | Memory |
|--------|-------------|---------------|---------|
| Standard MPPI | H × M | K × H | O(K × H × M) |
| LMPPI | d | 0 | O(K × d) |

Where:
- H: horizon length
- M: control dimension  
- K: number of samples
- d: latent dimension (d << H × M)

## Troubleshooting

### Common Issues

**Poor Reconstruction Quality**:
- Increase model capacity (hidden dimensions)
- Reduce β parameter for more capacity
- Improve training data diversity

**Slow Convergence**:
- Adjust latent noise scaling
- Increase number of samples
- Check cost function scaling

**Control Instability**:
- Add control bounds
- Reduce latent noise initially
- Validate dynamics model accuracy

### Debugging Tools

```python
# Evaluate reconstruction quality
metrics = trainer.evaluate_reconstruction()
print(f"Reconstruction MSE: {metrics['reconstruction_mse']}")

# Visualize latent space
plot_latent_space(vae_model, dataset)

# Compare with standard MPPI
compare_controllers(lmppi_controller, standard_mppi)
```

## Advanced Features

### Custom Architectures

Extend the base VAE for specific applications:

```python
class CustomTrajectoryVAE(TrajectoryVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom layers
        self.attention = nn.MultiheadAttention(...)
    
    def encode(self, x):
        # Custom encoding logic
        pass
```

### Multi-Task Learning

Train single VAE for multiple trajectory types:

```python
# Combined dataset
mixed_dataset = ConcatDataset([
    hover_trajectories,
    aggressive_trajectories,
    precision_trajectories
])

# Task-conditioned VAE
vae_model = ConditionalTrajectoryVAE(
    latent_dim=16,
    num_tasks=3
)
```

### Online Adaptation

Adapt VAE during deployment:

```python
# Continual learning setup
controller.enable_online_adaptation(
    adaptation_rate=0.01,
    memory_size=1000
)

# Update with new experiences
controller.update_model(new_trajectory_data)
```

## References

1. "Path Integral Networks: End-to-End Differentiable Optimal Control" (Okada et al., 2017)
2. "Acceleration of Gradient-Based Path Integral Method" (Okada & Taniguchi, 2018)
3. "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{lmppi2025,
  title={Latent Space Model Predictive Path Integral Control},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/diff-mppi}
}
```
