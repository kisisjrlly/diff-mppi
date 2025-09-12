# Diff-MPPI: Differentiable Model Predictive Path Integral Control

A PyTorch implementation of Path Integral Networks for differentiable optimal control, now featuring both standard MPPI and Latent-space MPPI (LMPPI), based on:

- Okada et al., 2017, "Path Integral Networks: End-to-End Differentiable Optimal Control"
- Okada & Taniguchi, 2018, "Acceleration of Gradient-Based Path Integral Method for Efficient Optimal and Inverse Optimal Control"

## Features

- **Standard MPPI**: Classic Model Predictive Path Integral control
- **Latent-space MPPI (LMPPI)**: **NEW!** VAE-based latent space control for high-dimensional trajectory optimization
- **Accelerated MPPI**: Gradient-based acceleration with Adam, NAG, and RMSprop
- **Batch Processing**: Efficient parallel processing of multiple initial states (3-4x speedup)
- **GPU Acceleration**: Full CUDA support for high-performance computing
- **End-to-End Learning**: Differentiable implementation for neural dynamics and cost learning
- **VAE Trajectory Modeling**: Learn compact latent representations of complex trajectory patterns
- **Clean Interface**: Simple, reusable API for different control problems
- **Pip Installable**: Easy installation and integration

## Installation

```bash
# Clone the repository
git clone https://github.com/kisisjrlly/diff-mppi.git
cd diff-mppi

# Install in development mode
pip install -e .
```

## Quick Start

### Standard MPPI

```python
import torch
import diff_mppi

# Define your system dynamics
def dynamics(state, control):
    # Your dynamics model here
    return next_state

# Define your cost function
def cost(state, control):
    # Your cost function here
    return cost_value

# Create controller
controller = diff_mppi.create_mppi_controller(
    state_dim=3,
    control_dim=1,
    dynamics_fn=dynamics,
    cost_fn=cost,
    horizon=30,
    num_samples=100,
    temperature=0.1,
    acceleration="adam",  # Optional: "adam", "nag", "rmsprop"
    device="cuda"
)

# Solve for optimal control
initial_state = torch.tensor([1.0, 0.0, 0.0])
optimal_control = controller.solve(initial_state, num_iterations=10)

# Batch processing for multiple initial states (NEW!)
initial_states = torch.tensor([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
batch_controls = controller.solve(initial_states)  # 3x speedup on GPU!

# Simulate trajectory
trajectory = controller.rollout(initial_state, optimal_control)
```

### Latent-space MPPI (LMPPI) **NEW!**

```python
import torch
from diff_mppi.lmppi import TrajectoryVAE, LMPPIController, LMPPITrainer
from diff_mppi.lmppi import create_synthetic_trajectories, VAEConfig, ControllerConfig

# Step 1: Generate or load trajectory data
trajectories = create_synthetic_trajectories(
    num_trajectories=1000,
    horizon=30,
    state_dim=4,
    control_dim=2
)

# Step 2: Train VAE for trajectory representation learning
vae_config = VAEConfig(
    input_dim=4,      # state dimension
    latent_dim=8,     # compressed latent dimension
    hidden_dims=[64, 32],
    architecture="mlp"
)

vae = TrajectoryVAE(vae_config)
trainer = LMPPITrainer(vae, learning_rate=1e-3)

# Train the VAE
trainer.train(trajectories, epochs=100, batch_size=32)

# Step 3: Create LMPPI controller
controller_config = ControllerConfig(
    num_samples=100,
    temperature=0.1,
    horizon=30,
    lambda_=1.0
)

def dynamics(state, control):
    # Your dynamics model
    return next_state

def cost_fn(state, control):
    # Your cost function
    return cost

controller = LMPPIController(vae, dynamics, cost_fn, controller_config)

# Step 4: Solve in latent space
initial_state = torch.tensor([1.0, 0.0, 0.5, -0.5])
optimal_control = controller.solve(initial_state, num_iterations=20)
```

## Examples

Run the included examples:

```bash
# Basic pendulum swing-up with acceleration method comparison
python examples/pendulum_example.py

# Batch processing demonstration
python examples/batch_processing_example.py

# Neural dynamics learning and control
python examples/neural_dynamics_example.py

# Complete imitation learning with end-to-end training
python examples/imitation_learning_example.py

# Simple LMPPI demonstration (NEW!)
python examples/simple_lmppi_demo.py
```

These examples demonstrate different aspects of the library:
- **Pendulum example**: Optimization with known models
- **Batch processing**: Efficient parallel processing of multiple states
- **Neural dynamics**: Learning dynamics models
- **Imitation learning**: Complete end-to-end differentiable learning
- **LMPPI demo**: Latent space trajectory optimization with VAE

## Documentation

📚 **[Complete Documentation](docs/README.md)** - Comprehensive documentation suite including:

- **[Installation Guide](docs/INSTALLATION.md)** - System requirements, setup, and troubleshooting
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation with examples
- **[Examples & Tutorials](docs/EXAMPLES.md)** - Practical examples and step-by-step tutorials
- **[Theoretical Background](docs/THEORY.md)** - Mathematical foundations and implementation details
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute to the project

## API Overview

### `DiffMPPI` Class

The main controller class implementing Path Integral Networks.

**Key Methods:**
- `solve(initial_state, num_iterations)`: Solve for optimal control sequence
- `rollout(initial_state, control_sequence)`: Simulate system trajectory
- `step(state)`: Get single control action for real-time control

### `LMPPI` Module **NEW!**

Latent-space MPPI for high-dimensional trajectory optimization.

**Key Classes:**
- `TrajectoryVAE`: Variational autoencoder for trajectory representation learning
- `LMPPIController`: Controller operating in latent space
- `LMPPITrainer`: Training utilities for VAE models
- `TrajectoryDataset`: Data handling and preprocessing utilities

**Key Features:**
- Support for MLP, LSTM, and CNN encoder/decoder architectures
- Comprehensive training with validation and checkpointing
- Configurable latent space dimensions and model complexity
- Built-in visualization and evaluation tools

### Helper Functions

```python
# Create controller with default settings
controller = diff_mppi.create_mppi_controller(
    state_dim, control_dim, dynamics_fn, cost_fn, **kwargs
)
```

## Acceleration Methods

The library supports several gradient-based acceleration methods:

1. **Standard MPPI** (`acceleration=None`): Classic path integral method
2. **Adam** (`acceleration="adam"`): Adaptive moment estimation
3. **NAG** (`acceleration="nag"`): Nesterov accelerated gradient
4. **RMSprop** (`acceleration="rmsprop"`): Root mean square propagation

## Performance

| Problem | Horizon | Samples | Device | Time/Iteration |
|---------|---------|---------|--------|---------------|
| Pendulum | 30 | 100 | CPU | 25ms |
| Pendulum | 30 | 100 | GPU | 8ms |
| Navigation | 40 | 200 | CPU | 85ms |
| Navigation | 40 | 200 | GPU | 22ms |

## Requirements

- Python >= 3.8
- PyTorch >= 1.9
- NumPy >= 1.21
- Matplotlib >= 3.3 (for examples)

## Architecture

Clean separation between core functionality and examples:

```
diff_mppi/
├── __init__.py          # Public API
├── core.py              # Core MPPI implementation
├── lmppi/               # NEW! Latent-space MPPI module
│   ├── __init__.py      # LMPPI API exports
│   ├── models.py        # VAE architectures (MLP/LSTM/CNN)
│   ├── controller.py    # Latent space controller
│   ├── trainer.py       # VAE training utilities
│   ├── data.py          # Trajectory data handling
│   ├── config.py        # Configuration management
│   ├── utils.py         # Visualization and evaluation
│   └── README.md        # LMPPI documentation
examples/
├── pendulum_example.py  # Pendulum swing-up example
├── simple_lmppi_demo.py # NEW! LMPPI demonstration
├── test_lmppi.py        # NEW! LMPPI test suite
docs/
├── README.md            # Documentation index
├── API_REFERENCE.md     # Complete API docs
├── THEORY.md           # Mathematical background
└── ...                 # Additional documentation
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for:

- Development environment setup
- Code style guidelines
- Testing requirements
- Pull request process

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes and version history.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite the original papers:

```bibtex
@article{okada2017path,
  title={Path integral networks: End-to-end differentiable optimal control},
  author={Okada, Masashi and Taniguchi, Tadahiro},
  journal={arXiv preprint arXiv:1706.09597},
  year={2017}
}

@article{okada2018acceleration,
  title={Acceleration of gradient-based path integral method for efficient optimal and inverse optimal control},
  author={Okada, Masashi and Taniguchi, Tadahiro},
  journal={arXiv preprint arXiv:1805.11897},
  year={2018}
}
```

## Support

- **Documentation**: Start with [docs/README.md](docs/README.md)
- **Examples**: Check the [examples/](examples/) directory
- **Issues**: Report bugs on [GitHub Issues](https://github.com/kisisjrlly/diff-mppi/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/kisisjrlly/diff-mppi/discussions)
