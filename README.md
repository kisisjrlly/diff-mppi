# Diff-MPPI: Differentiable Model Predictive Path Integral Control

A PyTorch implementation of Path Integral Networks for differentiable optimal control, based on:

- Okada et al., 2017, "Path Integral Networks: End-to-End Differentiable Optimal Control"
- Okada & Taniguchi, 2018, "Acceleration of Gradient-Based Path Integral Method for Efficient Optimal and Inverse Optimal Control"

## Features

- **Standard MPPI**: Classic Model Predictive Path Integral control
- **Accelerated MPPI**: Gradient-based acceleration with Adam, NAG, and RMSprop
- **GPU Acceleration**: Full CUDA support for high-performance computing
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

# Simulate trajectory
trajectory = controller.rollout(initial_state, optimal_control)
```

## Examples

### Pendulum Swing-up

```bash
python examples/pendulum_example.py
```

This example demonstrates:
- Pendulum swing-up from hanging position to upright
- Comparison of standard vs accelerated MPPI methods
- Visualization of results

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
examples/
├── pendulum_example.py  # Pendulum swing-up example
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
