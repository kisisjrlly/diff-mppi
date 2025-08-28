# Documentation Index

Welcome to the diff-mppi documentation! This directory contains comprehensive documentation for the Differentiable Model Predictive Path Integral (Diff-MPPI) library.

## Quick Navigation

### 📚 Getting Started
- **[Installation Guide](INSTALLATION.md)** - System requirements, installation methods, and troubleshooting
- **[API Reference](API_REFERENCE.md)** - Complete API documentation with examples
- **[Examples and Tutorials](EXAMPLES.md)** - Practical examples and step-by-step tutorials

### 🔬 Technical Documentation  
- **[Theoretical Background](THEORY.md)** - Mathematical foundations and implementation details
- **[Project Overview](PROJECT_OVERVIEW.md)** - Architecture, design principles, and roadmap

### 🤝 Community
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute code, documentation, and examples

## Document Overview

### [Installation Guide](INSTALLATION.md)
Comprehensive setup instructions covering:
- System requirements (hardware/software)
- Installation methods (pip, conda, docker)
- Environment setup and GPU configuration
- Verification and troubleshooting
- Platform-specific instructions

### [API Reference](API_REFERENCE.md)
Complete API documentation including:
- `DiffMPPI` class with all methods and parameters
- Helper functions and utilities
- Dynamics and cost function interfaces
- Acceleration methods (Adam, NAG, RMSprop)
- Error handling and performance guidelines

### [Examples and Tutorials](EXAMPLES.md)
Practical examples covering:
- Pendulum swing-up (classic benchmark)
- Vehicle navigation with obstacle avoidance
- Imitation learning with neural networks
- Real-time control implementation
- Performance benchmarks and troubleshooting

### [Theoretical Background](THEORY.md)
Deep dive into mathematical foundations:
- Path integral control theory
- Hamilton-Jacobi-Bellman equations
- PI-Net algorithm derivation
- Acceleration methods analysis
- Convergence properties and computational complexity

### [Project Overview](PROJECT_OVERVIEW.md)
High-level project information:
- Goals and design principles
- Architecture and code organization
- Target applications and use cases
- Development roadmap and future plans

### [Contributing Guide](CONTRIBUTING.md)
Guidelines for contributors:
- Development environment setup
- Code style and testing requirements
- Documentation standards
- Pull request process and community guidelines

## Quick Start

If you're new to diff-mppi, follow this learning path:

1. **Install the library**: Start with [Installation Guide](INSTALLATION.md)
2. **Run first example**: Follow the pendulum example in [Examples](EXAMPLES.md)
3. **Understand the API**: Read [API Reference](API_REFERENCE.md) 
4. **Learn the theory**: Explore [Theoretical Background](THEORY.md) for deeper understanding
5. **Contribute**: Check [Contributing Guide](CONTRIBUTING.md) to get involved

## Code Examples

### Basic Usage
```python
import diff_mppi

# Define your system
def dynamics(state, control):
    return state + 0.1 * control

def cost(state, control):
    return torch.sum(state**2) + 0.1 * torch.sum(control**2)

# Create controller
controller = diff_mppi.create_mppi_controller(
    state_dim=3, control_dim=1,
    dynamics_fn=dynamics, cost_fn=cost
)

# Solve
initial_state = torch.tensor([1.0, 0.0, 0.0])
optimal_control = controller.solve(initial_state)
```

### Advanced Usage with Acceleration
```python
# Accelerated MPPI with Adam optimizer
controller = diff_mppi.create_mppi_controller(
    state_dim=3, control_dim=1,
    dynamics_fn=dynamics, cost_fn=cost,
    acceleration="adam", lr=0.1,
    device="cuda"
)
```

## Key Features Highlighted in Documentation

### 🚀 Performance
- **GPU Acceleration**: Full CUDA support for high-performance computing
- **Vectorized Operations**: Efficient batch processing of trajectories
- **Multiple Optimizers**: Adam, NAG, RMSprop acceleration methods

### 🔧 Flexibility
- **Generic Interface**: Works with any differentiable dynamics and cost functions
- **Configurable**: Extensive hyperparameter tuning options
- **Extensible**: Easy to add new acceleration methods and features

### 📖 Educational
- **Theory Explained**: Mathematical foundations clearly documented
- **Examples Included**: Practical examples across different domains
- **Research Oriented**: Faithful implementation of published algorithms

## Documentation Standards

Our documentation follows these principles:

- **Clarity**: Clear, concise explanations with examples
- **Completeness**: Comprehensive coverage of all features
- **Accuracy**: Up-to-date information that matches the code
- **Accessibility**: Suitable for beginners and experts
- **Maintainability**: Well-organized and easy to update

## References and Citations

The library implements algorithms from these key papers:

1. **Okada et al., 2017**: "Path Integral Networks: End-to-End Differentiable Optimal Control"
2. **Okada & Taniguchi, 2018**: "Acceleration of Gradient-Based Path Integral Method for Efficient Optimal and Inverse Optimal Control"

For academic use, please cite the original papers and this implementation.

## Feedback and Improvements

We continuously improve our documentation based on user feedback:

- **Found an error?** Please open an issue on GitHub
- **Missing information?** Suggest improvements via pull requests
- **Have questions?** Join our community discussions

## Version Information

This documentation is for diff-mppi version 1.0.0. For version-specific information, check the changelog and release notes.

---

*Last updated: August 28, 2025*
