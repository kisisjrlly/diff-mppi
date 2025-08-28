# Diff-MPPI Project Overview

## Introduction

Diff-MPPI is a PyTorch-based implementation of differentiable Model Predictive Path Integral (MPPI) control, implementing the algorithms from two key papers:

1. **Okada et al., 2017**: "Path Integral Networks: End-to-End Differentiable Optimal Control"
2. **Okada & Taniguchi, 2018**: "Acceleration of Gradient-Based Path Integral Method for Efficient Optimal and Inverse Optimal Control"

## Project Goals

- **Research Implementation**: Faithful implementation of PI-Net algorithms for academic research
- **Practical Usage**: Easy-to-use library for real-world control applications
- **Educational Tool**: Clear, documented code for learning optimal control concepts
- **Extensibility**: Modular design allowing easy extension to new applications

## Key Features

### Core Capabilities
- **Standard MPPI**: Classical path integral control implementation
- **Accelerated MPPI**: Gradient-based acceleration using Adam, NAG, and RMSprop optimizers
- **GPU Acceleration**: Full CUDA support for high-performance computing
- **Differentiable**: End-to-end differentiable implementation for learning applications

### Software Engineering
- **Clean API**: Simple, intuitive interface for creating and using controllers
- **Pip Installable**: Standard Python packaging for easy installation
- **Modular Design**: Clear separation between core algorithms and application examples
- **Type Annotations**: Full type hints for better development experience

## Project Structure

```
diff-mppi/
├── diff_mppi/              # Core library
│   ├── __init__.py         # Public API exports
│   └── core.py             # Main MPPI implementation
├── examples/               # Application examples
│   ├── pendulum_example_clean.py
│   ├── navigation_example.py
│   └── imitation_learning_example.py
├── docs/                   # Documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── API_REFERENCE.md
│   ├── THEORY.md
│   └── EXAMPLES.md
├── tests/                  # Unit tests (future)
├── setup.py               # Package setup
├── pyproject.toml         # Modern packaging config
└── README.md              # Quick start guide
```

## Design Principles

### 1. Simplicity
- Single core module with unified implementation
- Minimal external dependencies
- Clear, readable code structure

### 2. Performance
- GPU acceleration by default
- Efficient tensor operations
- Vectorized computations

### 3. Flexibility
- Generic dynamics and cost function interfaces
- Configurable acceleration methods
- Extensible to new control problems

### 4. Reliability
- Type safety with annotations
- Comprehensive error handling
- Tested on standard benchmarks

## Target Applications

### Research Applications
- **Optimal Control**: Standard trajectory optimization problems
- **Imitation Learning**: Learning from expert demonstrations
- **Inverse Optimal Control**: Inferring cost functions from behavior
- **Reinforcement Learning**: As a planning component in model-based RL

### Practical Applications
- **Robotics**: Robot arm control, mobile robot navigation
- **Autonomous Vehicles**: Path planning and tracking
- **Process Control**: Industrial control systems
- **Game AI**: Strategic planning and decision making

## Development Workflow

### Installation
```bash
git clone https://github.com/kisisjrlly/diff-mppi.git
cd diff-mppi
pip install -e .
```

### Basic Usage
```python
import diff_mppi

# Define system
controller = diff_mppi.create_mppi_controller(
    state_dim=3, control_dim=1,
    dynamics_fn=my_dynamics,
    cost_fn=my_cost
)

# Solve
solution = controller.solve(initial_state)
```

### Testing
```bash
python examples/pendulum_example_clean.py
```

## Performance Characteristics

### Computational Complexity
- **Time**: O(H × K × (state_dim + control_dim)) per iteration
  - H: horizon length
  - K: number of samples
- **Space**: O(H × K × state_dim) for trajectory storage

### Convergence Properties
- **Standard MPPI**: Linear convergence to local optimum
- **Accelerated MPPI**: Faster convergence with momentum methods
- **Typical iterations**: 5-20 for convergence depending on problem complexity

### Hardware Requirements
- **Minimum**: CPU with 4GB RAM
- **Recommended**: NVIDIA GPU with 8GB+ VRAM for large-scale problems
- **Scalability**: Scales well with parallel trajectory sampling

## Extension Points

### Adding New Acceleration Methods
```python
class CustomAccelerator:
    def step(self, control_seq, gradient):
        # Custom optimization logic
        return updated_control_seq
```

### Custom Dynamics Models
```python
def neural_dynamics(state, control):
    # Neural network dynamics
    return next_state
```

### Custom Cost Functions
```python
def learned_cost(state, control):
    # Learned cost function
    return cost_tensor
```

## Future Roadmap

### Short Term (v1.1)
- [ ] Unit test suite
- [ ] Continuous integration
- [ ] Performance benchmarks
- [ ] Additional examples

### Medium Term (v1.5)
- [ ] Stochastic dynamics support
- [ ] Constrained optimization
- [ ] Multi-objective optimization
- [ ] Integration with RL libraries

### Long Term (v2.0)
- [ ] Distributed computing support
- [ ] Advanced acceleration methods
- [ ] Real-time control interfaces
- [ ] Web-based visualization

## Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Code Style**: Follow PEP 8 and use type hints
2. **Testing**: Add tests for new features
3. **Documentation**: Update docs for API changes
4. **Examples**: Provide examples for new capabilities

## License

MIT License - see LICENSE file for full terms.

## Acknowledgments

- Original paper authors: Masashi Okada and Tadahiro Taniguchi
- PyTorch team for the excellent deep learning framework
- Control theory community for foundational work on path integral methods
