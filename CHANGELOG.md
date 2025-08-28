# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-28

### Added
- Initial release of diff-mppi library
- Core `DiffMPPI` class implementing Path Integral Networks
- Support for multiple acceleration methods:
  - Standard MPPI (no acceleration)
  - Adam optimizer
  - Nesterov Accelerated Gradient (NAG)  
  - RMSprop optimizer
- GPU acceleration with CUDA support
- Helper function `create_mppi_controller` for easy setup
- Comprehensive documentation suite:
  - Installation guide
  - API reference
  - Theoretical background
  - Examples and tutorials
  - Contributing guidelines
- Example implementations:
  - Pendulum swing-up
  - Vehicle navigation with obstacle avoidance
  - Imitation learning
  - Real-time control
- Pip installable package with proper setup.py and pyproject.toml
- Type annotations throughout the codebase
- Error handling and input validation

### Core Features
- Batched trajectory sampling for efficiency
- Temperature-controlled exploration
- Control bounds support
- Device-agnostic implementation (CPU/GPU)
- Memory-efficient rollout implementation
- Numerical stability optimizations

### Performance
- Vectorized dynamics and cost evaluation
- Efficient importance sampling with softmax
- GPU memory optimization
- Support for large-scale problems (tested up to 1000 samples, 100 horizon)

### Documentation
- Mathematical foundations and derivations
- Implementation details and computational complexity
- Performance benchmarks and guidelines
- Troubleshooting guides
- Code examples across different applications

### References
Implementation based on:
- Okada et al., 2017: "Path Integral Networks: End-to-End Differentiable Optimal Control"
- Okada & Taniguchi, 2018: "Acceleration of Gradient-Based Path Integral Method for Efficient Optimal and Inverse Optimal Control"

## [Unreleased]

### Planned for v1.1.0
- [ ] Unit test suite with pytest
- [ ] Continuous integration with GitHub Actions
- [ ] Performance benchmarking suite
- [ ] Additional examples (robotic arm, quadcopter)
- [ ] Constraint handling improvements
- [ ] Documentation website with Sphinx

### Planned for v1.5.0
- [ ] Stochastic dynamics support
- [ ] Multi-objective optimization
- [ ] Advanced constraint handling
- [ ] Integration with popular RL libraries
- [ ] Distributed computing support

### Planned for v2.0.0
- [ ] Real-time control interfaces
- [ ] Web-based visualization tools
- [ ] Advanced acceleration methods
- [ ] Neural network dynamics/cost learning
- [ ] Automatic differentiation through dynamics

---

## Development Notes

### Version 1.0.0 Development Process

This version represents a complete rewrite and refactoring of the original codebase:

1. **Architecture Redesign**: Consolidated complex multi-file structure into a single, clean core module
2. **API Simplification**: Removed application-specific functions from the core library interface
3. **Documentation**: Created comprehensive documentation suite covering theory, implementation, and usage
4. **Examples Separation**: Moved application examples to separate directory with clean implementations
5. **Packaging**: Proper Python packaging with pip installation support
6. **Performance**: GPU optimization and memory efficiency improvements

### Breaking Changes from Pre-1.0

- Removed `create_pendulum_mppi`, `create_navigation_mppi` functions from core interface
- Consolidated `diff_mppi.solvers`, `diff_mppi.models`, `diff_mppi.utils` into single `diff_mppi.core` module
- Changed import structure: now `from diff_mppi import DiffMPPI, create_mppi_controller`
- Simplified constructor parameters for `DiffMPPI` class
- Removed CLI interface and configuration files

### Migration Guide from Pre-1.0

```python
# Old (pre-1.0)
from diff_mppi.interface import create_pendulum_mppi
controller = create_pendulum_mppi()

# New (1.0+)
import diff_mppi

def pendulum_dynamics(state, control):
    # Your dynamics implementation
    pass

def pendulum_cost(state, control):
    # Your cost implementation  
    pass

controller = diff_mppi.create_mppi_controller(
    state_dim=3, control_dim=1,
    dynamics_fn=pendulum_dynamics,
    cost_fn=pendulum_cost
)
```

This change makes the library more generic and reusable while maintaining all core functionality.
