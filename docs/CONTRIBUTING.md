# Contributing to Diff-MPPI

We welcome contributions to the diff-mppi project! This document provides guidelines for contributing code, documentation, examples, and bug reports.

## Getting Started

### Development Environment Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/diff-mppi.git
   cd diff-mppi
   ```

3. **Set up development environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode with dev dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

4. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/kisisjrlly/diff-mppi.git
   ```

### Development Dependencies

The development setup includes additional tools:

```bash
pip install -e ".[dev]"
```

This installs:
- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `pre-commit` - Git hooks
- `sphinx` - Documentation generation
- `jupyter` - Notebook development

## Contribution Types

### 1. Bug Reports

**Before submitting:**
- Check existing issues to avoid duplicates
- Test with the latest version
- Gather system information using our diagnostic script

**Bug report should include:**
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (Python version, OS, hardware)
- Minimal code example
- Error messages/stack traces

**Template:**
```markdown
## Bug Description
Brief description of the bug.

## To Reproduce
1. Step 1
2. Step 2
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- PyTorch: [e.g., 1.12.1]
- CUDA: [e.g., 11.6]
- diff-mppi: [e.g., 1.0.0]

## Code Example
```python
# Minimal example that reproduces the issue
```

## Additional Context
Any other relevant information.
```

### 2. Feature Requests

**Before submitting:**
- Check if feature already exists
- Consider if it fits the project scope
- Think about API design and backwards compatibility

**Feature request should include:**
- Clear description of the feature
- Use case and motivation
- Proposed API (if applicable)
- Implementation ideas (optional)

### 3. Code Contributions

#### Code Style

We follow Python PEP 8 with some modifications:

**Formatting:**
- Use `black` for automatic formatting
- Line length: 88 characters
- Use double quotes for strings

**Type Hints:**
- All public functions must have type hints
- Use `typing` module for complex types
- Optional parameters should use `Optional[T]`

**Documentation:**
- All public functions need docstrings
- Use Google style docstrings
- Include examples for complex functions

**Example:**
```python
def create_mppi_controller(
    state_dim: int,
    control_dim: int,
    dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    horizon: int = 30,
    num_samples: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
) -> DiffMPPI:
    """Create MPPI controller with default settings.
    
    Args:
        state_dim: Dimension of state space.
        control_dim: Dimension of control space.
        dynamics_fn: Function mapping (state, control) to next_state.
        cost_fn: Function mapping (state, control) to cost.
        horizon: Planning horizon length.
        num_samples: Number of trajectory samples.
        temperature: Temperature parameter for path integral.
        device: PyTorch device ("cpu" or "cuda").
        
    Returns:
        Configured DiffMPPI controller.
        
    Example:
        >>> def dynamics(state, control):
        ...     return state + 0.1 * control
        >>> def cost(state, control):
        ...     return torch.sum(state**2)
        >>> controller = create_mppi_controller(
        ...     state_dim=3, control_dim=1,
        ...     dynamics_fn=dynamics, cost_fn=cost
        ... )
    """
    return DiffMPPI(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics_fn=dynamics_fn,
        cost_fn=cost_fn,
        horizon=horizon,
        num_samples=num_samples,
        temperature=temperature,
        device=device,
    )
```

#### Testing

**Test Coverage:**
- All new code must have tests
- Aim for >90% test coverage
- Test both happy path and edge cases

**Test Structure:**
```python
import pytest
import torch
import diff_mppi

class TestDiffMPPI:
    """Test suite for DiffMPPI class."""
    
    def test_initialization(self):
        """Test controller initialization."""
        # Test code here
        pass
    
    def test_solve_basic(self):
        """Test basic solving functionality."""
        # Test code here
        pass
    
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_device_compatibility(self, device):
        """Test compatibility across devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # Test code here
        pass
```

**Running Tests:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=diff_mppi

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v
```

#### Code Quality Checks

Before submitting, run:

```bash
# Format code
black diff_mppi/ tests/ examples/

# Check linting
flake8 diff_mppi/ tests/ examples/

# Type checking
mypy diff_mppi/

# Run tests
pytest
```

Pre-commit hooks will automatically run these checks.

### 4. Documentation

#### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **User Guide**: High-level tutorials and examples
3. **Developer Documentation**: Implementation details
4. **Examples**: Practical use cases

#### Documentation Style

**Docstrings:**
- Use Google style
- Include Args, Returns, Raises sections
- Provide examples for complex functions

**Markdown Files:**
- Use clear headings and structure
- Include code examples
- Add cross-references where helpful

**Example Documentation:**
```python
def solve(
    self,
    initial_state: torch.Tensor,
    num_iterations: int = 10
) -> torch.Tensor:
    """Solve for optimal control sequence.
    
    Iteratively optimizes control sequence using MPPI algorithm.
    Each iteration samples trajectories, evaluates costs, and updates
    control sequence based on importance-weighted averages.
    
    Args:
        initial_state: Starting state [state_dim].
        num_iterations: Number of optimization iterations.
            More iterations generally improve solution quality
            but increase computation time.
            
    Returns:
        Optimal control sequence [horizon, control_dim].
        
    Raises:
        ValueError: If initial_state has wrong dimensions.
        RuntimeError: If solve fails due to numerical issues.
        
    Example:
        >>> controller = DiffMPPI(...)
        >>> x0 = torch.tensor([1.0, 0.0, 0.0])
        >>> optimal_control = controller.solve(x0, num_iterations=20)
        >>> trajectory = controller.rollout(x0, optimal_control)
    """
```

### 5. Examples

#### Creating New Examples

Examples should:
- Demonstrate specific features or use cases
- Be well-commented and educational
- Include visualization when appropriate
- Be self-contained and runnable

**Example structure:**
```python
"""
Example: [Brief Description]

This example demonstrates [specific feature/concept].
It shows how to [key learning objectives].

References:
- [Relevant papers or documentation]
"""

import torch
import diff_mppi
import matplotlib.pyplot as plt

def problem_dynamics(state, control):
    """Problem-specific dynamics function.
    
    Detailed explanation of the dynamics...
    """
    # Implementation
    pass

def problem_cost(state, control):
    """Problem-specific cost function.
    
    Detailed explanation of the cost...
    """
    # Implementation
    pass

def run_example():
    """Main example function."""
    print("Running [Example Name]...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create controller
    controller = diff_mppi.create_mppi_controller(
        # Parameters with comments
    )
    
    # Solve
    result = controller.solve(initial_state)
    
    # Analyze and visualize
    plot_results(result)
    
    return result

def plot_results(result):
    """Visualization function."""
    # Plotting code
    pass

if __name__ == "__main__":
    result = run_example()
```

## Pull Request Process

### 1. Branch Creation

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Development

- Make your changes
- Add tests for new functionality
- Update documentation
- Ensure all checks pass

### 3. Commit Guidelines

**Commit Message Format:**
```
type(scope): brief description

Longer explanation if needed.

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(core): add RMSprop acceleration method"
git commit -m "fix(examples): resolve CUDA memory issue in navigation example"
git commit -m "docs(api): add comprehensive docstrings to DiffMPPI class"
```

### 4. Submission

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

**Pull Request Template:**
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### 5. Review Process

1. **Automated Checks**: CI/CD runs tests and quality checks
2. **Code Review**: Maintainers review the code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, maintainers will merge

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backwards compatible)
- Patch: Bug fixes (backwards compatible)

### Release Checklist

1. Update version in `setup.py` and `__init__.py`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and upload to PyPI
5. Update documentation

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful**: Treat everyone with respect
- **Be inclusive**: Welcome newcomers and diverse perspectives
- **Be constructive**: Provide helpful feedback
- **Be patient**: Help others learn and grow

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community chat
- **Pull Requests**: Code contributions and reviews

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributor graphs

## Specific Contribution Areas

### High-Priority Areas

1. **Performance Optimization**
   - GPU memory optimization
   - Faster dynamics evaluation
   - Parallel trajectory sampling

2. **New Acceleration Methods**
   - Advanced optimization algorithms
   - Adaptive sampling strategies
   - Online learning methods

3. **Example Applications**
   - Robotics applications
   - Game AI examples
   - Real-world control problems

4. **Testing and Validation**
   - Comprehensive test suite
   - Performance benchmarks
   - Numerical validation

### Getting Started Suggestions

**For Beginners:**
- Fix documentation typos
- Add type hints to existing code
- Create simple examples
- Improve error messages

**For Intermediate Contributors:**
- Add new acceleration methods
- Implement constraint handling
- Create visualization tools
- Optimize memory usage

**For Advanced Contributors:**
- Implement distributed computing
- Add advanced acceleration methods
- Create real-time control interfaces
- Research integration with RL frameworks

## Resources

### Learning Materials

- [Path Integral Methods in Control](link-to-paper)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Git Workflow](https://guides.github.com/introduction/flow/)

### Development Tools

- [VS Code](https://code.visualstudio.com/) - Recommended IDE
- [PyCharm](https://www.jetbrains.com/pycharm/) - Alternative IDE
- [GitHub CLI](https://cli.github.com/) - Command-line GitHub interface
- [pre-commit](https://pre-commit.com/) - Git hooks framework

## Questions?

- Check existing [GitHub Issues](https://github.com/kisisjrlly/diff-mppi/issues)
- Ask in [GitHub Discussions](https://github.com/kisisjrlly/diff-mppi/discussions)
- Read the [documentation](docs/)

Thank you for contributing to diff-mppi!
