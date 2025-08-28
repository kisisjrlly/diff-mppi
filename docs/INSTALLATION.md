# Installation and Setup Guide

## System Requirements

### Hardware Requirements

**Minimum Requirements:**
- CPU: x86_64 processor with SSE4.2 support
- RAM: 4 GB
- Storage: 500 MB available space

**Recommended Requirements:**
- CPU: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- RAM: 8 GB or more
- GPU: NVIDIA GPU with CUDA support (GTX 1060 or better)
- Storage: 2 GB available space

### Software Requirements

**Operating Systems:**
- Linux (Ubuntu 18.04+, CentOS 7+, or equivalent)
- macOS 10.14+
- Windows 10+

**Python Environment:**
- Python 3.7, 3.8, 3.9, 3.10, or 3.11
- pip 21.0+

**Dependencies:**
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0 (for examples)
- SciPy >= 1.7.0 (optional, for advanced features)

## Installation Methods

### Method 1: Install from Source (Recommended)

This is the recommended method for development and getting the latest features.

```bash
# Clone the repository
git clone https://github.com/kisisjrlly/diff-mppi.git
cd diff-mppi

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Method 2: Install from PyPI (Future)

Once published to PyPI, you can install directly:

```bash
pip install diff-mppi
```

### Method 3: Docker Installation

For isolated environments or deployment:

```dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["python", "examples/pendulum_example_clean.py"]
```

```bash
# Build and run
docker build -t diff-mppi .
docker run --gpus all diff-mppi
```

## Environment Setup

### Conda Environment

If you prefer conda for package management:

```bash
# Create new environment
conda create -n diff-mppi python=3.9
conda activate diff-mppi

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install numpy matplotlib scipy

# Install diff-mppi
cd diff-mppi
pip install -e .
```

### GPU Setup

For GPU acceleration, ensure you have:

1. **NVIDIA Driver**: Version 470.57.02 or later
2. **CUDA Toolkit**: Version 11.0 or later
3. **PyTorch with CUDA**: Install appropriate version

**Verify GPU setup:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Jupyter Notebook Setup

For interactive development:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name diff-mppi --display-name "Diff-MPPI"

# Start Jupyter
jupyter notebook
```

## Verification

### Basic Installation Test

```bash
cd diff-mppi
python -c "import diff_mppi; print('✅ Import successful')"
```

### Comprehensive Test

```bash
# Run the pendulum example
python examples/pendulum_example_clean.py
```

Expected output:
```
Using device: cuda
Testing: Standard MPPI
  Final angle: 179.2°
  Final velocity: -0.018 rad/s
Testing: MPPI + Adam
  Final angle: 178.8°
  Final velocity: -0.006 rad/s
...
🎉 All tests passed! Library is working correctly.
```

### Performance Benchmark

```bash
python -c "
import torch
import time
import diff_mppi

# Simple benchmark
def simple_dynamics(state, control):
    return state + 0.1 * control

def simple_cost(state, control):
    return torch.sum(state**2) + 0.1 * torch.sum(control**2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
controller = diff_mppi.create_mppi_controller(
    state_dim=5, control_dim=2,
    dynamics_fn=simple_dynamics,
    cost_fn=simple_cost,
    horizon=30, num_samples=200,
    device=device
)

x0 = torch.randn(5, device=device)

# Benchmark
start = time.time()
solution = controller.solve(x0, num_iterations=5)
elapsed = time.time() - start

print(f'Device: {device}')
print(f'Time for 5 iterations: {elapsed:.3f}s')
print(f'Time per iteration: {elapsed/5:.3f}s')
"
```

## Troubleshooting

### Common Installation Issues

#### Issue 1: PyTorch Installation Fails

**Symptoms:** 
```
ERROR: Could not find a version that satisfies the requirement torch>=1.9.0
```

**Solution:**
```bash
# Install PyTorch first
pip install torch torchvision torchaudio

# Then install diff-mppi
pip install -e .
```

#### Issue 2: CUDA Not Available

**Symptoms:**
```
CUDA available: False
```

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

2. **Install CUDA-enabled PyTorch:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify CUDA installation:**
   ```bash
   nvcc --version
   ```

#### Issue 3: Import Errors

**Symptoms:**
```
ImportError: cannot import name 'DiffMPPI' from 'diff_mppi'
```

**Solutions:**

1. **Reinstall in development mode:**
   ```bash
   pip uninstall diff-mppi
   pip install -e .
   ```

2. **Check Python path:**
   ```bash
   python -c "import sys; print(sys.path)"
   ```

#### Issue 4: Memory Issues

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce problem size:**
   ```python
   controller = diff_mppi.create_mppi_controller(
       ...,
       horizon=20,      # Reduce from 50
       num_samples=100, # Reduce from 500
   )
   ```

2. **Use CPU:**
   ```python
   controller = diff_mppi.create_mppi_controller(..., device="cpu")
   ```

3. **Clear GPU cache:**
   ```python
   torch.cuda.empty_cache()
   ```

### Performance Issues

#### Slow Performance

**Diagnosis:**
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    solution = controller.solve(x0, num_iterations=3)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Common causes and solutions:**

1. **CPU-GPU transfers:**
   ```python
   # Keep all tensors on same device
   x0 = x0.to(controller.device)
   ```

2. **Inefficient dynamics/cost functions:**
   ```python
   # Vectorize operations
   def efficient_dynamics(state, control):
       # Use torch operations instead of loops
       return torch.matmul(state, A) + torch.matmul(control, B)
   ```

3. **Small batch sizes:**
   ```python
   # Increase num_samples for better GPU utilization
   controller = diff_mppi.create_mppi_controller(..., num_samples=500)
   ```

### Environment Conflicts

#### Package Version Conflicts

**Check versions:**
```bash
pip list | grep -E "(torch|numpy|matplotlib)"
```

**Create clean environment:**
```bash
# Remove old environment
conda env remove -n diff-mppi

# Create fresh environment
conda create -n diff-mppi python=3.9
conda activate diff-mppi

# Install from scratch
pip install torch numpy matplotlib
pip install -e .
```

#### Python Path Issues

**Clear Python cache:**
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

## Development Setup

### For Contributors

```bash
# Clone with development tools
git clone https://github.com/kisisjrlly/diff-mppi.git
cd diff-mppi

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
flake8 diff_mppi/
black diff_mppi/
mypy diff_mppi/
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add → Existing environment → Select `venv/bin/python`
3. Configure code style to use Black formatter

## Platform-Specific Notes

### Linux

**Ubuntu/Debian:**
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip git

# For GPU support
sudo apt install nvidia-driver-470 nvidia-cuda-toolkit
```

**CentOS/RHEL:**
```bash
# Install system dependencies
sudo yum install python3-devel python3-pip git

# For GPU support (requires EPEL)
sudo yum install nvidia-driver nvidia-cuda-toolkit
```

### macOS

**Using Homebrew:**
```bash
# Install Python
brew install python@3.9

# Clone and install
git clone https://github.com/kisisjrlly/diff-mppi.git
cd diff-mppi
pip3 install -e .
```

**Note:** GPU acceleration not available on macOS (no CUDA support).

### Windows

**Using Anaconda (Recommended):**
```cmd
# Install Anaconda from https://www.anaconda.com/
# Open Anaconda Prompt

conda create -n diff-mppi python=3.9
conda activate diff-mppi

# Install PyTorch with CUDA (if available)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install diff-mppi
git clone https://github.com/kisisjrlly/diff-mppi.git
cd diff-mppi
pip install -e .
```

**Using WSL2 (Alternative):**
```bash
# Install WSL2 with Ubuntu
# Follow Linux installation instructions inside WSL2
```

## Next Steps

After successful installation:

1. **Read the documentation:** Start with `docs/API_REFERENCE.md`
2. **Run examples:** Explore `examples/` directory
3. **Join the community:** Check GitHub issues and discussions
4. **Contribute:** See `CONTRIBUTING.md` for contribution guidelines

## Support

If you encounter issues not covered here:

1. **Check GitHub Issues:** Search existing issues
2. **Create new issue:** Provide detailed error messages and system info
3. **Discussion forum:** Join community discussions
4. **Documentation:** Read full documentation in `docs/` directory

### System Information Script

For bug reports, run this script and include the output:

```python
import sys
import torch
import platform
import diff_mppi

print("System Information:")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Diff-MPPI: {diff_mppi.__version__}")
```
