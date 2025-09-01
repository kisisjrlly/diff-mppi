# API Reference

## Core Classes and Functions

### `DiffMPPI` Class

The main controller class implementing Path Integral Networks.

```python
class DiffMPPI:
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        horizon: int = 20,
        num_samples: int = 100,
        temperature: float = 1.0,
        control_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        acceleration: Optional[str] = None,
        lr: float = 0.01,
        momentum: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        device: str = "cpu",
        **kwargs
    )
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_dim` | int | Dimension of the state space |
| `control_dim` | int | Dimension of the control space |
| `dynamics_fn` | Callable | Function mapping (state, control) → next_state |
| `cost_fn` | Callable | Function mapping (state, control) → cost |
| `horizon` | int | Planning horizon length (default: 20) |
| `num_samples` | int | Number of trajectory samples for Monte Carlo (default: 100) |
| `temperature` | float | Temperature parameter λ for path integral (default: 1.0) |
| `control_bounds` | Optional[Tuple] | (min_control, max_control) tensors for clipping |
| `acceleration` | Optional[str] | Acceleration method: "adam", "nag", "rmsprop", or None |
| `lr` | float | Learning rate for acceleration methods (default: 0.01) |
| `momentum` | float | Momentum parameter for NAG/RMSprop (default: 0.9) |
| `eps` | float | Epsilon parameter for Adam/RMSprop (default: 1e-8) |
| `weight_decay` | float | Weight decay for regularization (default: 0.0) |
| `device` | str | PyTorch device: "cpu" or "cuda" (default: "cpu") |

#### Methods

##### `solve(initial_state, num_iterations=10, verbose=False) -> torch.Tensor`

Solve for optimal control sequence using iterative MPPI.

**Parameters:**
- `initial_state` (torch.Tensor): Initial state [state_dim]
- `num_iterations` (int): Number of optimization iterations (default: 10)
- `verbose` (bool): Print convergence information (default: False)

**Returns:**
- `torch.Tensor`: Optimal control sequence [horizon, control_dim]

**Example:**
```python
controller = DiffMPPI(...)
x0 = torch.tensor([1.0, 0.0, 0.0])
optimal_control = controller.solve(x0, num_iterations=20)
```

##### `rollout(initial_state, control_sequence=None) -> torch.Tensor`

Simulate system trajectory given control sequence.

**Parameters:**
- `initial_state` (torch.Tensor): Initial state [state_dim]
- `control_sequence` (torch.Tensor, optional): Control inputs [horizon, control_dim]. If None, uses current control sequence.

**Returns:**
- `torch.Tensor`: State trajectory [horizon+1, state_dim]

**Example:**
```python
trajectory = controller.rollout(x0, optimal_control)
```

##### `step(state) -> torch.Tensor`

Get single control action for Model Predictive Control.

**Parameters:**
- `state` (torch.Tensor): Current state [state_dim]

**Returns:**
- `torch.Tensor`: Control action [control_dim]

**Example:**
```python
# Real-time control loop
current_state = get_current_state()
control_action = controller.step(current_state)
apply_control(control_action)
```

### Helper Functions

#### `create_mppi_controller(**kwargs) -> DiffMPPI`

Convenience function to create MPPI controller with default settings.

**Parameters:**
- All parameters from `DiffMPPI.__init__()`

**Returns:**
- `DiffMPPI`: Configured controller instance

**Example:**
```python
import diff_mppi

controller = diff_mppi.create_mppi_controller(
    state_dim=3,
    control_dim=1,
    dynamics_fn=pendulum_dynamics,
    cost_fn=pendulum_cost,
    horizon=30,
    num_samples=100
)
```

##### `reset() -> None`

Reset controller state, including control sequence and acceleration-specific state variables.

**Example:**
```python
controller.reset()  # Clear current solution and optimizer state
```

## Function Interfaces

### Dynamics Function

The dynamics function must have the following signature:

```python
def dynamics_fn(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """
    Compute next state given current state and control.
    
    Args:
        state: Current state [batch_size, state_dim]
        control: Control input [batch_size, control_dim]
        
    Returns:
        Next state [batch_size, state_dim]
    """
    pass
```

**Important Notes:**
- Function must handle batched inputs
- Must be differentiable if using gradient-based methods
- Should implement proper integration scheme (Euler, RK4, etc.)

**Example:**
```python
def pendulum_dynamics(state, control):
    # Parameters
    dt = 0.05
    g, l, m = 9.81, 1.0, 1.0
    
    cos_theta, sin_theta, theta_dot = state[:, 0:1], state[:, 1:2], state[:, 2:3]
    torque = control[:, 0:1]
    
    # Compute angle and dynamics
    theta = torch.atan2(sin_theta, cos_theta)
    theta_ddot = (3*g/(2*l)) * torch.sin(theta) + (3/(m*l**2)) * torque
    
    # Integration
    new_theta_dot = theta_dot + dt * theta_ddot
    new_theta = theta + dt * new_theta_dot
    
    return torch.cat([torch.cos(new_theta), torch.sin(new_theta), new_theta_dot], dim=1)
```

### Cost Function

The cost function must have the following signature:

```python
def cost_fn(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """
    Compute instantaneous cost for state-control pair.
    
    Args:
        state: State [batch_size, state_dim]
        control: Control input [batch_size, control_dim]
        
    Returns:
        Cost values [batch_size]
    """
    pass
```

**Important Notes:**
- Must return scalar cost for each sample
- Should be differentiable for gradient-based optimization
- Typically includes state tracking cost + control effort penalty

**Example:**
```python
def pendulum_cost(state, control):
    cos_theta = state[:, 0]
    theta_dot = state[:, 2]
    torque = control[:, 0]
    
    # Cost components
    angle_cost = (1.0 + cos_theta)**2  # Reward upright position
    velocity_cost = 0.1 * theta_dot**2
    control_cost = 0.01 * torque**2
    
    return angle_cost + velocity_cost + control_cost
```

## Acceleration Methods

### None (Standard MPPI)

Classic path integral method without acceleration.

```python
controller = DiffMPPI(..., acceleration=None)
```

### Adam

Adaptive moment estimation with bias correction.

```python
controller = DiffMPPI(
    ...,
    acceleration="adam",
    lr=0.01,
    eps=1e-8,
    weight_decay=0.0
)
```

**Parameters:**
- `lr`: Learning rate (default: 0.01)  
- `eps`: Numerical stability parameter (default: 1e-8)
- `weight_decay`: L2 regularization weight (default: 0.0)

**Note:** Uses fixed β₁=0.9, β₂=0.999 for moment decay rates.

### NAG (Nesterov Accelerated Gradient)

Momentum-based acceleration with look-ahead.

```python
controller = DiffMPPI(
    ...,
    acceleration="nag",
    lr=0.01,
    momentum=0.9
)
```

### RMSprop

Root mean square propagation for adaptive learning rates.

```python
controller = DiffMPPI(
    ...,
    acceleration="rmsprop",
    lr=0.01,
    momentum=0.9,
    eps=1e-8
)
```

## Error Handling

The library provides informative error messages for common issues:

### Invalid Parameters
```python
# Raises ValueError
controller = DiffMPPI(state_dim=0, ...)  # Invalid dimensions
controller = DiffMPPI(..., acceleration="invalid")  # Unknown acceleration
```

### Runtime Errors
```python
# Raises RuntimeError
controller.solve(torch.tensor([1, 2]))  # Wrong state dimension
controller.rollout(x0, torch.zeros(5, 2))  # Wrong control dimension
```

### Device Mismatch
```python
# Raises RuntimeError
controller = DiffMPPI(..., device="cuda")
x0_cpu = torch.tensor([1.0, 0.0])  # CPU tensor
controller.solve(x0_cpu)  # Device mismatch
```

## Performance Guidelines

### Memory Usage

Memory usage scales as O(horizon × num_samples × state_dim):

| Configuration | GPU Memory |
|---------------|------------|
| H=20, K=100, state_dim=3 | ~1 MB |
| H=50, K=500, state_dim=10 | ~100 MB |
| H=100, K=1000, state_dim=50 | ~2 GB |

### Computational Complexity

Time complexity per iteration: O(H × K × (dynamics_cost + cost_cost))

**Recommendations:**
- Start with smaller num_samples for prototyping
- Use GPU for num_samples > 200
- Increase horizon for better long-term planning
- Balance num_samples vs num_iterations

### Convergence Tips

1. **Temperature Selection:**
   - Lower temperature (0.01-0.1): More exploitative, faster convergence
   - Higher temperature (0.5-2.0): More explorative, better global search

2. **Sample Count:**
   - More samples: Better approximation, slower computation
   - Typical range: 50-500 samples

3. **Acceleration Method Selection:**
   - **Standard**: Most stable, good baseline
   - **Adam**: Best for most problems, adaptive learning
   - **NAG**: Fast convergence, may oscillate
   - **RMSprop**: Good for problems with varying curvature

## Integration Examples

### With Neural Networks

```python
import torch.nn as nn

class NeuralDynamics(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, control):
        x = torch.cat([state, control], dim=1)
        return state + self.net(x)  # Residual connection

# Usage
dynamics_model = NeuralDynamics(3, 1)
controller = diff_mppi.create_mppi_controller(
    state_dim=3, control_dim=1,
    dynamics_fn=dynamics_model,
    cost_fn=my_cost
)
```

### With Reinforcement Learning

```python
# Use MPPI as a planning component
class MPPIAgent:
    def __init__(self, dynamics_fn, cost_fn):
        self.controller = diff_mppi.create_mppi_controller(
            state_dim=3, control_dim=1,
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            horizon=20
        )
    
    def act(self, observation):
        return self.controller.step(observation)
```

### Real-time Control

```python
import time

# Real-time control loop
controller = diff_mppi.create_mppi_controller(...)

while True:
    start_time = time.time()
    
    # Get current state
    current_state = get_sensor_data()
    
    # Compute control
    control = controller.step(current_state)
    
    # Apply control
    send_control_command(control)
    
    # Maintain control frequency
    elapsed = time.time() - start_time
    time.sleep(max(0, 0.05 - elapsed))  # 20 Hz control
```

## Batch Processing Support

The diff-mppi library supports efficient batch processing of multiple initial states simultaneously, which is crucial for applications requiring high throughput or parallel processing.

### Key Features

1. **Automatic batch detection**: The library automatically detects whether inputs are single states or batches based on tensor dimensions
2. **Parallel GPU processing**: Leverages GPU parallelism for significant speedup (typically 3-4x faster than sequential processing)
3. **Memory efficient**: Optimized memory usage for large batch sizes through efficient tensor operations
4. **Full API compatibility**: All methods (solve, step, rollout) support both single and batch modes transparently

### Batch Processing API

All core methods automatically support batch processing:

```python
# Single initial state
initial_state = torch.tensor([1.0, 0.0])  # Shape: [state_dim]
controls = controller.solve(initial_state)  # Shape: [horizon, control_dim]

# Batch of initial states  
initial_states = torch.tensor([
    [1.0, 0.0], 
    [2.0, 1.0], 
    [0.5, -0.5]
])  # Shape: [batch_size, state_dim]
controls = controller.solve(initial_states)  # Shape: [batch_size, horizon, control_dim]
```

### Applications

Batch processing is particularly beneficial for:

- **Monte Carlo simulations**: Process multiple random initial states in parallel
- **Robust control design**: Evaluate performance across multiple scenarios simultaneously
- **Multi-agent planning**: Plan for multiple agents in parallel
- **Imitation learning**: Process multiple demonstration trajectories efficiently
- **Hyperparameter tuning**: Test different parameter configurations in parallel

### Performance Benefits

Typical performance improvements with batch processing:

| Batch Size | GPU Speedup | Memory Usage |
|------------|-------------|--------------|
| 1          | 1.0x        | Baseline     |
| 4          | 2.8x        | 3.2x         |
| 8          | 3.5x        | 6.1x         |
| 16         | 3.8x        | 11.8x        |

### Example: Batch Processing

```python
import torch
from diff_mppi import DiffMPPI

# Define dynamics and cost functions (must support batch processing)
def batch_dynamics(state, control):
    # state: [batch_size, state_dim], control: [batch_size, control_dim]
    # return: [batch_size, state_dim]
    return state + control  # Simple example

def batch_cost(state, control):
    # state: [batch_size, state_dim], control: [batch_size, control_dim]  
    # return: [batch_size]
    return torch.sum(state**2 + control**2, dim=1)

# Create controller
controller = DiffMPPI(
    dynamics_fn=batch_dynamics,
    cost_fn=batch_cost,
    state_dim=2,
    control_dim=1,
    horizon=20,
    num_samples=100,
    device="cuda"
)

# Process multiple initial states
batch_initial_states = torch.randn(8, 2, device="cuda")
batch_controls = controller.solve(batch_initial_states)

print(f"Input shape: {batch_initial_states.shape}")    # [8, 2]
print(f"Output shape: {batch_controls.shape}")         # [8, 20, 1]
```

### Implementation Notes

1. **Function signatures**: Dynamics and cost functions must handle batch dimensions properly
2. **Memory considerations**: Large batch sizes may require reducing num_samples to fit in GPU memory
3. **Synchronization**: All trajectories in a batch are processed for the same number of iterations
4. **Gradient flow**: Batch processing maintains proper gradient flow for end-to-end learning
