# API Reference

## Core Classes and Functions

### `DiffMPPI` Class

The main controller class implementing Differentiable Model Predictive Path Integral (Diff-MPPI) control with gradient-based acceleration methods.

```python
class DiffMPPI:
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        terminal_cost_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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
        # NAG-specific parameters (paper defaults)
        nag_gamma: float = 0.8,
        # Adam-specific parameters (paper defaults)
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_lr: float = 1e-3,
        # AdaGrad-specific parameters
        adagrad_eta0: Optional[float] = None,
        **kwargs
    )
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_dim` | int | Dimension of the state space. Must be positive integer. |
| `control_dim` | int | Dimension of the control space. Must be positive integer. |
| `dynamics_fn` | Callable | Function mapping (state, control) → next_state. Must handle batched inputs with signature `f(state: [batch, state_dim], control: [batch, control_dim]) -> [batch, state_dim]`. Should be differentiable for gradient-based learning. |
| `cost_fn` | Callable | Function mapping (state, control) → cost. Must handle batched inputs with signature `g(state: [batch, state_dim], control: [batch, control_dim]) -> [batch]`. Should return scalar costs per sample. |
| `terminal_cost_fn` | Optional[Callable] | Optional terminal cost function φ(state) → cost. If provided, adds terminal cost at horizon end. Signature: `φ(state: [batch, state_dim]) -> [batch]`. |
| `horizon` | int | Planning horizon length T (number of time steps). Larger horizons provide better long-term planning but increase computational cost. (default: 20) |
| `num_samples` | int | Number of trajectory samples K for Monte Carlo approximation. More samples improve accuracy but increase computation. Typical range: 50-1000. (default: 100) |
| `temperature` | float | Temperature parameter λ for path integral weight calculation. Lower values (0.01-0.1) are more exploitative, higher values (0.5-2.0) are more explorative. Controls the sharpness of trajectory selection. (default: 1.0) |
| `control_bounds` | Optional[Tuple] | Optional control bounds as (min_control, max_control) tensors of shape [control_dim]. Controls are clipped to these bounds after each update. |
| `acceleration` | Optional[str] | Acceleration method for gradient-based optimization. Options: `"adam"`, `"nag"`, `"adagrad"`, or `None` for standard MPPI. Each method has different convergence properties. |
| `lr` | float | Base learning rate for acceleration methods. Used as fallback learning rate and for AdaGrad when `adagrad_eta0` not specified. (default: 0.01) |
| `momentum` | float | Legacy momentum parameter kept for compatibility. Not used in current implementation. (default: 0.9) |
| `eps` | float | Epsilon parameter for numerical stability in Adam and AdaGrad optimizers. Prevents division by zero in adaptive learning rate computations. (default: 1e-8) |
| `weight_decay` | float | L2 regularization weight for control sequences. Adds penalty proportional to control magnitude to prevent excessive control effort. (default: 0.0) |
| `device` | str | PyTorch device for computation: `"cpu"` or `"cuda"`. All tensors and computations will be performed on this device. (default: "cpu") |
| **NAG Parameters** | | **Nesterov Accelerated Gradient (Paper: Okada & Taniguchi, 2018)** |
| `nag_gamma` | float | NAG momentum decay coefficient γ. Controls influence of historical momentum in sampling. Paper recommends 0.8. Must be < 1 for stability. (default: 0.8) |
| **Adam Parameters** | | **Adaptive Moment Estimation (Paper: Section IV-C)** |
| `adam_beta1` | float | Adam first moment decay rate β₁. Controls exponential moving average of gradients. Fixed at paper specification. (default: 0.9) |
| `adam_beta2` | float | Adam second moment decay rate β₂. Controls exponential moving average of squared gradients. Fixed at paper specification. (default: 0.999) |
| `adam_lr` | float | Adam-specific learning rate η. Used for Adam optimizer updates, separate from base `lr`. Paper default value. (default: 1e-3) |
| **AdaGrad Parameters** | | **Adaptive Gradient Algorithm (Paper: Section IV-D)** |
| `adagrad_eta0` | Optional[float] | AdaGrad initial step size η₀. If None, uses `lr` value. Controls initial learning rate before adaptation. |

#### Methods

##### `solve(initial_state, num_iterations=10, verbose=False) -> torch.Tensor`

Solve for optimal control sequence using iterative Diff-MPPI algorithm.

**Parameters:**
- `initial_state` (torch.Tensor): Initial state(s). Can be:
  - Single state: `[state_dim]` - returns `[horizon, control_dim]`
  - Batch of states: `[batch_size, state_dim]` - returns `[batch_size, horizon, control_dim]`
- `num_iterations` (int): Number of MPPI optimization iterations. More iterations improve convergence but increase computation time. (default: 10)
- `verbose` (bool): Print convergence information every 5 iterations, showing average cost. (default: False)

**Returns:**
- `torch.Tensor`: Optimal control sequence(s) with gradients detached for immediate use.

**Example:**
```python
# Single initial state
controller = DiffMPPI(state_dim=3, control_dim=1, ...)
x0 = torch.tensor([1.0, 0.0, 0.0])
optimal_control = controller.solve(x0, num_iterations=20, verbose=True)
print(optimal_control.shape)  # [20, 1]

# Batch of initial states
x0_batch = torch.tensor([[1.0, 0.0, 0.0], [2.0, 1.0, 0.5]])
optimal_controls = controller.solve(x0_batch, num_iterations=15)
print(optimal_controls.shape)  # [2, 20, 1]
```

##### `rollout(initial_state, control_sequence=None) -> torch.Tensor`

Simulate system trajectory given initial state(s) and control sequence(s).

**Parameters:**
- `initial_state` (torch.Tensor): Initial state(s). Supports both single states `[state_dim]` and batches `[batch_size, state_dim]`.
- `control_sequence` (torch.Tensor, optional): Control inputs. If None, uses current internal control sequence. Shapes:
  - For single state: `[horizon, control_dim]`
  - For batch: `[batch_size, horizon, control_dim]` or `[horizon, control_dim]` (broadcast to all batch elements)

**Returns:**
- `torch.Tensor`: State trajectory(ies):
  - Single state: `[horizon+1, state_dim]` (includes initial state)
  - Batch: `[batch_size, horizon+1, state_dim]`

**Example:**
```python
# Rollout with current control sequence
trajectory = controller.rollout(x0)

# Rollout with custom control sequence
custom_controls = torch.randn(20, 1)  # Random controls
trajectory = controller.rollout(x0, custom_controls)
print(trajectory.shape)  # [21, 3] (horizon+1, state_dim)
```

##### `step(state) -> torch.Tensor`

Get single control action for Model Predictive Control (MPC) style operation.

**Parameters:**
- `state` (torch.Tensor): Current state(s). Supports:
  - Single state: `[state_dim]` - returns `[control_dim]`
  - Batch of states: `[batch_size, state_dim]` - returns `[batch_size, control_dim]`

**Returns:**
- `torch.Tensor`: Control action(s) for the current time step (first element of optimized sequence).

**Note:** This method runs a short optimization (5 iterations) for real-time performance.

**Example:**
```python
# Real-time MPC control loop
current_state = get_current_state()  # Get from sensors
control_action = controller.step(current_state)
apply_control(control_action)  # Send to actuators

# Batch processing for multiple agents
states_batch = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
actions_batch = controller.step(states_batch)
print(actions_batch.shape)  # [2, 1] - one action per agent
```

##### `reset() -> None`

Reset controller state, clearing all internal variables and acceleration states.

**Clears:**
- Current control sequence (reset to zeros)
- Acceleration-specific state variables (Adam moments, NAG momentum, AdaGrad accumulation)
- Batch processing variables if they exist

**Example:**
```python
# Reset before solving a new problem
controller.reset()
new_solution = controller.solve(new_initial_state)

# Reset when switching between different tasks
controller.reset()  
```

### Helper Functions

#### `create_mppi_controller(**kwargs) -> DiffMPPI`

Convenience factory function to create MPPI controller with default settings.

**Parameters:**
- All parameters from `DiffMPPI.__init__()` are supported as keyword arguments

**Returns:**
- `DiffMPPI`: Configured controller instance with specified parameters

**Example:**
```python
import diff_mppi

# Create controller with custom parameters
controller = diff_mppi.create_mppi_controller(
    state_dim=3,
    control_dim=1,
    dynamics_fn=pendulum_dynamics,
    cost_fn=pendulum_cost,
    horizon=30,
    num_samples=200,
    acceleration="adam",
    device="cuda"
)

# Equivalent to:
# controller = diff_mppi.DiffMPPI(
#     state_dim=3, control_dim=1,
#     dynamics_fn=pendulum_dynamics, cost_fn=pendulum_cost,
#     horizon=30, num_samples=200, acceleration="adam", device="cuda"
# )
```

## Function Interfaces

### Dynamics Function

The dynamics function must have the following signature and properties:

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

**Requirements:**
- **Batched processing**: Must handle batched inputs efficiently
- **Differentiability**: Must be differentiable for gradient-based acceleration methods
- **Integration scheme**: Should implement proper numerical integration (Euler, RK4, etc.)
- **Tensor operations**: Use PyTorch operations for GPU compatibility
- **Shape preservation**: Output must match state dimensions

**Example:**
```python
def pendulum_dynamics(state, control):
    """Pendulum dynamics with cos/sin state representation."""
    # Parameters
    dt = 0.05  # Integration time step
    g, l, m = 9.81, 1.0, 1.0  # Gravity, length, mass
    
    # Extract state components
    cos_theta, sin_theta, theta_dot = state[:, 0:1], state[:, 1:2], state[:, 2:3]
    torque = control[:, 0:1]
    
    # Compute angle and angular acceleration
    theta = torch.atan2(sin_theta, cos_theta)
    theta_ddot = (3*g/(2*l)) * torch.sin(theta) + (3/(m*l**2)) * torque
    
    # Euler integration
    new_theta_dot = theta_dot + dt * theta_ddot
    new_theta = theta + dt * new_theta_dot
    
    # Return next state in cos/sin representation
    return torch.cat([
        torch.cos(new_theta), 
        torch.sin(new_theta), 
        new_theta_dot
    ], dim=1)

def cartpole_dynamics(state, control):
    """Cart-pole dynamics example."""
    dt = 0.02
    gravity, masscart, masspole, length = 9.8, 1.0, 0.1, 0.5
    
    x, x_dot, theta, theta_dot = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]
    force = control[:, 0:1]
    
    costheta, sintheta = torch.cos(theta), torch.sin(theta)
    temp = (force + masspole * length * theta_dot**2 * sintheta) / (masscart + masspole)
    
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0/3.0 - masspole * costheta**2 / (masscart + masspole))
    )
    xacc = temp - masspole * length * thetaacc * costheta / (masscart + masspole)
    
    # Euler integration
    new_x = x + dt * x_dot
    new_x_dot = x_dot + dt * xacc
    new_theta = theta + dt * theta_dot
    new_theta_dot = theta_dot + dt * thetaacc
    
    return torch.cat([new_x, new_x_dot, new_theta, new_theta_dot], dim=1)
```

### Cost Function

The cost function must have the following signature and properties:

```python
def cost_fn(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """
    Compute instantaneous cost for state-control pairs.
    
    Args:
        state: State [batch_size, state_dim]
        control: Control input [batch_size, control_dim]
        
    Returns:
        Cost values [batch_size] - scalar cost per sample
    """
    pass
```

**Requirements:**
- **Scalar output**: Must return one scalar cost per sample
- **Differentiability**: Should be differentiable for gradient-based optimization
- **Batch processing**: Handle batched inputs efficiently
- **Non-negative**: Costs should typically be non-negative
- **Meaningful gradients**: Should provide informative gradients for optimization

**Cost Design Principles:**
- **State tracking**: Penalize deviation from desired states
- **Control effort**: Penalize excessive control usage
- **Constraints**: Add penalties for constraint violations
- **Smoothness**: Consider adding smoothness penalties

**Example:**
```python
def pendulum_cost(state, control):
    """Pendulum swing-up cost function."""
    cos_theta = state[:, 0]  # cos(θ)
    theta_dot = state[:, 2]  # angular velocity
    torque = control[:, 0]   # control torque
    
    # Cost components
    angle_cost = (1.0 - cos_theta)**2  # Reward upright position (cos(0) = 1)
    velocity_cost = 0.1 * theta_dot**2  # Penalize high velocities
    control_cost = 0.01 * torque**2     # Penalize control effort
    
    return angle_cost + velocity_cost + control_cost

def quadrotor_cost(state, control):
    """Quadrotor trajectory tracking cost."""
    # Extract state components
    position = state[:, 0:3]      # [x, y, z]
    velocity = state[:, 3:6]      # [vx, vy, vz]
    orientation = state[:, 6:10]  # quaternion [qw, qx, qy, qz]
    angular_vel = state[:, 10:13] # [wx, wy, wz]
    
    # Control inputs (motor commands)
    motor_commands = control  # [u1, u2, u3, u4]
    
    # Target (hover at origin)
    target_pos = torch.zeros_like(position)
    target_vel = torch.zeros_like(velocity)
    target_quat = torch.tensor([1., 0., 0., 0.]).expand_as(orientation)
    
    # Cost components
    position_cost = torch.sum((position - target_pos)**2, dim=1)
    velocity_cost = 0.1 * torch.sum(velocity**2, dim=1)
    orientation_cost = torch.sum((orientation - target_quat)**2, dim=1)
    angular_cost = 0.1 * torch.sum(angular_vel**2, dim=1)
    control_cost = 0.01 * torch.sum(motor_commands**2, dim=1)
    
    return position_cost + velocity_cost + orientation_cost + angular_cost + control_cost

def path_following_cost(state, control, reference_path):
    """Path following cost with time-varying reference."""
    current_pos = state[:, 0:2]  # [x, y]
    current_vel = state[:, 2:4]  # [vx, vy]
    
    # Find closest point on reference path (simplified)
    distances = torch.norm(current_pos.unsqueeze(1) - reference_path.unsqueeze(0), dim=2)
    closest_indices = torch.argmin(distances, dim=1)
    closest_points = reference_path[closest_indices]
    
    # Cost components
    tracking_cost = torch.sum((current_pos - closest_points)**2, dim=1)
    velocity_cost = 0.1 * torch.sum(current_vel**2, dim=1)
    control_cost = 0.01 * torch.sum(control**2, dim=1)
    
    return tracking_cost + velocity_cost + control_cost
```

### Terminal Cost Function (Optional)

The terminal cost function adds cost at the end of the planning horizon:

```python
def terminal_cost_fn(state: torch.Tensor) -> torch.Tensor:
    """
    Compute terminal cost for final states.
    
    Args:
        state: Terminal state [batch_size, state_dim]
        
    Returns:
        Terminal cost values [batch_size]
    """
    pass
```

**Example:**
```python
def pendulum_terminal_cost(state):
    """Penalize final state deviation from upright position."""
    cos_theta = state[:, 0]
    theta_dot = state[:, 2]
    
    # Heavy penalty for not reaching upright position
    final_angle_cost = 10.0 * (1.0 - cos_theta)**2
    final_velocity_cost = 1.0 * theta_dot**2
    
    return final_angle_cost + final_velocity_cost
```

## Acceleration Methods

The library implements three gradient-based acceleration methods based on the paper "Acceleration of Gradient-Based Path Integral Method" (Okada & Taniguchi, 2018).

### None (Standard MPPI)

Classic path integral method without acceleration.

```python
controller = DiffMPPI(..., acceleration=None)
```

**Characteristics:**
- Most stable and reliable
- Good baseline performance
- No additional hyperparameters
- Suitable for all problem types

### Adam (Adaptive Moment Estimation)

Adaptive optimization with bias-corrected moment estimates.

```python
controller = DiffMPPI(
    ...,
    acceleration="adam",
    adam_lr=1e-3,      # Adam-specific learning rate
    adam_beta1=0.9,    # First moment decay (fixed)
    adam_beta2=0.999,  # Second moment decay (fixed)
    eps=1e-8,          # Numerical stability
    weight_decay=0.0   # L2 regularization
)
```

**Parameters (Paper-Compliant):**
- `adam_lr` (float): Adam learning rate η. Paper default: 1e-3. Controls step size for parameter updates.
- `adam_beta1` (float): First moment decay rate β₁. Fixed at 0.9 per paper specification. Controls momentum smoothing.
- `adam_beta2` (float): Second moment decay rate β₂. Fixed at 0.999 per paper specification. Controls variance smoothing.
- `eps` (float): Numerical stability parameter ε. Prevents division by zero in adaptive learning rates.
- `weight_decay` (float): L2 regularization weight. Adds penalty to prevent excessive control magnitudes.

**Characteristics:**
- Best for low-noise, well-conditioned problems
- Adaptive learning rates per parameter
- May be unstable in high-dimensional noisy systems
- Fast convergence when suitable

**Recommended for:**
- Low-noise dynamical systems
- Problems with well-behaved gradients
- Car racing, robot arm control

### NAG (Nesterov Accelerated Gradient)

Momentum-based acceleration with look-ahead sampling.

```python
controller = DiffMPPI(
    ...,
    acceleration="nag",
    nag_gamma=0.8,     # Momentum decay coefficient
    eps=1e-8,          # Numerical stability
    weight_decay=0.0   # L2 regularization
)
```

**Parameters (Paper-Compliant):**
- `nag_gamma` (float): Momentum decay coefficient γ. Paper recommends 0.8. Controls influence of historical momentum. Must be < 1 for stability.
- `eps` (float): Numerical stability parameter.
- `weight_decay` (float): L2 regularization weight.

**Key Features:**
- **Momentum drift sampling**: Modifies trajectory sampling distribution using historical momentum (Equation 18)
- **Look-ahead mechanism**: Uses momentum to predict better sampling directions
- **Universal stability**: Works reliably across all system types

**Characteristics:**
- Most stable and universally applicable
- 30-60% iteration reduction in experiments
- No oscillation or instability issues
- Recommended as default acceleration method

**Recommended for:**
- All types of dynamical systems
- High-dimensional problems
- Noisy systems (quadrotor, robotics)
- When reliability is crucial

### AdaGrad (Adaptive Gradient)

Adaptive learning rates based on gradient history.

```python
controller = DiffMPPI(
    ...,
    acceleration="adagrad",
    adagrad_eta0=1e-2,  # Initial step size
    eps=1e-8,           # Numerical stability
    weight_decay=0.0    # L2 regularization
)
```

**Parameters:**
- `adagrad_eta0` (float): Initial step size η₀. If None, uses `lr`. Controls initial learning rate before adaptation.
- `eps` (float): Numerical stability parameter to prevent division by zero.
- `weight_decay` (float): L2 regularization weight.

**Characteristics:**
- Adaptive step sizes per parameter dimension
- Step sizes continuously decrease over time
- **Limited effectiveness**: Poor performance with sampling noise in path integrals
- True cumulative gradient accumulation (not exponential moving average)

**Limitations:**
- Incompatible with Monte Carlo sampling noise
- Step sizes decay too quickly to zero
- No experimental acceleration demonstrated
- Not recommended for practical use

### Acceleration Method Selection Guide

| System Type | Noise Level | Recommended Method | Rationale |
|-------------|-------------|-------------------|-----------|
| **Low-dimensional, low-noise** | Low | Adam or NAG | Both work well, Adam may converge faster |
| **High-dimensional** | Any | NAG | Most stable for complex systems |
| **Noisy dynamics** | High | NAG | Robust to sampling noise |
| **Real-time applications** | Any | NAG | Reliable convergence |
| **Research/experimentation** | Any | Try all, default NAG | Compare performance for your specific problem |

### Performance Comparison (From Paper)

| Method | Convergence Speed | Stability | Universal Applicability | Final Cost Quality |
|--------|------------------|-----------|------------------------|-------------------|
| **Standard MPPI** | Baseline | ★★★★★ | ★★★★★ | ★★★★☆ |
| **NAG** | 30-60% faster | ★★★★★ | ★★★★★ | ★★★★★ |
| **Adam** | Fastest (when suitable) | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| **AdaGrad** | No improvement | ★★☆☆☆ | ★☆☆☆☆ | ★★☆☆☆ |

## Error Handling

The library provides comprehensive error handling with informative messages for common issues:

### Parameter Validation Errors

```python
# Invalid dimensions
controller = DiffMPPI(state_dim=0, control_dim=1, ...)  
# Raises: ValueError("state_dim must be positive")

controller = DiffMPPI(state_dim=3, control_dim=-1, ...)  
# Raises: ValueError("control_dim must be positive")

# Invalid acceleration method
controller = DiffMPPI(..., acceleration="invalid_method")  
# Raises: ValueError("Unknown acceleration method. Choose from: 'adam', 'nag', 'adagrad', None")

# Invalid temperature
controller = DiffMPPI(..., temperature=0.0)  
# Raises: ValueError("temperature must be positive")

# Invalid horizon
controller = DiffMPPI(..., horizon=0)  
# Raises: ValueError("horizon must be positive")
```

### Runtime Errors

```python
# Wrong state dimension
controller = DiffMPPI(state_dim=3, ...)
wrong_state = torch.tensor([1.0, 2.0])  # Only 2 dimensions
controller.solve(wrong_state)  
# Raises: RuntimeError("Expected state dimension 3, got 2")

# Wrong control dimension in rollout
wrong_controls = torch.zeros(20, 2)  # Wrong control_dim
controller.rollout(x0, wrong_controls)  
# Raises: RuntimeError("Expected control dimension 1, got 2")

# Batch size mismatch
initial_states = torch.randn(5, 3)  # 5 states
control_sequences = torch.randn(3, 20, 1)  # 3 control sequences
controller.rollout(initial_states, control_sequences)  
# Raises: RuntimeError("Batch size mismatch: states=5, controls=3")
```

### Device Mismatch Errors

```python
# Controller on GPU, input on CPU
controller = DiffMPPI(..., device="cuda")
x0_cpu = torch.tensor([1.0, 0.0, 0.0])  # CPU tensor
controller.solve(x0_cpu)  
# Raises: RuntimeError("Input tensor must be on device 'cuda', got 'cpu'")

# Mixed device tensors
control_bounds = (
    torch.tensor([-1.0]),           # CPU
    torch.tensor([1.0]).cuda()      # GPU
)
controller = DiffMPPI(..., control_bounds=control_bounds)
# Raises: RuntimeError("All tensors must be on the same device")
```

### Function Interface Errors

```python
# Dynamics function with wrong signature
def bad_dynamics(state):  # Missing control parameter
    return state

controller = DiffMPPI(dynamics_fn=bad_dynamics, ...)
# Raises: TypeError during solve() call

# Cost function returning wrong shape
def bad_cost(state, control):
    return torch.sum(state**2)  # Should return [batch_size], not scalar

controller = DiffMPPI(cost_fn=bad_cost, ...)
# Raises: RuntimeError("Cost function must return [batch_size] tensor, got shape []")
```

### Memory and Performance Warnings

```python
# Large memory usage warning
controller = DiffMPPI(horizon=100, num_samples=10000, ...)
# Warning: "Large configuration detected. Memory usage: ~8GB. Consider reducing num_samples."

# GPU memory overflow
controller = DiffMPPI(..., device="cuda", num_samples=50000)
controller.solve(torch.randn(1000, 10).cuda())  # Very large batch
# RuntimeError: "CUDA out of memory. Try reducing batch_size or num_samples."
```

## Performance Guidelines

### Memory Usage

Memory usage scales as **O(batch_size × horizon × num_samples × max(state_dim, control_dim))**:

| Configuration | Estimated GPU Memory | Use Case |
|---------------|---------------------|----------|
| H=20, K=100, state_dim=3, batch=1 | ~0.5 MB | Small problems, prototyping |
| H=30, K=200, state_dim=6, batch=1 | ~2 MB | Medium complexity systems |
| H=50, K=500, state_dim=10, batch=1 | ~20 MB | High-dimensional systems |
| H=20, K=1000, state_dim=3, batch=32 | ~80 MB | Batch processing |
| H=100, K=1000, state_dim=50, batch=1 | ~2 GB | Very complex systems |

**Memory Optimization Tips:**
- Reduce `num_samples` for prototyping, increase for final performance
- Use smaller `horizon` for real-time applications
- Process large batches sequentially if memory limited
- Monitor GPU memory usage with `torch.cuda.memory_allocated()`

### Computational Complexity

**Time complexity per iteration:** O(H × K × (T_dynamics + T_cost))
where T_dynamics and T_cost are the computational costs of dynamics and cost functions.

**Scaling recommendations:**
- **num_samples**: Start with 50-100, increase to 500-1000 for final performance
- **horizon**: Use 10-20 for real-time, 30-50 for offline planning
- **num_iterations**: 5-10 for MPC, 10-50 for offline optimization
- **GPU acceleration**: Recommended for num_samples > 200

### Convergence and Hyperparameter Tuning

#### Temperature Selection

| Temperature Range | Behavior | Best For |
|------------------|----------|----------|
| **0.001 - 0.01** | Very exploitative, sharp trajectory selection | Fine-tuning near optimal solution |
| **0.1 - 0.5** | Balanced exploration/exploitation | Most applications |
| **1.0 - 2.0** | More explorative, diverse trajectory sampling | Initial exploration, difficult landscapes |
| **> 5.0** | Very explorative, nearly uniform sampling | Global search, very noisy problems |

#### Sample Count Guidelines

| num_samples | Convergence Quality | Computation Time | Recommended For |
|-------------|-------------------|------------------|-----------------|
| **50-100** | Basic approximation | Fast | Prototyping, real-time MPC |
| **200-500** | Good approximation | Moderate | Most applications |
| **1000-2000** | High quality | Slow | Offline planning, research |
| **> 5000** | Diminishing returns | Very slow | Specialized applications only |

#### Acceleration Method Selection

```python
# For most applications (recommended default)
controller = DiffMPPI(..., acceleration="nag", nag_gamma=0.8)

# For low-noise, well-behaved systems
controller = DiffMPPI(..., acceleration="adam", adam_lr=1e-3)

# For debugging or baseline comparison
controller = DiffMPPI(..., acceleration=None)

# AdaGrad not recommended due to poor performance with sampling noise
```

#### Iteration Count Guidelines

| Application | num_iterations | Rationale |
|-------------|---------------|-----------|
| **Real-time MPC** | 3-10 | Balance performance vs computation time |
| **Offline planning** | 20-100 | Allow full convergence |
| **Research experiments** | 50-200 | Ensure convergence for fair comparison |
| **Fine-tuning** | 5-20 | Quick refinement of existing solution |

### Performance Profiling

```python
import time
import torch

# Profile memory usage
def profile_memory(controller, initial_state):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run solve
        solution = controller.solve(initial_state)
        
        # Check memory usage
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {memory_mb:.1f} MB")
        
        return solution

# Profile computation time
def profile_timing(controller, initial_state, num_runs=10):
    # Warmup
    controller.solve(initial_state, num_iterations=1)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        solution = controller.solve(initial_state, num_iterations=10)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average solve time: {avg_time:.3f}s")
    return solution

# Usage
controller = DiffMPPI(...)
x0 = torch.randn(3)

solution = profile_memory(controller, x0)
solution = profile_timing(controller, x0)
```

### Real-time Performance Tips

1. **Use GPU**: Significant speedup for num_samples > 100
```python
controller = DiffMPPI(..., device="cuda")
```

2. **Warm start MPC**: Reuse previous solution
```python
# Don't reset between MPC steps
for t in range(simulation_steps):
    control = controller.step(current_state)
    # controller.reset()  # Don't do this in MPC!
```

3. **Reduce iterations for real-time**:
```python
# Fast MPC update
control = controller.step(current_state)  # Uses 5 iterations internally
```

4. **Batch processing for multiple agents**:
```python
# Process multiple agents simultaneously
all_states = torch.stack([agent.state for agent in agents])
all_controls = controller.step(all_states)  # Batch processing
```

5. **Profile your specific problem**:
```python
# Find optimal balance for your hardware and requirements
configs = [
    {"num_samples": 100, "num_iterations": 5},
    {"num_samples": 200, "num_iterations": 3},
    {"num_samples": 50, "num_iterations": 10},
]

for config in configs:
    # Test each configuration and measure performance vs quality
    pass
```

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

The diff-mppi library provides comprehensive batch processing capabilities for efficiently handling multiple initial states, control sequences, or scenarios simultaneously.

### Key Features

1. **Automatic batch detection**: Library automatically detects single vs batch inputs based on tensor dimensions
2. **Parallel GPU processing**: Leverages GPU parallelism for 3-4x speedup over sequential processing  
3. **Memory efficient**: Optimized tensor operations for large batch sizes
4. **Full API compatibility**: All methods seamlessly support both single and batch modes
5. **Gradient preservation**: Maintains proper gradient flow for end-to-end learning

### Batch Processing API

All core methods automatically handle batch processing:

```python
# Single initial state
initial_state = torch.tensor([1.0, 0.0])  # Shape: [state_dim]
controls = controller.solve(initial_state)  # Returns: [horizon, control_dim]

# Batch of initial states  
initial_states = torch.tensor([
    [1.0, 0.0], 
    [2.0, 1.0], 
    [0.5, -0.5]
])  # Shape: [batch_size, state_dim]
controls = controller.solve(initial_states)  # Returns: [batch_size, horizon, control_dim]

# Mixed batch processing
single_state = torch.tensor([1.0, 0.0])
batch_controls = torch.randn(3, 20, 1)
# rollout() broadcasts single state to match batch dimension
trajectories = controller.rollout(single_state, batch_controls)  # [3, 21, 2]
```

### Batch Dimensions

| Method | Single Input | Single Output | Batch Input | Batch Output |
|--------|-------------|---------------|-------------|--------------|
| `solve()` | `[state_dim]` | `[horizon, control_dim]` | `[B, state_dim]` | `[B, horizon, control_dim]` |
| `step()` | `[state_dim]` | `[control_dim]` | `[B, state_dim]` | `[B, control_dim]` |
| `rollout()` | `[state_dim]` | `[horizon+1, state_dim]` | `[B, state_dim]` | `[B, horizon+1, state_dim]` |

### Applications and Use Cases

#### 1. Monte Carlo Simulation
```python
# Generate multiple random initial conditions
num_scenarios = 100
initial_states = torch.randn(num_scenarios, state_dim)

# Solve all scenarios in parallel
optimal_controls = controller.solve(initial_states, num_iterations=20)

# Analyze statistical properties
costs = []
for i in range(num_scenarios):
    trajectory = controller.rollout(initial_states[i], optimal_controls[i])
    costs.append(evaluate_trajectory_cost(trajectory))

mean_cost = torch.mean(torch.tensor(costs))
std_cost = torch.std(torch.tensor(costs))
```

#### 2. Multi-Agent Planning
```python
# Plan for multiple robots simultaneously
robot_states = torch.tensor([
    [0.0, 0.0, 0.0],  # Robot 1 at origin
    [5.0, 5.0, 0.0],  # Robot 2 at (5,5)
    [2.0, 8.0, 1.57], # Robot 3 at (2,8) facing north
])

# Get control sequences for all robots
all_controls = controller.solve(robot_states)

# Execute first control action for each robot
current_actions = controller.step(robot_states)
for i, action in enumerate(current_actions):
    robots[i].execute_control(action)
```

#### 3. Robust Control Design
```python
# Test robustness across parameter variations
nominal_params = {'mass': 1.0, 'length': 0.5}
param_variations = [
    {'mass': 0.8, 'length': 0.5},  # Lighter mass
    {'mass': 1.2, 'length': 0.5},  # Heavier mass  
    {'mass': 1.0, 'length': 0.4},  # Shorter length
    {'mass': 1.0, 'length': 0.6},  # Longer length
]

# Create dynamics for each parameter set
def create_batch_dynamics(param_list):
    def batch_dynamics(state, control):
        # Apply different parameters to different batch elements
        results = []
        for i, params in enumerate(param_list):
            # Extract state and control for this parameter set
            s_i = state[i:i+1]  # Keep batch dimension
            u_i = control[i:i+1]
            
            # Apply dynamics with specific parameters
            result_i = single_dynamics(s_i, u_i, **params)
            results.append(result_i)
        
        return torch.cat(results, dim=0)
    return batch_dynamics

# Test performance across all parameter variations
batch_dynamics = create_batch_dynamics([nominal_params] + param_variations)
controller_robust = DiffMPPI(dynamics_fn=batch_dynamics, ...)

initial_state_batch = torch.tensor([[1.0, 0.0]] * len(param_variations))
robust_controls = controller_robust.solve(initial_state_batch)
```

#### 4. Hyperparameter Tuning
```python
# Test different controller configurations in parallel
configs = [
    {'temperature': 0.1, 'num_samples': 100},
    {'temperature': 0.5, 'num_samples': 100}, 
    {'temperature': 1.0, 'num_samples': 100},
    {'temperature': 0.1, 'num_samples': 200},
    {'temperature': 0.5, 'num_samples': 200},
]

results = []
for config in configs:
    controller_test = DiffMPPI(**config, ...)
    
    # Test on batch of initial states
    test_states = torch.randn(20, state_dim)
    controls = controller_test.solve(test_states)
    
    # Evaluate performance
    performance = evaluate_controls(test_states, controls)
    results.append((config, performance))

# Find best configuration
best_config = min(results, key=lambda x: x[1])
```

#### 5. Imitation Learning
```python
# Process multiple demonstration trajectories
demo_initial_states = torch.stack([demo['initial_state'] for demo in demonstrations])
demo_controls = torch.stack([demo['controls'] for demo in demonstrations])

# Learn from demonstrations by minimizing imitation loss
def imitation_loss(predicted_controls, demo_controls):
    return torch.mean((predicted_controls - demo_controls)**2)

# Train with batch processing
optimizer = torch.optim.Adam(dynamics_model.parameters())

for epoch in range(num_epochs):
    # Generate controls for all demonstrations
    predicted_controls = controller.solve(demo_initial_states)
    
    # Compute loss across all demonstrations
    loss = imitation_loss(predicted_controls, demo_controls)
    
    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Performance Benefits

Batch processing provides significant computational speedups:

| Batch Size | Sequential Time | Batch Time | Speedup | Memory Usage |
|------------|----------------|------------|---------|--------------|
| 1 | 1.0x | 1.0x | 1.0x | 100% |
| 4 | 4.0x | 1.4x | 2.9x | 320% |
| 8 | 8.0x | 2.3x | 3.5x | 610% |
| 16 | 16.0x | 4.2x | 3.8x | 1180% |
| 32 | 32.0x | 8.7x | 3.7x | 2300% |

**Note:** Speedups measured on NVIDIA RTX 3080, actual performance varies by hardware and problem size.

### Memory Considerations

```python
# Monitor memory usage for large batches
def safe_batch_solve(controller, initial_states, max_batch_size=16):
    """Solve in chunks to avoid memory overflow."""
    batch_size = initial_states.shape[0]
    results = []
    
    for i in range(0, batch_size, max_batch_size):
        end_idx = min(i + max_batch_size, batch_size)
        batch = initial_states[i:end_idx]
        
        # Process chunk
        chunk_result = controller.solve(batch)
        results.append(chunk_result)
        
        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return torch.cat(results, dim=0)

# Usage for very large batches
large_initial_states = torch.randn(1000, state_dim)
all_controls = safe_batch_solve(controller, large_initial_states)
```

### Implementation Notes

1. **Function compatibility**: Dynamics and cost functions must handle batch dimensions properly
2. **Device consistency**: All batch elements must be on the same device
3. **Gradient flow**: Batch processing preserves gradients for end-to-end learning
4. **Memory scaling**: Memory usage scales linearly with batch size
5. **Synchronization**: All elements in a batch are processed for the same number of iterations
