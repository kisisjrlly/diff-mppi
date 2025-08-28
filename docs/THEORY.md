# Theoretical Background and Implementation

## Mathematical Foundation

### 1. Optimal Control Problem

The path integral approach solves the stochastic optimal control problem:

```
minimize E[∫₀ᵀ q(x(t), u(t))dt + φ(x(T))]
subject to: dx = f(x, u)dt + σdW
```

Where:
- `x(t)` is the state trajectory
- `u(t)` is the control trajectory  
- `q(x, u)` is the instantaneous cost
- `φ(x(T))` is the terminal cost
- `f(x, u)` is the deterministic dynamics
- `σdW` is the stochastic noise term

### 2. Hamilton-Jacobi-Bellman Equation

The optimal value function V(x,t) satisfies the HJB equation:

```
-∂V/∂t = min_u [q(x,u) + (∇V)ᵀf(x,u) + (1/2)σ²∇²V]
```

### 3. Path Integral Transformation

Using the Feynman-Kac theorem, the HJB equation transforms into a path integral:

```
V(x,t) = -λ log ∫ exp(-S(τ)/λ) Dτ
```

Where:
- `S(τ)` is the action functional along path τ
- `λ` is the temperature parameter
- The integral is over all possible trajectories τ

### 4. Action Functional

For the control-affine case with noise only in control:

```
S(τ) = φ(x(T)) + ∫₀ᵀ [q(x,u) + (1/2)uᵀR⁻¹u] dt
```

Where R⁻¹ relates to the noise covariance.

## Path Integral Networks (PI-Net)

### Core Algorithm

PI-Net discretizes and implements the path integral as a neural network module:

#### 1. Sample Generation
```python
# Generate K control perturbations
ε^(k) ~ N(0, Σ) for k = 1, ..., K
u^(k) = u₀ + ε^(k)
```

#### 2. Trajectory Rollout
```python
# Forward simulate each perturbed trajectory
for k in range(K):
    x₀^(k) = x₀
    for t in range(H):
        x_{t+1}^(k) = f(x_t^(k), u_t^(k))
```

#### 3. Cost Evaluation
```python
# Compute trajectory costs
S^(k) = Σₜ q(x_t^(k), u_t^(k)) + φ(x_H^(k))
```

#### 4. Importance Weighting
```python
# Softmax weighting based on costs
w^(k) = exp(-S^(k)/λ) / Σⱼ exp(-S^(j)/λ)
```

#### 5. Control Update
```python
# Weighted average of perturbations
u* = u₀ + Σₖ w^(k) ε^(k)
```

### Implementation Details

#### Sampling Strategy

The library uses Gaussian sampling for control perturbations:

```python
def sample_noise(self, control_sequence):
    """Sample control perturbations."""
    noise = torch.randn(
        self.num_samples, 
        self.horizon, 
        self.control_dim,
        device=self.device
    )
    return noise * self.noise_std
```

**Key considerations:**
- Noise covariance affects exploration vs exploitation
- Per-dimension scaling for different control units
- Adaptive noise scheduling for convergence

#### Trajectory Simulation

Efficient batched rollouts using vectorized operations:

```python
def rollout_trajectories(self, initial_state, control_sequences):
    """Simulate multiple trajectories in parallel."""
    batch_size = control_sequences.shape[0]
    states = torch.zeros(batch_size, self.horizon + 1, self.state_dim)
    states[:, 0] = initial_state
    
    for t in range(self.horizon):
        states[:, t + 1] = self.dynamics_fn(
            states[:, t], 
            control_sequences[:, t]
        )
    
    return states
```

**Optimization techniques:**
- Vectorized dynamics evaluation
- Memory-efficient state storage
- GPU acceleration for large batch sizes

#### Cost Computation

Total trajectory cost accumulation:

```python
def compute_trajectory_costs(self, states, controls):
    """Compute total cost for each trajectory."""
    costs = torch.zeros(states.shape[0], device=self.device)
    
    # Accumulate stage costs
    for t in range(self.horizon):
        stage_costs = self.cost_fn(states[:, t], controls[:, t])
        costs += stage_costs
    
    # Add terminal cost if specified
    if self.terminal_cost_fn:
        terminal_costs = self.terminal_cost_fn(states[:, -1])
        costs += terminal_costs
    
    return costs
```

#### Importance Sampling

Temperature-scaled softmax for trajectory weighting:

```python
def compute_weights(self, costs):
    """Compute importance weights from trajectory costs."""
    # Numerical stability: subtract minimum cost
    costs_normalized = costs - torch.min(costs)
    
    # Temperature scaling and softmax
    exp_costs = torch.exp(-costs_normalized / self.temperature)
    weights = exp_costs / torch.sum(exp_costs)
    
    return weights
```

**Numerical considerations:**
- Overflow prevention via cost normalization
- Temperature scheduling for annealing
- Weight clipping for stability

## Accelerated Methods

### Standard Gradient Descent

The basic MPPI update can be viewed as:

```
u_{k+1} = u_k + α∇J(u_k)
```

Where the gradient is approximated via importance sampling.

### Nesterov Accelerated Gradient (NAG)

NAG uses momentum with look-ahead:

```python
def nag_update(self, control_seq, gradient):
    """Nesterov accelerated gradient update."""
    # Look-ahead step
    lookahead_control = control_seq + self.momentum * self.velocity
    
    # Compute gradient at look-ahead point
    lookahead_gradient = self.compute_gradient(lookahead_control)
    
    # Update velocity and control
    self.velocity = self.momentum * self.velocity + self.lr * lookahead_gradient
    new_control = control_seq + self.velocity
    
    return new_control
```

**Mathematical formulation:**
```
v_{k+1} = μv_k + α∇J(u_k + μv_k)
u_{k+1} = u_k + v_{k+1}
```

### Adam Optimizer

Adam combines momentum with adaptive learning rates:

```python
def adam_update(self, control_seq, gradient):
    """Adam optimizer update."""
    # Update biased moments
    self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
    self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
    
    # Bias correction
    m_hat = self.m / (1 - self.beta1**self.t)
    v_hat = self.v / (1 - self.beta2**self.t)
    
    # Update control
    new_control = control_seq + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    return new_control
```

**Mathematical formulation:**
```
m_k = β₁m_{k-1} + (1-β₁)∇J(u_k)
v_k = β₂v_{k-1} + (1-β₂)[∇J(u_k)]²
û_m = m_k/(1-β₁ᵏ), û_v = v_k/(1-β₂ᵏ)
u_{k+1} = u_k + α·û_m/(√û_v + ε)
```

### RMSprop

Root mean square propagation for adaptive learning:

```python
def rmsprop_update(self, control_seq, gradient):
    """RMSprop optimizer update."""
    # Update moving average of squared gradients
    self.v = self.alpha * self.v + (1 - self.alpha) * gradient**2
    
    # Update control with adaptive learning rate
    new_control = control_seq + self.lr * gradient / (torch.sqrt(self.v) + self.eps)
    
    return new_control
```

## Convergence Analysis

### Theoretical Guarantees

**Standard MPPI:**
- Converges to local optimum under mild conditions
- Convergence rate: O(1/k) for general non-convex problems
- Global convergence for convex costs with sufficient exploration

**Accelerated Methods:**
- NAG: O(1/k²) convergence for smooth convex problems
- Adam: Adaptive convergence, good empirical performance
- RMSprop: Robust to ill-conditioned problems

### Practical Considerations

#### Sample Size Effects
```
Bias = O(1/√K)    # Monte Carlo error
Variance = O(1/K)  # Sample variance
```

#### Temperature Scheduling
```python
def adaptive_temperature(self, iteration, max_iterations):
    """Adaptive temperature annealing."""
    # Exponential decay
    return self.temp_initial * (self.temp_final / self.temp_initial)**(iteration / max_iterations)
```

#### Convergence Criteria
```python
def check_convergence(self, cost_history, tolerance=1e-4, window=5):
    """Check if optimization has converged."""
    if len(cost_history) < window:
        return False
    
    recent_costs = cost_history[-window:]
    relative_change = (max(recent_costs) - min(recent_costs)) / abs(recent_costs[0])
    
    return relative_change < tolerance
```

## Computational Complexity

### Time Complexity

**Per iteration:**
- Sampling: O(H × K × D)
- Rollouts: O(H × K × dynamics_cost)
- Cost evaluation: O(H × K × cost_cost)
- Weight computation: O(K)
- Update: O(H × D)

**Total:** O(H × K × (D + dynamics_cost + cost_cost))

Where:
- H: horizon length
- K: number of samples
- D: control dimension

### Space Complexity

**Memory requirements:**
- Control sequences: O(H × K × D)
- State trajectories: O(H × K × state_dim)
- Gradients/moments: O(H × D) per optimizer

**Total:** O(H × K × max(D, state_dim))

### Scalability Analysis

**Scaling with problem size:**

| Parameter | Effect on Computation | Effect on Memory |
|-----------|----------------------|------------------|
| Horizon (H) | Linear | Linear |
| Samples (K) | Linear | Linear |
| State dim | Depends on dynamics | Linear |
| Control dim | Linear | Linear |

**GPU Acceleration Benefits:**
- Parallel trajectory simulation: ~10-100x speedup
- Vectorized operations: Eliminates Python loops
- Memory bandwidth: Efficient for large K

## Implementation Optimizations

### Memory Management

```python
def optimize_memory_usage(self):
    """Memory optimization strategies."""
    # Preallocate tensors
    self.trajectory_buffer = torch.zeros(
        self.num_samples, self.horizon + 1, self.state_dim,
        device=self.device
    )
    
    # Reuse control perturbation buffer
    self.noise_buffer = torch.zeros(
        self.num_samples, self.horizon, self.control_dim,
        device=self.device
    )
    
    # In-place operations where possible
    torch.randn(self.noise_buffer.shape, out=self.noise_buffer)
```

### Numerical Stability

```python
def ensure_numerical_stability(self, costs):
    """Prevent numerical issues in weight computation."""
    # Clip extreme costs
    costs = torch.clamp(costs, max=self.max_cost_value)
    
    # Subtract minimum for overflow prevention
    costs_centered = costs - torch.min(costs)
    
    # Use log-sum-exp trick if needed
    if self.temperature < 1e-6:
        weights = self.log_sum_exp_weights(costs_centered)
    else:
        weights = torch.softmax(-costs_centered / self.temperature, dim=0)
    
    return weights
```

### Performance Profiling

Key bottlenecks and optimization strategies:

1. **Dynamics Evaluation**: Usually dominates runtime
   - Vectorize dynamics computation
   - Use compiled models (TorchScript)
   - Consider approximate dynamics for real-time control

2. **Memory Bandwidth**: Important for GPU performance
   - Minimize device transfers
   - Use appropriate tensor layouts
   - Batch operations efficiently

3. **Python Overhead**: Minimize pure Python loops
   - Vectorize all operations
   - Use torch.jit.script for critical paths
   - Consider C++ extensions for ultra-performance

## Advanced Topics

### Constrained Optimization

Handling control and state constraints:

```python
def handle_constraints(self, control_seq):
    """Apply control and state constraints."""
    # Control box constraints
    if self.control_bounds is not None:
        control_seq = torch.clamp(
            control_seq, 
            self.control_bounds[0], 
            self.control_bounds[1]
        )
    
    # State constraints via penalty method
    trajectory = self.rollout(self.current_state, control_seq)
    constraint_violations = self.compute_constraint_violations(trajectory)
    constraint_penalty = self.penalty_weight * constraint_violations
    
    return control_seq, constraint_penalty
```

### Stochastic Dynamics

Handling process noise in dynamics:

```python
def stochastic_dynamics(self, state, control):
    """Dynamics with process noise."""
    # Deterministic part
    next_state_mean = self.deterministic_dynamics(state, control)
    
    # Add process noise
    process_noise = self.process_noise_std * torch.randn_like(next_state_mean)
    next_state = next_state_mean + process_noise
    
    return next_state
```

### Learning Applications

Using MPPI for learning cost functions or dynamics:

```python
class LearnableMPPI:
    def __init__(self, learned_dynamics, learned_cost):
        self.dynamics_model = learned_dynamics  # Neural network
        self.cost_model = learned_cost          # Neural network
        
    def update_models(self, expert_trajectories):
        """Update models based on expert demonstrations."""
        # Inverse optimal control
        self.cost_model.train()
        for trajectory in expert_trajectories:
            predicted_cost = self.cost_model(trajectory)
            loss = self.compute_imitation_loss(predicted_cost, trajectory)
            loss.backward()
            self.cost_optimizer.step()
```

This theoretical foundation provides the mathematical and computational basis for understanding and extending the diff-mppi library.
