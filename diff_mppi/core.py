"""
Differentiable Model Predictive Path Integral (Diff-MPPI) Control

A PyTorch-based implementation of MPPI with gradient-based acceleration methods.
This module provides a unified, clean interface for MPPI with various optimization 
enhancements including Adam, NAG, and AdaGrad acceleration.

Based on:
- "Path Integral Networks: End-to-End Differentiable Optimal Control" (Okada et al., 2017)
- "Acceleration of Gradient-Based Path Integral Method" (Okada & Taniguchi, 2018)
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional, Dict, Any, Tuple


class DiffMPPI:
    """
    Differentiable Model Predictive Path Integral Controller.
    
    This class implements MPPI with optional gradient-based acceleration methods
    for improved convergence and performance in optimal control problems.
    """
    
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
    ):
        """
        Initialize Diff-MPPI controller.
        
        Args:
            state_dim: Dimension of state space
            control_dim: Dimension of control space  
            dynamics_fn: Function f(state, control) -> next_state
            cost_fn: Function g(state, control) -> cost
            terminal_cost_fn: Optional terminal cost function φ(state) -> cost
            horizon: Planning horizon length
            num_samples: Number of trajectory samples
            temperature: Temperature parameter for sampling
            control_bounds: Optional (min_control, max_control) bounds
            acceleration: Acceleration method ('adam', 'nag', 'adagrad', None)
            lr: Base learning rate for acceleration methods
            momentum: Momentum parameter (legacy, kept for compatibility)
            eps: Epsilon parameter for Adam/AdaGrad numerical stability
            weight_decay: Weight decay for regularization
            device: Device for computation ('cpu' or 'cuda')
            nag_gamma: NAG momentum decay coefficient (paper default: 0.8)
            adam_beta1: Adam first moment decay rate (paper default: 0.9)
            adam_beta2: Adam second moment decay rate (paper default: 0.999)
            adam_lr: Adam learning rate (paper default: 1e-3)
            adagrad_eta0: AdaGrad initial step size (default: uses lr)
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.device = device
        
        # Control bounds
        if control_bounds is not None:
            self.control_min = control_bounds[0].to(device)
            self.control_max = control_bounds[1].to(device)
        else:
            self.control_min = None
            self.control_max = None
        
        # Initialize control sequence
        self.control_sequence = torch.zeros(
            horizon, control_dim, device=device, requires_grad=True
        )
        
        # Acceleration settings
        self.acceleration = acceleration
        self.lr = lr
        self.momentum = momentum  # Legacy parameter
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Algorithm-specific parameters with paper defaults
        self.nag_gamma = nag_gamma
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_lr = adam_lr
        self.adagrad_eta0 = adagrad_eta0 if adagrad_eta0 is not None else lr
        
        # Initialize acceleration state
        self._init_acceleration()
        
    def _init_acceleration(self):
        """Initialize acceleration-specific state variables."""
        if self.acceleration == "adam":
            # Adam parameters - configurable with paper defaults
            self.adam_m = torch.zeros_like(self.control_sequence)
            self.adam_v = torch.zeros_like(self.control_sequence)
            self.adam_t = 0
        elif self.acceleration == "nag":
            # NAG: need to store previous update for momentum
            self.nag_prev_update = torch.zeros_like(self.control_sequence)
        elif self.acceleration == "adagrad":
            # AdaGrad: cumulative squared gradients
            self.adagrad_G = torch.zeros_like(self.control_sequence)
        
    def solve(
        self, 
        initial_state: torch.Tensor, 
        num_iterations: int = 10,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Solve optimal control problem using Diff-MPPI.
        
        Args:
            initial_state: Initial state [state_dim] or [batch_size, state_dim]
            num_iterations: Number of MPPI iterations
            verbose: Print convergence information
            
        Returns:
            Optimal control sequence [horizon, control_dim] or [batch_size, horizon, control_dim]
        """
        # Handle both single state and batch of states
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
            single_state = True
        else:
            single_state = False
            
        batch_size = initial_state.shape[0]
        
        # Initialize control sequences for each state in batch
        if not hasattr(self, 'batch_control_sequences') or self.batch_control_sequences.shape[0] != batch_size:
            self.batch_control_sequences = torch.zeros(
                batch_size, self.horizon, self.control_dim, 
                device=self.device, requires_grad=True
            )
            # Initialize acceleration state for batch
            self._init_batch_acceleration(batch_size)
        
        best_costs = torch.full((batch_size,), float('inf'), device=self.device)
        
        for iteration in range(num_iterations):
            # Sample control perturbations
            noise = torch.randn(
                batch_size, self.num_samples, self.horizon, self.control_dim, 
                device=self.device
            )
            
            # Generate candidate control sequences - NAG requires momentum drift in sampling
            if self.acceleration == "nag" and hasattr(self, 'batch_nag_prev_update'):
                # NAG: Apply momentum drift to sampling distribution (Equation 18)
                # E[u] = μ^(j-1) + γ·Δμ^(j-2)
                momentum_drift = self.nag_gamma * self.batch_nag_prev_update
                candidate_controls = (self.batch_control_sequences.unsqueeze(1) + 
                                    momentum_drift.unsqueeze(1) + noise)
            else:
                # Standard sampling for other methods
                candidate_controls = self.batch_control_sequences.unsqueeze(1) + noise
            
            # Apply control bounds if specified
            if self.control_min is not None and self.control_max is not None:
                candidate_controls = torch.clamp(
                    candidate_controls, self.control_min, self.control_max
                )
            
            # Evaluate trajectories for all batch elements
            costs = self._evaluate_trajectories_batch(initial_state, candidate_controls)
            
            # Compute weights using softmax for each batch element
            weights = F.softmax(-costs / self.temperature, dim=1)
            
            # Update control sequences for each batch element
            if self.acceleration is None:
                # Standard MPPI update - Algorithm Line 15: u_t,i ← u_t,i + weighted_perturbations
                weighted_perturbations = torch.sum(
                    weights.unsqueeze(-1).unsqueeze(-1) * noise, dim=1
                )
                self.batch_control_sequences = self.batch_control_sequences + weighted_perturbations
            else:
                # Gradient-based update
                gradients = torch.sum(
                    weights.unsqueeze(-1).unsqueeze(-1) * noise, dim=1
                )
                self._apply_batch_acceleration(gradients)
            
            # Track best costs
            current_costs = torch.min(costs, dim=1)[0]
            better_mask = current_costs < best_costs
            best_costs[better_mask] = current_costs[better_mask]
                
            if verbose and iteration % 5 == 0:
                avg_cost = torch.mean(current_costs).item()
                print(f"Iteration {iteration}: Avg Cost = {avg_cost:.6f}")
        
        if single_state:
            return self.batch_control_sequences[0].detach()
        else:
            return self.batch_control_sequences.detach()
    
    def _evaluate_trajectories(
        self, 
        initial_state: torch.Tensor, 
        control_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate cost for multiple control sequences.
        
        Args:
            initial_state: Initial state [1, state_dim]
            control_sequences: Control sequences [num_samples, horizon, control_dim]
            
        Returns:
            Costs for each sequence [num_samples]
        """
        batch_size = control_sequences.shape[0]
        
        # Initialize states
        states = initial_state.repeat(batch_size, 1)
        total_costs = torch.zeros(batch_size, device=self.device)
        
        # Rollout trajectories
        for t in range(self.horizon):
            controls = control_sequences[:, t, :]
            
            # Compute costs
            step_costs = self.cost_fn(states, controls)
            total_costs += step_costs
            
            # Update states
            states = self.dynamics_fn(states, controls)
            
        return total_costs
    
    def _evaluate_trajectories_batch(
        self, 
        initial_states: torch.Tensor, 
        control_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate cost for multiple control sequences across multiple initial states.
        
        Args:
            initial_states: Initial states [batch_size, state_dim]
            control_sequences: Control sequences [batch_size, num_samples, horizon, control_dim]
            
        Returns:
            Costs for each sequence [batch_size, num_samples]
        """
        batch_size, num_samples = control_sequences.shape[:2]
        
        # Flatten for parallel processing: [batch_size * num_samples, horizon, control_dim]
        flat_controls = control_sequences.view(batch_size * num_samples, self.horizon, self.control_dim)
        
        # Repeat initial states: [batch_size * num_samples, state_dim]
        flat_initial_states = initial_states.repeat_interleave(num_samples, dim=0)
        
        # Initialize states
        states = flat_initial_states
        total_costs = torch.zeros(batch_size * num_samples, device=self.device)
        
        # Rollout trajectories
        for t in range(self.horizon):
            controls = flat_controls[:, t, :]
            
            # Compute costs
            step_costs = self.cost_fn(states, controls)
            total_costs += step_costs
            
            # Update states
            states = self.dynamics_fn(states, controls)
        
        # Add terminal cost if provided (Algorithm Line 7: q_T,i^(k) ← φ(x_T,N^(k)))
        if self.terminal_cost_fn is not None:
            terminal_costs = self.terminal_cost_fn(states)
            total_costs += terminal_costs
        
        # Reshape back to [batch_size, num_samples]
        return total_costs.view(batch_size, num_samples)
    
    def _init_batch_acceleration(self, batch_size: int):
        """Initialize acceleration-specific state variables for batch processing."""
        if self.acceleration == "adam":
            self.batch_adam_m = torch.zeros(batch_size, self.horizon, self.control_dim, device=self.device)
            self.batch_adam_v = torch.zeros(batch_size, self.horizon, self.control_dim, device=self.device)
            self.batch_adam_t = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        elif self.acceleration == "nag":
            self.batch_nag_prev_update = torch.zeros(batch_size, self.horizon, self.control_dim, device=self.device)
        elif self.acceleration == "adagrad":
            self.batch_adagrad_G = torch.zeros(batch_size, self.horizon, self.control_dim, device=self.device)
    
    def _apply_batch_acceleration(self, gradients: torch.Tensor):
        """Apply gradient-based acceleration update for batch processing."""
        batch_size = gradients.shape[0]
        
        if self.acceleration == "adam":
            # Adam algorithm following paper specifications
            self.batch_adam_t += 1
            
            # Update biased first and second moments (Equation from paper)
            self.batch_adam_m = (self.adam_beta1 * self.batch_adam_m + 
                               (1 - self.adam_beta1) * gradients)
            self.batch_adam_v = (self.adam_beta2 * self.batch_adam_v + 
                               (1 - self.adam_beta2) * gradients**2)
            
            # Bias correction and parameter update
            updates = torch.zeros_like(self.batch_control_sequences)
            for i in range(batch_size):
                t_i = self.batch_adam_t[i].item()
                m_hat = self.batch_adam_m[i] / (1 - self.adam_beta1**t_i)
                v_hat = self.batch_adam_v[i] / (1 - self.adam_beta2**t_i)
                updates[i] = self.adam_lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                
        elif self.acceleration == "nag":
            # NAG following paper Algorithm 2 and Equation 19
            # Δμ^(j-1) = γ·Δμ^(j-2) + δμ^(j-1)
            # where δμ^(j-1) is the gradient from momentum-drifted sampling
            delta_mu = self.nag_gamma * self.batch_nag_prev_update + gradients
            updates = delta_mu
            
            # Store this update for next iteration's momentum
            self.batch_nag_prev_update = delta_mu.clone()
            
        elif self.acceleration == "adagrad":
            # AdaGrad following paper Equation 20
            # G^(j-1) = G^(j-2) + (Δμ^(j-1))²
            self.batch_adagrad_G += gradients**2
            
            # η^(j-1) = η₀ / √(G^(j-1) + ε)
            adaptive_lr = self.adagrad_eta0 / (torch.sqrt(self.batch_adagrad_G) + self.eps)
            
            # μ^(j) = μ^(j-1) + η^(j-1) ⊙ Δμ^(j-1) (element-wise multiplication)
            updates = adaptive_lr * gradients
            
        else:
            # Fallback: simple gradient update
            updates = self.lr * gradients
        
        # Apply weight decay if specified
        if self.weight_decay > 0:
            updates += self.weight_decay * self.batch_control_sequences
            
        # Update control sequences
        self.batch_control_sequences = self.batch_control_sequences + updates
        
        # Apply bounds if specified
        if self.control_min is not None and self.control_max is not None:
            self.batch_control_sequences = torch.clamp(
                self.batch_control_sequences, self.control_min, self.control_max
            )
    
    def _apply_acceleration(self, gradient: torch.Tensor):
        """Apply gradient-based acceleration update."""
        
        if self.acceleration == "adam":
            # Adam algorithm following paper specifications
            self.adam_t += 1
            
            # Update biased first and second moments
            self.adam_m = (self.adam_beta1 * self.adam_m + 
                          (1 - self.adam_beta1) * gradient)
            self.adam_v = (self.adam_beta2 * self.adam_v + 
                          (1 - self.adam_beta2) * gradient**2)
            
            # Bias correction
            m_hat = self.adam_m / (1 - self.adam_beta1**self.adam_t)
            v_hat = self.adam_v / (1 - self.adam_beta2**self.adam_t)
            
            # Update parameters
            update = self.adam_lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            
        elif self.acceleration == "nag":
            # NAG following paper Algorithm 2 and Equation 19
            # Δμ^(j-1) = γ·Δμ^(j-2) + δμ^(j-1)
            delta_mu = self.nag_gamma * self.nag_prev_update + gradient
            update = delta_mu
            
            # Store this update for next iteration
            self.nag_prev_update = delta_mu.clone()
            
        elif self.acceleration == "adagrad":
            # AdaGrad following paper Equation 20
            # G^(j-1) = G^(j-2) + (Δμ^(j-1))²
            self.adagrad_G += gradient**2
            
            # η^(j-1) = η₀ / √(G^(j-1) + ε)
            adaptive_lr = self.adagrad_eta0 / (torch.sqrt(self.adagrad_G) + self.eps)
            
            # μ^(j) = μ^(j-1) + η^(j-1) ⊙ Δμ^(j-1) (element-wise multiplication)
            update = adaptive_lr * gradient
            
        else:
            # Fallback: simple gradient update
            update = self.lr * gradient
        
        # Apply weight decay if specified
        if self.weight_decay > 0:
            update += self.weight_decay * self.control_sequence
            
        # Update control sequence
        self.control_sequence = self.control_sequence + update
        
        # Apply bounds if specified
        if self.control_min is not None and self.control_max is not None:
            self.control_sequence = torch.clamp(
                self.control_sequence, self.control_min, self.control_max
            )
    
    def rollout(
        self, 
        initial_state: torch.Tensor, 
        control_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Rollout a trajectory given initial state(s) and control sequence(s).
        
        Args:
            initial_state: Initial state(s) [state_dim] or [batch_size, state_dim]
            control_sequence: Control sequence(s) [horizon, control_dim] or 
                            [batch_size, horizon, control_dim].
                            If None, uses current control sequence.
                            
        Returns:
            State trajectory [horizon+1, state_dim] or [batch_size, horizon+1, state_dim]
        """
        if control_sequence is None:
            control_sequence = self.control_sequence
            
        is_batch = initial_state.dim() > 1
        
        if is_batch:
            # Batch processing
            batch_size = initial_state.shape[0]
            
            if control_sequence.dim() == 2:
                # Broadcast single control sequence to all batch elements
                control_sequence = control_sequence.unsqueeze(0).repeat(batch_size, 1, 1)
            
            trajectory = torch.zeros(batch_size, self.horizon + 1, self.state_dim, device=self.device)
            trajectory[:, 0, :] = initial_state
            
            for t in range(self.horizon):
                controls = control_sequence[:, t, :]
                next_states = self.dynamics_fn(trajectory[:, t, :], controls)
                trajectory[:, t + 1, :] = next_states
                
            return trajectory
        else:
            # Single state processing
            if initial_state.dim() == 1:
                initial_state = initial_state.unsqueeze(0)
                
            trajectory = [initial_state.squeeze(0)]
            state = initial_state
            
            for t in range(self.horizon):
                control = control_sequence[t:t+1]
                state = self.dynamics_fn(state, control)
                trajectory.append(state.squeeze(0))
                
            return torch.stack(trajectory)
    
    def step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get next control action(s) for current state(s) (MPC-style).
        
        Args:
            state: Current state(s) [state_dim] or [batch_size, state_dim]
            
        Returns:
            Control action(s) [control_dim] or [batch_size, control_dim]
        """
        is_batch = state.dim() > 1
        
        if is_batch:
            # Batch processing
            control_sequences = self.solve(state, num_iterations=5)
            return control_sequences[:, 0, :].detach()  # First control for each batch
        else:
            # Single state processing
            self.solve(state, num_iterations=5)
            return self.control_sequence[0].detach()
    
    def reset(self):
        """Reset controller state."""
        self.control_sequence = torch.zeros(
            self.horizon, self.control_dim, device=self.device, requires_grad=True
        )
        self._init_acceleration()
        
        # Reset batch variables if they exist
        if hasattr(self, 'batch_control_sequences'):
            del self.batch_control_sequences
        if hasattr(self, 'batch_adam_m'):
            del self.batch_adam_m
        if hasattr(self, 'batch_adam_v'): 
            del self.batch_adam_v
        if hasattr(self, 'batch_adam_t'):
            del self.batch_adam_t
        if hasattr(self, 'batch_nag_prev_update'):
            del self.batch_nag_prev_update
        if hasattr(self, 'batch_adagrad_G'):
            del self.batch_adagrad_G


def create_mppi_controller(
    state_dim: int,
    control_dim: int, 
    dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    **kwargs
) -> DiffMPPI:
    """
    Factory function to create a Diff-MPPI controller.
    
    This is a convenience function that provides a clean interface for creating
    MPPI controllers with various configurations.
    
    Args:
        state_dim: Dimension of state space
        control_dim: Dimension of control space
        dynamics_fn: Dynamics function f(state, control) -> next_state
        cost_fn: Cost function g(state, control) -> cost
        **kwargs: Additional arguments passed to DiffMPPI constructor
        
    Returns:
        Configured DiffMPPI controller
        
    Example:
        >>> def dynamics(state, control):
        ...     return state + 0.1 * control
        >>> 
        >>> def cost(state, control):
        ...     return torch.sum(state**2) + torch.sum(control**2)
        >>>
        >>> controller = create_mppi_controller(
        ...     state_dim=2,
        ...     control_dim=1,
        ...     dynamics_fn=dynamics,
        ...     cost_fn=cost,
        ...     horizon=20,
        ...     num_samples=100,
        ...     acceleration="adam"
        ... )
    """
    return DiffMPPI(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics_fn=dynamics_fn,
        cost_fn=cost_fn,
        **kwargs
    )