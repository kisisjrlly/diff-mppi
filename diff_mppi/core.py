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
        lr: float = 1e-3,
        momentum: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        device: str = "cpu",
        # NAG-specific parameters (paper defaults)
        nag_gamma: float = 0.8,
        # Adam-specific parameters (paper defaults)
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
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
            lr: Learning rate, used for Adam (paper default: 1e-3) and other accelerations
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
        self.adam_lr = lr  # Use lr parameter for Adam learning rate
        self.adagrad_eta0 = adagrad_eta0 if adagrad_eta0 is not None else lr
        
    def solve(
        self, 
        initial_state: torch.Tensor, 
        num_iterations: int = 10,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Solve optimal control problem using Diff-MPPI.
        
        Args:
            initial_state: Initial state [batch_size, state_dim]
            num_iterations: Number of MPPI iterations
            verbose: Print convergence information
            
        Returns:
            Optimal control sequence [batch_size, horizon, control_dim]
        """
        # Ensure batch input
        if initial_state.dim() == 1:
            raise ValueError("Input must be batch mode: [batch_size, state_dim]. Got single state.")
            
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
        
        return self.batch_control_sequences.detach()
    
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
    
    def rollout(
        self, 
        initial_state: torch.Tensor, 
        control_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Rollout a trajectory given initial state(s) and control sequence(s).
        
        Args:
            initial_state: Initial states [batch_size, state_dim]
            control_sequence: Control sequences [batch_size, horizon, control_dim].
                            If None, uses current batch control sequences.
                            
        Returns:
            State trajectory [batch_size, horizon+1, state_dim]
        """
        # Ensure batch input
        if initial_state.dim() == 1:
            raise ValueError("Input must be batch mode: [batch_size, state_dim]. Got single state.")
            
        if control_sequence is None:
            if not hasattr(self, 'batch_control_sequences'):
                raise ValueError("No batch control sequences available. Call solve() first or provide control_sequence.")
            control_sequence = self.batch_control_sequences
            
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
    
    def step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get next control action(s) for current state(s) (MPC-style).
        
        This method implements Algorithm 3 from the paper when using NAG acceleration:
        1. Solve optimization problem for current state
        2. Return first control action
        3. Apply warm start shifting for next iteration (NAG only)
        
        Args:
            state: Current states [batch_size, state_dim]
            
        Returns:
            Control actions [batch_size, control_dim]
        """
        # Ensure batch input
        if state.dim() == 1:
            raise ValueError("Input must be batch mode: [batch_size, state_dim]. Got single state.")
        
        # Batch processing
        control_sequences = self.solve(state, num_iterations=5)
        first_control = control_sequences[:, 0, :].detach()
        
        # Algorithm 3, Steps 7-9: Apply warm start shifting for next MPC iteration
        # This is only applied for NAG acceleration as specified in the paper
        if self.acceleration == "nag":
            self.warm_start_shift(fill_method="replicate")
            
        return first_control
    
    def warm_start_shift(self, fill_method: str = "replicate"):
        """
        Perform Algorithm 3 warm start: shift control sequences and NAG momentum terms.
        
        This implements the key MPC warm start mechanism from the paper's Algorithm 3,
        which is specifically designed for NAG acceleration:
        - Step 7: Shift control sequence μ forward by one time step
        - Step 8: Shift NAG momentum terms Δμ forward by one time step  
        - Step 9: Initialize the last element using specified fill method
        
        Note: This method should only be called when using NAG acceleration.
        
        Args:
            fill_method: Method to fill the last time step
                - "replicate": Copy the second-to-last element (paper default)
                - "zero": Set to zero
                - "extrapolate": Linear extrapolation from last two elements
        """
        # Only apply warm start for NAG acceleration as specified in the paper
        if self.acceleration != "nag":
            return
            
        # Ensure batch control sequences exist
        if not hasattr(self, 'batch_control_sequences'):
            return
            
        # Algorithm 3, Step 7: Shift control sequence μ forward
        # μ_new[0:T-1] = μ_old[1:T]
        batch_size = self.batch_control_sequences.shape[0]
        
        # Store original sequences for reference before shifting
        original_controls = self.batch_control_sequences.clone()
        
        # Shift control sequences
        self.batch_control_sequences[:, :-1, :] = self.batch_control_sequences[:, 1:, :].clone()
        
        # Algorithm 3, Step 9: Initialize last element
        if fill_method == "replicate":
            # Copy second-to-last element from ORIGINAL sequence (paper default)
            self.batch_control_sequences[:, -1, :] = original_controls[:, -2, :].clone()
        elif fill_method == "zero":
            self.batch_control_sequences[:, -1, :] = 0.0
        elif fill_method == "extrapolate":
            # Linear extrapolation: u_T = 2*u_{T-1} - u_{T-2} (using original sequence)
            if self.horizon >= 2:
                extrapolated = (2.0 * original_controls[:, -2, :] - 
                              original_controls[:, -3, :])
                self.batch_control_sequences[:, -1, :] = extrapolated
            else:
                self.batch_control_sequences[:, -1, :] = original_controls[:, -2, :].clone()
        
        # Algorithm 3, Step 8: Shift NAG momentum terms Δμ forward  
        if hasattr(self, 'batch_nag_prev_update'):
            # Store original momentum for reference
            original_momentum = self.batch_nag_prev_update.clone()
            
            # Δμ_new[0:T-1] = Δμ_old[1:T]
            self.batch_nag_prev_update[:, :-1, :] = self.batch_nag_prev_update[:, 1:, :].clone()
            
            # Initialize last momentum element
            if fill_method == "replicate":
                self.batch_nag_prev_update[:, -1, :] = original_momentum[:, -2, :].clone()
            elif fill_method == "zero":
                self.batch_nag_prev_update[:, -1, :] = 0.0
            elif fill_method == "extrapolate":
                if self.horizon >= 2:
                    extrapolated = (2.0 * original_momentum[:, -2, :] - 
                                  original_momentum[:, -3, :])
                    self.batch_nag_prev_update[:, -1, :] = extrapolated
                else:
                    self.batch_nag_prev_update[:, -1, :] = original_momentum[:, -2, :].clone()
                    
        # Apply bounds if specified after shifting
        if self.control_min is not None and self.control_max is not None:
            self.batch_control_sequences = torch.clamp(
                self.batch_control_sequences, self.control_min, self.control_max
            )
    
    def reset(self, initial_states=None):
        """
        Reset the MPPI planner for batch processing mode only.
        
        Args:
            initial_states: Batch of initial states [batch_size, state_dim]
        """
        if initial_states is not None:
            if initial_states.dim() != 2:
                raise ValueError("Initial states must be 2D tensor [batch_size, state_dim]")
            batch_size = initial_states.shape[0]
        else:
            # If no initial states provided, use existing batch size or default
            if hasattr(self, 'batch_control_sequences'):
                batch_size = self.batch_control_sequences.shape[0]
            else:
                batch_size = 1  # Default batch size
                
        # Initialize batch control sequences
        self.batch_control_sequences = torch.zeros(
            batch_size, self.horizon, self.control_dim, device=self.device
        )
        
        # Initialize acceleration-specific parameters for batch mode
        if self.acceleration == "nag":
            self.batch_nag_prev_update = torch.zeros_like(self.batch_control_sequences)
        elif self.acceleration == "adam":
            self.batch_adam_m = torch.zeros_like(self.batch_control_sequences)
            self.batch_adam_v = torch.zeros_like(self.batch_control_sequences)
            self.batch_adam_t = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        elif self.acceleration == "adagrad":
            self.batch_adagrad_sum_squared = torch.zeros_like(self.batch_control_sequences)
            
        # Store current states if provided
        if initial_states is not None:
            self.current_states = initial_states


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