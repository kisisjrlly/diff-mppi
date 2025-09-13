"""
Latent Space Model Predictive Path Integral (LMPPI) Controller

This module implements the main LMPPI controller that performs online control
using a pre-trained VAE model. The controller samples in the low-dimensional
latent space rather than the high-dimensional control space, achieving
significant computational speedup while maintaining trajectory feasibility.

Key Features:
- Latent space sampling instead of control sequence sampling
- Integration with pre-trained VAE models
- Compatible with existing MPPI framework
- Supports warm starting and MPC-style control
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple
import warnings

from .models import TrajectoryVAE


class LMPPIController:
    """
    Latent Space Model Predictive Path Integral Controller.
    
    This controller replaces the high-dimensional control sequence sampling
    in standard MPPI with low-dimensional latent space sampling using a
    pre-trained VAE model.
    
    Workflow:
    1. Encode current/reference trajectory to latent space
    2. Sample noise in latent space (much lower dimensional)
    3. Decode samples to full trajectories (replaces forward integration)
    4. Evaluate trajectory costs
    5. Compute MPPI weights and update latent representation
    6. Decode optimal latent vector to get control action
    """
    
    def __init__(
        self,
        vae_model: TrajectoryVAE,
        state_dim: int,
        control_dim: int,
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        terminal_cost_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        horizon: int = 20,
        num_samples: int = 100,
        temperature: float = 1.0,
        control_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        latent_noise_scale: float = 1.0,
        device: str = "cpu"
    ):
        """
        Initialize LMPPI controller.
        
        Args:
            vae_model: Pre-trained VAE model for trajectory encoding/decoding
            state_dim: Dimension of state space
            control_dim: Dimension of control space
            cost_fn: Cost function g(state, control) -> cost
            terminal_cost_fn: Optional terminal cost function φ(state) -> cost
            horizon: Planning horizon length
            num_samples: Number of trajectory samples (K in algorithm)
            temperature: Temperature parameter for MPPI weighting
            control_bounds: Optional (min_control, max_control) bounds
            latent_noise_scale: Scale factor for latent space noise
            device: Device for computation
        """
        self.vae_model = vae_model.to(device)
        self.vae_model.eval()  # Set to evaluation mode
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.cost_fn = cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.latent_noise_scale = latent_noise_scale
        self.device = device
        
        # Control bounds
        if control_bounds is not None:
            self.control_min = control_bounds[0].to(device)
            self.control_max = control_bounds[1].to(device)
        else:
            self.control_min = None
            self.control_max = None
        
        # Feature dimension for VAE
        self.feature_dim = state_dim + control_dim
        
        # Initialize base latent vector (will be updated)
        self.base_latent = torch.zeros(
            1, self.vae_model.latent_dim, device=device
        )
        
        # Warm start trajectory storage
        self.current_trajectory = None
    
    def set_reference_trajectory(self, trajectory: torch.Tensor):
        """
        Set reference trajectory and encode to latent space.
        
        Args:
            trajectory: Reference trajectory [horizon, state_dim + control_dim]
                       or [batch_size, horizon, state_dim + control_dim]
        """
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)  # Add batch dimension
            
        # Ensure correct format for VAE
        if self.vae_model.architecture == "mlp":
            # Flatten for MLP
            trajectory_input = trajectory.view(trajectory.size(0), -1)
        else:
            # Keep structured for LSTM/CNN
            trajectory_input = trajectory
            
        # Encode to latent space
        with torch.no_grad():
            mu, _ = self.vae_model.encode(trajectory_input)
            self.base_latent = mu
            
        # Store the trajectory for warm starting
        self.current_trajectory = trajectory.clone()
    
    def solve(
        self,
        initial_state: torch.Tensor,
        num_iterations: int = 10,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Solve optimal control problem using LMPPI.
        
        Args:
            initial_state: Initial state [batch_size, state_dim]
            num_iterations: Number of LMPPI iterations
            verbose: Print convergence information
            
        Returns:
            Optimal control sequence [batch_size, horizon, control_dim]
        """
        # Ensure batch input
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
            
        batch_size = initial_state.shape[0]
        
        # Initialize base latent for batch if needed
        if self.base_latent.shape[0] != batch_size:
            self.base_latent = self.base_latent.repeat(batch_size, 1)
        
        best_costs = torch.full((batch_size,), float('inf'), device=self.device)
        best_latent = self.base_latent.clone()
        
        for iteration in range(num_iterations):
            # Sample in latent space (much lower dimensional!)
            latent_noise = torch.randn(
                batch_size, self.num_samples, self.vae_model.latent_dim,
                device=self.device
            ) * self.latent_noise_scale
            
            # Generate candidate latent vectors
            candidate_latents = (self.base_latent.unsqueeze(1) + latent_noise)
            
            # Decode to trajectories (replaces expensive forward integration!)
            candidate_trajectories = self._decode_latent_batch(candidate_latents)
            
            # Extract state and control sequences
            states, controls = self._split_trajectory(
                candidate_trajectories, initial_state
            )
            
            # Evaluate trajectory costs
            costs = self._evaluate_trajectories_batch(states, controls)
            
            # Compute MPPI weights
            weights = F.softmax(-costs / self.temperature, dim=1)
            
            # Update latent representation (weighted average in latent space)
            weighted_noise = torch.sum(
                weights.unsqueeze(-1) * latent_noise, dim=1
            )
            self.base_latent = self.base_latent + weighted_noise
            
            # Track best solutions
            current_costs = torch.min(costs, dim=1)[0]
            better_mask = current_costs < best_costs
            best_costs[better_mask] = current_costs[better_mask]
            
            # Update best latent representations
            best_indices = torch.argmin(costs, dim=1)
            for i in range(batch_size):
                if better_mask[i]:
                    best_latent[i] = candidate_latents[i, best_indices[i]]
            
            if verbose and iteration % 5 == 0:
                avg_cost = torch.mean(current_costs).item()
                print(f"LMPPI Iteration {iteration}: Avg Cost = {avg_cost:.6f}")
        
        # Use best latent for final trajectory
        self.base_latent = best_latent
        
        # Decode final trajectory and extract controls
        final_trajectory = self._decode_latent_batch(
            self.base_latent.unsqueeze(1)
        ).squeeze(1)
        
        _, final_controls = self._split_trajectory(final_trajectory, initial_state)
        
        return final_controls
    
    def _decode_latent_batch(self, latent_batch: torch.Tensor) -> torch.Tensor:
        """
        Decode batch of latent vectors to trajectories.
        
        Args:
            latent_batch: Latent vectors [batch_size, num_samples, latent_dim]
                         or [batch_size, latent_dim]
            
        Returns:
            Decoded trajectories [batch_size, num_samples, horizon, feature_dim]
            or [batch_size, horizon, feature_dim]
        """
        original_shape = latent_batch.shape
        
        # Flatten for decoding
        if latent_batch.dim() == 3:
            batch_size, num_samples, latent_dim = latent_batch.shape
            flat_latents = latent_batch.view(batch_size * num_samples, latent_dim)
        else:
            batch_size = latent_batch.shape[0]
            num_samples = 1
            flat_latents = latent_batch
            
        with torch.no_grad():
            decoded_flat = self.vae_model.decode(flat_latents)
            
        # Reshape back
        if len(original_shape) == 3:
            if self.vae_model.architecture == "mlp":
                # Reshape flattened output
                decoded = decoded_flat.view(
                    batch_size, num_samples, self.horizon, self.feature_dim
                )
            else:
                # Already in correct shape for LSTM/CNN
                decoded = decoded_flat.view(
                    batch_size, num_samples, self.horizon, self.feature_dim
                )
        else:
            if self.vae_model.architecture == "mlp":
                decoded = decoded_flat.view(-1, self.horizon, self.feature_dim)
            else:
                decoded = decoded_flat
                
        return decoded
    
    def _split_trajectory(
        self, 
        trajectories: torch.Tensor, 
        initial_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split trajectory into state and control sequences.
        
        Args:
            trajectories: Full trajectories [..., horizon, state_dim + control_dim]
            initial_states: Initial states [batch_size, state_dim]
            
        Returns:
            states: State sequences [..., horizon+1, state_dim]
            controls: Control sequences [..., horizon, control_dim]
        """
        # Extract controls from trajectory
        controls = trajectories[..., self.state_dim:]
        
        # Reconstruct states by integration (assuming trajectory includes states)
        trajectory_states = trajectories[..., :self.state_dim]
        
        # Create full state trajectory with initial state
        batch_dims = trajectories.shape[:-2]
        states = torch.zeros(
            *batch_dims, self.horizon + 1, self.state_dim, 
            device=self.device
        )
        
        # Set initial states
        if len(batch_dims) == 1:  # [batch_size, ...]
            states[:, 0, :] = initial_states
            states[:, 1:, :] = trajectory_states
        elif len(batch_dims) == 2:  # [batch_size, num_samples, ...]
            states[:, :, 0, :] = initial_states.unsqueeze(1)
            states[:, :, 1:, :] = trajectory_states
        else:
            raise ValueError(f"Unsupported trajectory shape: {trajectories.shape}")
        
        return states, controls
    
    def _evaluate_trajectories_batch(
        self,
        states: torch.Tensor,
        controls: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate costs for batch of trajectories.
        
        Args:
            states: State sequences [batch_size, num_samples, horizon+1, state_dim]
            controls: Control sequences [batch_size, num_samples, horizon, control_dim]
            
        Returns:
            Costs [batch_size, num_samples]
        """
        batch_size, num_samples = controls.shape[:2]
        
        # Flatten for cost evaluation
        flat_states = states.view(-1, self.horizon + 1, self.state_dim)
        flat_controls = controls.view(-1, self.horizon, self.control_dim)
        
        total_costs = torch.zeros(batch_size * num_samples, device=self.device)
        
        # Evaluate running costs
        for t in range(self.horizon):
            step_states = flat_states[:, t, :]
            step_controls = flat_controls[:, t, :]
            
            # Apply control bounds if specified
            if self.control_min is not None and self.control_max is not None:
                step_controls = torch.clamp(
                    step_controls, self.control_min, self.control_max
                )
            
            step_costs = self.cost_fn(step_states, step_controls)
            total_costs += step_costs
        
        # Add terminal cost if provided
        if self.terminal_cost_fn is not None:
            final_states = flat_states[:, -1, :]
            terminal_costs = self.terminal_cost_fn(final_states)
            total_costs += terminal_costs
        
        # Reshape back
        return total_costs.view(batch_size, num_samples)
    
    def step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get next control action for current state (MPC-style).
        
        Args:
            state: Current state [batch_size, state_dim]
            
        Returns:
            Control action [batch_size, control_dim]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Solve for optimal control sequence
        control_sequence = self.solve(state, num_iterations=5)
        
        # Return first control action
        first_control = control_sequence[:, 0, :].detach()
        
        # Apply warm start shifting for next iteration
        self.warm_start_shift()
        
        return first_control
    
    def warm_start_shift(self):
        """
        Perform warm start shifting by updating the base trajectory.
        
        This updates the stored trajectory for the next MPC iteration
        by shifting it forward one timestep.
        """
        if self.current_trajectory is not None:
            # Decode current latent to get trajectory
            current_decoded = self._decode_latent_batch(self.base_latent)
            
            # Shift trajectory forward (remove first timestep, duplicate last)
            shifted_trajectory = torch.cat([
                current_decoded[:, 1:, :],
                current_decoded[:, -1:, :].clone()
            ], dim=1)
            
            # Re-encode shifted trajectory
            if self.vae_model.architecture == "mlp":
                trajectory_input = shifted_trajectory.view(shifted_trajectory.size(0), -1)
            else:
                trajectory_input = shifted_trajectory
                
            with torch.no_grad():
                mu, _ = self.vae_model.encode(trajectory_input)
                self.base_latent = mu
                
            self.current_trajectory = shifted_trajectory.clone()
    
    def reset(self, initial_state: Optional[torch.Tensor] = None):
        """
        Reset the controller state.
        
        Args:
            initial_state: Optional initial state to reset to
        """
        # Reset base latent to zero
        batch_size = 1 if initial_state is None else initial_state.shape[0]
        self.base_latent = torch.zeros(
            batch_size, self.vae_model.latent_dim, device=self.device
        )
        
        # Clear trajectory cache
        self.current_trajectory = None
    
    def rollout(
        self,
        initial_state: torch.Tensor,
        control_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Rollout trajectory from decoded controls.
        
        Args:
            initial_state: Initial states [batch_size, state_dim]
            control_sequence: Optional control sequence to use
            
        Returns:
            Full trajectory [batch_size, horizon+1, state_dim + control_dim]
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
            
        if control_sequence is None:
            # Decode current latent representation
            decoded_trajectory = self._decode_latent_batch(self.base_latent)
            _, controls = self._split_trajectory(decoded_trajectory, initial_state)
        else:
            controls = control_sequence
            
        batch_size = initial_state.shape[0]
        
        # Create full trajectory
        trajectory = torch.zeros(
            batch_size, self.horizon + 1, self.feature_dim, device=self.device
        )
        
        # Set initial state
        trajectory[:, 0, :self.state_dim] = initial_state
        
        # Fill in trajectory
        current_states = initial_state
        for t in range(self.horizon):
            step_controls = controls[:, t, :]
            
            # Apply bounds
            if self.control_min is not None and self.control_max is not None:
                step_controls = torch.clamp(
                    step_controls, self.control_min, self.control_max
                )
            
            # Store control
            trajectory[:, t, self.state_dim:] = step_controls
            
            # Update state (this requires a dynamics model)
            if hasattr(self, '_dynamics_fn'):
                if t < self.horizon - 1:  # Only update if not at the last step
                    next_states = self._dynamics_fn(current_states, step_controls)
                    trajectory[:, t + 1, :self.state_dim] = next_states
                    current_states = next_states
            else:
                warnings.warn(
                    "No dynamics function provided. Using VAE-decoded states."
                )
                decoded_trajectory = self._decode_latent_batch(self.base_latent)
                if t < self.horizon - 1:  # Only update if not at the last step
                    trajectory[:, t + 1, :self.state_dim] = decoded_trajectory[:, t + 1, :self.state_dim]
                
        return trajectory
    
    def set_dynamics_function(self, dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Set dynamics function for accurate rollout.
        
        Args:
            dynamics_fn: Dynamics function f(state, control) -> next_state
        """
        self._dynamics_fn = dynamics_fn
