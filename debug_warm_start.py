#!/usr/bin/env python3
"""Debug warm start shift functionality."""

import torch
import sys
import os

# Add the current directory to Python path  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diff_mppi.core import DiffMPPI


def simple_dynamics(states, controls):
    """Simple dynamics for testing."""
    batch_size = states.shape[0]
    next_states = torch.zeros_like(states)
    
    for i in range(batch_size):
        theta, theta_dot = states[i, 0], states[i, 1]
        u = controls[i, 0] if controls.dim() == 2 else controls[i, 0, 0]
        
        theta_ddot = -torch.sin(theta) + u
        next_states[i, 0] = theta + 0.1 * theta_dot
        next_states[i, 1] = theta_dot + 0.1 * theta_ddot
    
    return next_states


def simple_cost(states, controls):
    """Simple cost function."""
    batch_size = states.shape[0]
    if controls.dim() == 3:
        costs = torch.zeros(batch_size, controls.shape[1])
        for i in range(batch_size):
            costs[i] = torch.sum(states[i]**2) + 0.1 * torch.sum(controls[i]**2, dim=-1)
    else:
        costs = torch.zeros(batch_size)
        for i in range(batch_size):
            costs[i] = torch.sum(states[i]**2) + 0.1 * torch.sum(controls[i]**2)
    
    return costs


def debug_warm_start():
    """Debug the warm start shift functionality."""
    print("Debugging warm start shift...")
    
    state_dim = 2
    control_dim = 1
    horizon = 5
    batch_size = 2
    
    # Create MPPI with NAG
    mppi = DiffMPPI(
        dynamics_fn=simple_dynamics,
        cost_fn=simple_cost,
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
        num_samples=50,
        acceleration="nag",
        gamma=0.8,
        device="cpu"
    )
    
    # Initialize
    batch_states = torch.randn(batch_size, state_dim)
    mppi.reset(batch_states)
    
    # Set specific test values
    test_controls = torch.arange(batch_size * horizon * control_dim, dtype=torch.float32).reshape(
        batch_size, horizon, control_dim
    )
    mppi.batch_control_sequences = test_controls.clone()
    
    print("Original control sequences:")
    print(mppi.batch_control_sequences)
    print()
    
    print("Expected after shift (elements 1:T):")
    expected_shifted = test_controls[:, 1:, :]
    print(expected_shifted)
    print()
    
    print("Expected last element (copy of element T-2):")
    expected_last = test_controls[:, -2, :]  # This should be element at index -2
    print(f"Element at index -2: {expected_last}")
    print()
    
    # Apply warm start shift
    mppi.warm_start_shift(fill_method="replicate")
    
    print("After warm start shift:")
    print(mppi.batch_control_sequences)
    print()
    
    print("Actual shifted part (elements 0:T-1):")
    actual_shifted = mppi.batch_control_sequences[:, :-1, :]
    print(actual_shifted)
    print()
    
    print("Actual last element:")
    actual_last = mppi.batch_control_sequences[:, -1, :]
    print(actual_last)
    print()
    
    # Check if shifting worked
    shift_correct = torch.allclose(expected_shifted, actual_shifted)
    print(f"Shift correct: {shift_correct}")
    
    # Check if last element is correct
    last_correct = torch.allclose(expected_last, actual_last)
    print(f"Last element correct: {last_correct}")
    
    if not last_correct:
        print(f"Expected last: {expected_last}")
        print(f"Actual last: {actual_last}")
        print(f"Difference: {torch.abs(expected_last - actual_last)}")


if __name__ == "__main__":
    debug_warm_start()
