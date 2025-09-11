#!/usr/bin/env python3
"""
Demonstration of Algorithm 3 warm start functionality for NAG acceleration.

This script shows how the warm start mechanism improves convergence by
shifting control sequences and momentum terms between MPC steps.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_mppi.core import DiffMPPI


def simple_pendulum_dynamics(states, controls):
    """
    Simple pendulum dynamics for batch processing.
    
    Args:
        states: [batch_size, state_dim] where state = [theta, theta_dot]
        controls: [batch_size, control_dim] or [batch_size, horizon, control_dim]
        
    Returns:
        next_states: [batch_size, state_dim]
    """
    batch_size = states.shape[0]
    
    if controls.dim() == 3:  # Rollout mode: [batch_size, horizon, control_dim]
        # For rollout, we typically only use the first control
        u = controls[:, 0, 0]  # First control for each batch
    else:  # Step mode: [batch_size, control_dim]
        u = controls[:, 0]
    
    theta = states[:, 0]
    theta_dot = states[:, 1]
    
    # Simple pendulum dynamics: theta_ddot = -sin(theta) + u
    dt = 0.1
    theta_ddot = -torch.sin(theta) + u
    
    # Euler integration
    next_theta = theta + dt * theta_dot
    next_theta_dot = theta_dot + dt * theta_ddot
    
    next_states = torch.stack([next_theta, next_theta_dot], dim=1)
    return next_states


def pendulum_cost(states, controls):
    """
    Cost function for pendulum stabilization.
    
    Args:
        states: [batch_size, state_dim]
        controls: [batch_size, control_dim] or [batch_size, horizon, control_dim]
        
    Returns:
        costs: [batch_size] or [batch_size, horizon]
    """
    batch_size = states.shape[0]
    
    # Target: upright position (theta=0, theta_dot=0)
    target_theta = 0.0
    target_theta_dot = 0.0
    
    theta_error = states[:, 0] - target_theta
    theta_dot_error = states[:, 1] - target_theta_dot
    
    # State cost: quadratic in angle and velocity errors
    state_cost = 10.0 * theta_error**2 + 1.0 * theta_dot_error**2
    
    if controls.dim() == 3:  # Rollout mode: [batch_size, horizon, control_dim]
        # Control cost for each time step
        control_cost = 0.1 * torch.sum(controls**2, dim=2)  # [batch_size, horizon]
        
        # Add state cost to each time step
        costs = state_cost.unsqueeze(1) + control_cost  # [batch_size, horizon]
    else:  # Step mode: [batch_size, control_dim]
        # Single step cost
        control_cost = 0.1 * torch.sum(controls**2, dim=1)  # [batch_size]
        costs = state_cost + control_cost  # [batch_size]
    
    return costs


def compare_with_without_warm_start():
    """Compare NAG performance with and without warm start."""
    print("🔄 Comparing NAG performance with and without warm start...")
    
    # Configuration
    state_dim = 2
    control_dim = 1
    horizon = 15
    num_samples = 500
    num_steps = 50
    batch_size = 3
    
    # Initial states: pendulum starting at different angles
    initial_states = torch.tensor([
        [np.pi, 0.0],      # Inverted (180°)
        [np.pi/2, 0.0],    # 90°
        [np.pi/4, 0.0]     # 45°
    ], dtype=torch.float32)
    
    # Test both with and without warm start
    results = {}
    
    for use_warm_start in [False, True]:
        print(f"\n{'WITH' if use_warm_start else 'WITHOUT'} warm start:")
        
        # Create MPPI controller
        mppi = DiffMPPI(
            dynamics_fn=simple_pendulum_dynamics,
            cost_fn=pendulum_cost,
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=horizon,
            num_samples=num_samples,
            acceleration="nag",
            gamma=0.8,
            lr=0.01,
            device="cpu"
        )
        
        # Reset and run simulation
        mppi.reset(initial_states)
        
        states_history = [initial_states.clone()]
        costs_history = []
        solve_times = []
        
        current_states = initial_states.clone()
        
        for step in range(num_steps):
            # Measure solve time
            import time
            start_cpu = time.time()
            
            # Solve for optimal control
            if use_warm_start:
                # Normal solve with warm start (automatic for NAG)
                optimal_controls = mppi.solve(current_states)
            else:
                # Disable warm start by reinitializing control sequences
                mppi.batch_control_sequences = torch.zeros_like(mppi.batch_control_sequences)
                if hasattr(mppi, 'batch_nag_prev_update'):
                    mppi.batch_nag_prev_update = torch.zeros_like(mppi.batch_nag_prev_update)
                optimal_controls = mppi.solve(current_states)
            
            solve_time = time.time() - start_cpu
            solve_times.append(solve_time)
            
            # Apply first control and simulate
            control_input = optimal_controls[:, 0, :]  # First control from horizon
            next_states = simple_pendulum_dynamics(current_states, control_input)
            
            # Calculate cost
            step_cost = pendulum_cost(current_states, control_input)
            costs_history.append(step_cost)
            
            # Update states
            current_states = next_states
            states_history.append(current_states.clone())
            
            # Progress update
            if (step + 1) % 10 == 0:
                avg_angle_error = torch.abs(current_states[:, 0]).mean() * 180 / np.pi
                avg_cost = step_cost.mean()
                print(f"  Step {step+1:2d}: avg_angle_error={avg_angle_error:.1f}°, avg_cost={avg_cost:.3f}")
        
        # Store results
        results[use_warm_start] = {
            'states_history': torch.stack(states_history),  # [num_steps+1, batch_size, state_dim]
            'costs_history': torch.stack(costs_history),    # [num_steps, batch_size]
            'solve_times': solve_times,
            'final_errors': torch.abs(current_states[:, 0]) * 180 / np.pi  # Convert to degrees
        }
        
        # Summary
        total_cost = torch.stack(costs_history).sum(dim=0)  # Sum over time for each batch
        avg_solve_time = np.mean(solve_times)
        
        print(f"  Final angle errors: {results[use_warm_start]['final_errors'].tolist()}")
        print(f"  Total costs: {total_cost.tolist()}")
        print(f"  Average solve time: {avg_solve_time:.4f}s")
    
    # Plot comparison
    plot_warm_start_comparison(results)
    
    return results


def plot_warm_start_comparison(results):
    """Plot comparison between with and without warm start."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Colors for different scenarios
    colors = ['red', 'green', 'blue']
    labels = ['180° start', '90° start', '45° start']
    
    # Plot 1: Angle trajectories
    ax = axes[0, 0]
    for use_warm_start, label_prefix in [(False, 'No WS'), (True, 'With WS')]:
        states_history = results[use_warm_start]['states_history']
        angles = states_history[:, :, 0] * 180 / np.pi  # Convert to degrees
        
        for i in range(3):  # 3 different initial conditions
            linestyle = '-' if use_warm_start else '--'
            ax.plot(angles[:, i], color=colors[i], linestyle=linestyle, 
                   alpha=0.8, label=f'{labels[i]} ({label_prefix})')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Pendulum Angle Trajectories')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Target')
    
    # Plot 2: Cost evolution
    ax = axes[0, 1]
    for use_warm_start, label_prefix in [(False, 'No WS'), (True, 'With WS')]:
        costs_history = results[use_warm_start]['costs_history']
        avg_costs = costs_history.mean(dim=1)  # Average across batch
        
        linestyle = '-' if use_warm_start else '--'
        ax.plot(avg_costs, linestyle=linestyle, linewidth=2,
               label=f'Average Cost ({label_prefix})')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Cost')
    ax.set_title('Cost Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # Plot 3: Final errors comparison
    ax = axes[1, 0]
    x_pos = np.arange(3)
    width = 0.35
    
    no_ws_errors = results[False]['final_errors']
    ws_errors = results[True]['final_errors']
    
    ax.bar(x_pos - width/2, no_ws_errors, width, label='Without Warm Start', alpha=0.7)
    ax.bar(x_pos + width/2, ws_errors, width, label='With Warm Start', alpha=0.7)
    
    ax.set_xlabel('Initial Condition')
    ax.set_ylabel('Final Angle Error (degrees)')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Solve time comparison
    ax = axes[1, 1]
    no_ws_times = results[False]['solve_times']
    ws_times = results[True]['solve_times']
    
    ax.plot(no_ws_times, '--', alpha=0.7, label='Without Warm Start')
    ax.plot(ws_times, '-', alpha=0.7, label='With Warm Start')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Solve Time (seconds)')
    ax.set_title('Computational Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('warm_start_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved as 'warm_start_comparison.png'")
    
    # Print summary
    print("\n" + "="*60)
    print("WARM START PERFORMANCE SUMMARY")
    print("="*60)
    
    for use_warm_start in [False, True]:
        label = "WITH warm start" if use_warm_start else "WITHOUT warm start"
        print(f"\n{label}:")
        
        final_errors = results[use_warm_start]['final_errors']
        total_costs = results[use_warm_start]['costs_history'].sum(dim=0)
        avg_solve_time = np.mean(results[use_warm_start]['solve_times'])
        
        print(f"  Average final error: {final_errors.mean():.1f}°")
        print(f"  Average total cost: {total_costs.mean():.1f}")
        print(f"  Average solve time: {avg_solve_time:.4f}s")
    
    # Calculate improvement
    improvement_error = (results[False]['final_errors'].mean() - results[True]['final_errors'].mean())
    improvement_cost = (results[False]['costs_history'].sum(dim=0).mean() - 
                       results[True]['costs_history'].sum(dim=0).mean())
    improvement_time = (np.mean(results[False]['solve_times']) - 
                       np.mean(results[True]['solve_times']))
    
    print(f"\nIMPROVEMENT with warm start:")
    print(f"  Angle error reduction: {improvement_error:.1f}°")
    print(f"  Cost reduction: {improvement_cost:.1f}")
    print(f"  Time reduction: {improvement_time:.4f}s per step")


if __name__ == "__main__":
    print("🚀 Algorithm 3 Warm Start Demonstration")
    print("="*60)
    print("This demo compares NAG-accelerated MPPI with and without warm start.")
    print("The warm start mechanism shifts control sequences and momentum terms")
    print("forward between MPC steps, as described in Algorithm 3 of the paper.")
    print("="*60)
    
    try:
        results = compare_with_without_warm_start()
        print("\n✅ Demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
