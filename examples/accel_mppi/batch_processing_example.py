#!/usr/bin/env python3
"""
Example demonstrating batch processing capabilities in diff-mppi.
This example shows how to efficiently solve MPPI for multiple initial states simultaneously.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from diff_mppi import DiffMPPI

def pendulum_dynamics(state, control):
    """
    Pendulum dynamics function supporting batch processing.
    
    Args:
        state: State tensor [batch_size, 2] where state[:, 0] = theta, state[:, 1] = theta_dot
        control: Control tensor [batch_size, 1] where control[:, 0] = torque
        
    Returns:
        Next state [batch_size, 2]
    """
    # Ensure batch dimensions
    if state.dim() == 1:
        state = state.unsqueeze(0)
    if control.dim() == 1:
        control = control.unsqueeze(0)
        
    theta, theta_dot = state[:, 0], state[:, 1]
    u = control[:, 0]
    
    # Pendulum parameters
    g, l, m, b = 9.81, 1.0, 1.0, 0.1
    dt = 0.05
    
    # Dynamics: theta_ddot = (3g/2l) * sin(theta) + (3/ml^2) * u - (b/ml^2) * theta_dot
    theta_ddot = (3*g)/(2*l) * torch.sin(theta) + (3/(m*l**2)) * u - (b/(m*l**2)) * theta_dot
    
    # Euler integration
    new_theta = theta + dt * theta_dot
    new_theta_dot = theta_dot + dt * theta_ddot
    
    return torch.stack([new_theta, new_theta_dot], dim=1)

def pendulum_cost(state, control):
    """
    Pendulum cost function supporting batch processing.
    
    Args:
        state: State tensor [batch_size, 2]
        control: Control tensor [batch_size, 1]
        
    Returns:
        Cost tensor [batch_size]
    """
    # Ensure batch dimensions
    if state.dim() == 1:
        state = state.unsqueeze(0)
    if control.dim() == 1:
        control = control.unsqueeze(0)
        
    theta, theta_dot = state[:, 0], state[:, 1]
    u = control[:, 0]
    
    # Target: upright position (theta=0, theta_dot=0)
    target_theta = 0.0
    target_theta_dot = 0.0
    
    # Compute angle difference (handle wrapping)
    angle_diff = torch.atan2(torch.sin(theta - target_theta), torch.cos(theta - target_theta))
    
    # Quadratic cost: state deviation + control effort
    cost = (angle_diff**2 + 0.1 * (theta_dot - target_theta_dot)**2 + 0.01 * u**2).squeeze()
    
    return cost

def run_batch_example():
    """Run the batch processing example."""
    print("Batch Processing Example for Pendulum Swing-up")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create MPPI controller
    mppi = DiffMPPI(
        dynamics_fn=pendulum_dynamics,
        cost_fn=pendulum_cost,
        horizon=20,
        num_samples=200,
        state_dim=2,
        control_dim=1,
        control_min=-5.0,
        control_max=5.0,
        temperature=1.0,
        acceleration="adam",
        lr=0.1,
        device=str(device)
    )
    
    # Define multiple initial states
    initial_states = torch.tensor([
        [np.pi, 0.0],        # Downward, no velocity
        [np.pi, 2.0],        # Downward, high velocity  
        [np.pi/2, 0.0],      # Right side, no velocity
        [-np.pi/2, 0.0],     # Left side, no velocity
        [3*np.pi/4, -1.0],   # Upper-left, negative velocity
        [np.pi/4, 1.5],      # Upper-right, positive velocity
        [0.0, 0.0],          # Already at target
        [np.pi + 0.1, 0.0],  # Near downward
    ], device=device)
    
    batch_size = initial_states.shape[0]
    print(f"Processing {batch_size} initial states simultaneously")
    
    # Solve for all initial states in batch
    print("\\nSolving MPPI for all initial states...")
    optimal_controls = mppi.solve(initial_states, num_iterations=20, verbose=True)
    
    print(f"\\nOptimal control sequences shape: {optimal_controls.shape}")
    
    # Simulate trajectories for each initial state
    print("\\nSimulating optimal trajectories...")
    trajectories = mppi.rollout(initial_states, optimal_controls)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    time_steps = np.arange(trajectories.shape[1]) * 0.05  # dt = 0.05
    
    for i in range(batch_size):
        ax = axes[i]
        
        # Extract trajectory for this initial state
        traj = trajectories[i].cpu().numpy()
        theta_traj = traj[:, 0]
        theta_dot_traj = traj[:, 1]
        
        # Plot angle trajectory
        ax.plot(time_steps, theta_traj, 'b-', linewidth=2, label='θ (angle)')
        ax.plot(time_steps, theta_dot_traj, 'r--', linewidth=2, label='θ̇ (velocity)')
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='Target')
        
        ax.set_title(f'State {i+1}: θ₀={initial_states[i][0].item():.2f}, θ̇₀={initial_states[i][1].item():.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('batch_pendulum_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'batch_pendulum_results.png'")
    
    # Compute and display final costs
    print("\\nFinal costs for each initial state:")
    for i in range(batch_size):
        state = initial_states[i]
        controls = optimal_controls[i]
        
        # Compute total trajectory cost
        total_cost = 0.0
        current_state = state.unsqueeze(0)
        for t in range(mppi.horizon):
            control = controls[t].unsqueeze(0)
            cost = pendulum_cost(current_state, control)
            total_cost += cost.item()
            current_state = pendulum_dynamics(current_state, control)
        
        final_theta = trajectories[i, -1, 0].item()
        final_theta_dot = trajectories[i, -1, 1].item()
        
        print(f"State {i+1}: Initial=[{state[0].item():.3f}, {state[1].item():.3f}] "
              f"→ Final=[{final_theta:.3f}, {final_theta_dot:.3f}], Cost={total_cost:.4f}")
    
    # Performance analysis
    print("\\nPerformance Analysis:")
    print(f"- Batch size: {batch_size}")
    print(f"- Horizon: {mppi.horizon}")
    print(f"- Samples per iteration: {mppi.num_samples}")
    print(f"- Total evaluations: {batch_size * mppi.num_samples * 20}")  # 20 iterations
    
    # Compare with sequential processing
    import time
    
    print("\\nTiming comparison:")
    
    # Sequential processing
    start_time = time.time()
    for i in range(batch_size):
        mppi.solve(initial_states[i], num_iterations=10)
    sequential_time = time.time() - start_time
    
    # Batch processing
    start_time = time.time()
    mppi.solve(initial_states, num_iterations=10)
    batch_time = time.time() - start_time
    
    print(f"Sequential processing: {sequential_time:.4f}s")
    print(f"Batch processing: {batch_time:.4f}s")
    print(f"Speedup: {sequential_time / batch_time:.2f}x")
    
    print("\\n" + "="*50)
    print("Batch processing example completed successfully!")
    print("The batch implementation allows efficient parallel processing")
    print("of multiple initial states, which is crucial for applications like:")
    print("- Monte Carlo simulations")
    print("- Robust control design")
    print("- Multi-agent planning")
    print("- Imitation learning with multiple demonstrations")

if __name__ == "__main__":
    run_batch_example()
