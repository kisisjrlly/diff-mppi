"""
Simple LMPPI Example for Pendulum Control

This example demonstrates how to use LMPPI for controlling a simple pendulum.
It shows the complete workflow from data generation to control.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the diff_mppi path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from diff_mppi.lmppi.models import TrajectoryVAE
    from diff_mppi.lmppi.controller import LMPPIController
    from diff_mppi.lmppi.trainer import LMPPITrainer
    from diff_mppi.lmppi.data import TrajectoryDataset, create_synthetic_trajectories
    from diff_mppi.lmppi.config import pendulum_config
    print("✓ All LMPPI modules imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Checking individual imports...")
    
    try:
        from diff_mppi.lmppi.models import TrajectoryVAE
        print("✓ Models imported")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
    
    try:
        from diff_mppi.lmppi.data import create_synthetic_trajectories
        print("✓ Data imported")
    except ImportError as e:
        print(f"✗ Data import failed: {e}")
    
    sys.exit(1)


def pendulum_dynamics(state, control):
    """
    Simple pendulum dynamics.
    State: [theta, theta_dot]
    Control: [torque]
    """
    g = 9.81
    l = 1.0
    m = 1.0
    dt = 0.05
    
    theta = state[..., 0]
    theta_dot = state[..., 1]
    torque = control[..., 0]
    
    # Pendulum equation: theta_ddot = -(g/l)*sin(theta) + torque/(m*l^2)
    theta_ddot = -(g/l) * torch.sin(theta) + torque / (m * l**2)
    
    # Euler integration
    new_theta = theta + dt * theta_dot
    new_theta_dot = theta_dot + dt * theta_ddot
    
    # Wrap angle to [-pi, pi]
    new_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta))
    
    return torch.stack([new_theta, new_theta_dot], dim=-1)


def pendulum_cost(state, control):
    """
    Cost function for pendulum swing-up.
    Goal: swing pendulum to upright position (theta=0, theta_dot=0)
    """
    theta = state[..., 0]
    theta_dot = state[..., 1]
    torque = control[..., 0]
    
    # Cost for being away from upright position
    angle_cost = 1 - torch.cos(theta)  # 0 when upright, 2 when hanging down
    
    # Cost for angular velocity
    velocity_cost = 0.1 * theta_dot**2
    
    # Control effort cost
    control_cost = 0.01 * torque**2
    
    return angle_cost + velocity_cost + control_cost


def generate_pendulum_trajectories(num_trajectories=200, horizon=25):
    """Generate training data for pendulum."""
    print(f"Generating {num_trajectories} pendulum trajectories...")
    
    trajectories = []
    
    for i in range(num_trajectories):
        # Random initial conditions
        theta_0 = np.random.uniform(-np.pi, np.pi)
        theta_dot_0 = np.random.uniform(-5, 5)
        
        # Generate random control sequence (simple exploration)
        controls = np.random.uniform(-2, 2, size=(horizon, 1))
        
        # Simulate trajectory
        states = np.zeros((horizon + 1, 2))
        states[0] = [theta_0, theta_dot_0]
        
        for t in range(horizon):
            state_tensor = torch.tensor(states[t], dtype=torch.float32)
            control_tensor = torch.tensor(controls[t], dtype=torch.float32)
            next_state = pendulum_dynamics(state_tensor, control_tensor)
            states[t + 1] = next_state.numpy()
        
        # Combine states and controls for trajectory
        trajectory = np.zeros((horizon, 3))  # [theta, theta_dot, torque]
        trajectory[:, :2] = states[:-1]  # States (excluding last)
        trajectory[:, 2:] = controls  # Controls
        
        trajectories.append(trajectory)
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{num_trajectories} trajectories")
    
    return np.array(trajectories)


def main():
    """Main LMPPI demonstration."""
    print("🚀 Starting LMPPI Pendulum Example")
    print("=" * 50)
    
    # Set device
    device = "cpu"  # Force CPU for compatibility
    print(f"Using device: {device}")
    
    # Parameters
    horizon = 25
    state_dim = 2  # [theta, theta_dot]
    control_dim = 1  # [torque]
    latent_dim = 6
    
    # Step 1: Generate training data
    print("\n📊 Step 1: Generating training data")
    trajectories_np = generate_pendulum_trajectories(
        num_trajectories=100,  # Reduced for quick demo
        horizon=horizon
    )
    
    # Create dataset
    dataset = TrajectoryDataset(
        trajectories=trajectories_np,
        state_dim=state_dim,
        control_dim=control_dim,
        normalize=True,
        device=device
    )
    
    print(f"Dataset created: {len(dataset)} trajectories")
    print(f"Trajectory shape: {dataset.trajectories.shape}")
    
    # Split dataset
    train_dataset, val_dataset = dataset.split(train_ratio=0.8)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Step 2: Create and train VAE
    print("\n🧠 Step 2: Creating and training VAE model")
    
    vae_model = TrajectoryVAE(
        input_dim=horizon * (state_dim + control_dim),
        latent_dim=latent_dim,
        hidden_dims=[128, 64, 32],
        architecture="mlp",
        beta=0.5
    )
    
    print(f"VAE Model created:")
    print(f"  Input dim: {horizon * (state_dim + control_dim)}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Architecture: MLP")
    
    # Quick training
    trainer = LMPPITrainer(
        model=vae_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=16,
        learning_rate=1e-3,
        device=device,
        save_dir="./lmppi_pendulum_demo"
    )
    
    print("Training VAE (quick demo - 10 epochs)...")
    metrics = trainer.train(num_epochs=10, early_stopping_patience=5)
    
    print("Training completed!")
    print(f"Final train loss: {metrics['train_loss'][-1]:.4f}")
    if 'val_loss' in metrics:
        print(f"Final val loss: {metrics['val_loss'][-1]:.4f}")
    
    # Step 3: Create LMPPI controller
    print("\n🎮 Step 3: Creating LMPPI controller")
    
    controller = LMPPIController(
        vae_model=vae_model,
        state_dim=state_dim,
        control_dim=control_dim,
        cost_fn=pendulum_cost,
        horizon=horizon,
        num_samples=50,  # Reduced for quick demo
        temperature=1.0,
        control_bounds=(torch.tensor([-5.0]), torch.tensor([5.0])),
        device=device
    )
    
    print("LMPPI Controller created successfully!")
    
    # Step 4: Test control
    print("\n🎯 Step 4: Testing control")
    
    # Initial state: pendulum hanging down
    initial_state = torch.tensor([[np.pi, 0.0]], device=device)  # [theta, theta_dot]
    print(f"Initial state: theta={initial_state[0,0]:.2f}, theta_dot={initial_state[0,1]:.2f}")
    
    # Solve for optimal control
    print("Solving optimal control problem...")
    optimal_controls = controller.solve(initial_state, num_iterations=5, verbose=True)
    
    print("Control solution found!")
    print(f"First control action: {optimal_controls[0, 0, 0]:.4f}")
    
    # Step 5: Simulate controlled trajectory
    print("\n📈 Step 5: Simulating controlled trajectory")
    
    # Simulate with LMPPI controls
    trajectory_states = torch.zeros(horizon + 1, 2, device=device)
    trajectory_controls = torch.zeros(horizon, 1, device=device)
    trajectory_costs = torch.zeros(horizon, device=device)
    
    current_state = initial_state.clone()
    trajectory_states[0] = current_state[0]
    
    for t in range(horizon):
        # Get control from LMPPI
        control = optimal_controls[0, t:t+1]  # Take t-th control
        trajectory_controls[t] = control[0]
        
        # Compute cost
        cost = pendulum_cost(current_state, control)
        trajectory_costs[t] = cost[0]
        
        # Simulate dynamics
        next_state = pendulum_dynamics(current_state, control)
        trajectory_states[t + 1] = next_state[0]
        current_state = next_state
    
    total_cost = torch.sum(trajectory_costs).item()
    final_state = trajectory_states[-1]
    
    print("Simulation completed!")
    print(f"Total cost: {total_cost:.4f}")
    print(f"Final state: theta={final_state[0]:.2f}, theta_dot={final_state[1]:.2f}")
    
    # Convert to numpy for plotting
    states_np = trajectory_states.cpu().numpy()
    controls_np = trajectory_controls.cpu().numpy()
    costs_np = trajectory_costs.cpu().numpy()
    
    # Step 6: Plot results
    print("\n📊 Step 6: Plotting results")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot trajectory
    time_steps = np.arange(horizon + 1) * 0.05  # dt = 0.05
    control_time = np.arange(horizon) * 0.05
    
    # Angle trajectory
    axes[0, 0].plot(time_steps, states_np[:, 0])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (rad)')
    axes[0, 0].set_title('Pendulum Angle')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0, 0].legend()
    
    # Angular velocity
    axes[0, 1].plot(time_steps, states_np[:, 1])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 1].set_title('Angular Velocity')
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].legend()
    
    # Control input
    axes[1, 0].plot(control_time, controls_np[:, 0])
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Torque (Nm)')
    axes[1, 0].set_title('Control Input')
    axes[1, 0].grid(True)
    
    # Cost over time
    axes[1, 1].plot(control_time, costs_np)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Instantaneous Cost')
    axes[1, 1].set_title('Cost Over Time')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('lmppi_pendulum_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'lmppi_pendulum_results.png'")
    
    # Summary
    print("\n✅ LMPPI Demo Summary")
    print("=" * 50)
    print(f"✓ Generated {len(dataset)} training trajectories")
    print(f"✓ Trained VAE with {latent_dim}D latent space")
    print(f"✓ LMPPI controller with {controller.num_samples} samples")
    print(f"✓ Achieved total cost: {total_cost:.4f}")
    print(f"✓ Final angle error: {abs(final_state[0]):.4f} rad")
    
    # Comparison info
    print(f"\n📊 Computational Benefits:")
    standard_mppi_dim = horizon * control_dim  # 25 * 1 = 25
    lmppi_dim = latent_dim  # 6
    print(f"Standard MPPI sampling dimension: {standard_mppi_dim}")
    print(f"LMPPI sampling dimension: {lmppi_dim}")
    print(f"Dimension reduction: {standard_mppi_dim / lmppi_dim:.1f}x")
    
    print("\n🎉 LMPPI demonstration completed successfully!")


if __name__ == "__main__":
    main()
