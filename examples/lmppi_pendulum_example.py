"""
LMPPI Example: Pendulum Control with Latent Space Learning

This example demonstrates the complete LMPPI workflow:
1. Generate/collect trajectory data
2. Train a VAE on trajectory data 
3. Use trained VAE for online LMPPI control
4. Compare performance with standard MPPI

The example uses a simple pendulum system to showcase the approach.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Add the parent directory to the path so we can import diff_mppi
import sys
sys.path.append(str(Path(__file__).parent.parent))

from diff_mppi.lmppi import (
    TrajectoryVAE, LMPPIController, LMPPITrainer,
    TrajectoryDataset, create_synthetic_trajectories
)
from diff_mppi import DiffMPPI


def pendulum_dynamics(state, control):
    """
    Simple pendulum dynamics: [theta, theta_dot]
    Control: [torque]
    """
    theta, theta_dot = state[..., 0], state[..., 1]
    torque = control[..., 0]
    
    # Pendulum parameters
    g = 9.81
    l = 1.0
    m = 1.0
    b = 0.1  # damping
    
    # Dynamics
    theta_ddot = (torque - m * g * l * torch.sin(theta) - b * theta_dot) / (m * l**2)
    
    # Integration (Euler)
    dt = 0.05
    next_theta = theta + theta_dot * dt
    next_theta_dot = theta_dot + theta_ddot * dt
    
    return torch.stack([next_theta, next_theta_dot], dim=-1)


def pendulum_cost(state, control):
    """Cost function for pendulum swing-up."""
    theta, theta_dot = state[..., 0], state[..., 1]
    torque = control[..., 0]
    
    # Target is upright position (theta = pi)
    target_theta = torch.pi
    
    # Angle cost (wrap around)
    angle_diff = torch.atan2(torch.sin(theta - target_theta), torch.cos(theta - target_theta))
    angle_cost = angle_diff**2
    
    # Velocity cost
    velocity_cost = 0.1 * theta_dot**2
    
    # Control cost
    control_cost = 0.01 * torque**2
    
    return angle_cost + velocity_cost + control_cost


def generate_demonstration_data(num_trajectories=1000, horizon=50, device="cpu"):
    """
    Generate demonstration trajectories using a simple controller.
    In practice, this could be expert demonstrations or trajectories from other controllers.
    """
    print("Generating demonstration trajectories...")
    
    trajectories = []
    
    for i in range(num_trajectories):
        # Random initial state
        theta_0 = np.random.uniform(-np.pi, np.pi)
        theta_dot_0 = np.random.uniform(-5, 5)
        
        state = torch.tensor([theta_0, theta_dot_0], device=device, dtype=torch.float32)
        
        # Storage for trajectory
        states = [state.clone()]
        controls = []
        
        # Simple PD controller for swing-up
        kp = np.random.uniform(10, 30)  # Random controller gains for diversity
        kd = np.random.uniform(1, 5)
        
        for t in range(horizon):
            # Current state
            theta, theta_dot = state[0], state[1]
            
            # Target is upright
            target_theta = np.pi
            
            # PD control
            angle_error = torch.atan2(torch.sin(theta - target_theta), torch.cos(theta - target_theta))
            velocity_error = -theta_dot  # Target velocity is 0
            
            torque = kp * angle_error + kd * velocity_error
            
            # Add some noise for diversity
            torque += torch.randn(1, device=device) * 0.5
            
            # Clamp control
            torque = torch.clamp(torque, -10, 10)
            
            control = torque.unsqueeze(0)
            controls.append(control.clone())
            
            # Dynamics
            next_state = pendulum_dynamics(state.unsqueeze(0), control.unsqueeze(0)).squeeze(0)
            states.append(next_state.clone())
            state = next_state
        
        # Convert to trajectory format [horizon, state_dim + control_dim]
        states_tensor = torch.stack(states[:-1])  # Remove last state to match control length
        controls_tensor = torch.stack(controls)
        trajectory = torch.cat([states_tensor, controls_tensor], dim=1)
        
        trajectories.append(trajectory.cpu().numpy())
    
    print(f"Generated {len(trajectories)} trajectories")
    return trajectories


def train_vae_model(trajectories, state_dim=2, control_dim=1, latent_dim=8, device="cpu"):
    """Train VAE on trajectory data."""
    print("Training VAE model...")
    
    # Create dataset
    dataset = TrajectoryDataset(
        trajectories=trajectories,
        state_dim=state_dim,
        control_dim=control_dim,
        normalize=True,
        augment=True,
        device=device
    )
    
    # Split into train/val
    train_dataset, val_dataset = dataset.split(train_ratio=0.8)
    
    # Create VAE model
    horizon = dataset.horizon
    feature_dim = state_dim + control_dim
    
    vae_model = TrajectoryVAE(
        input_dim=horizon * feature_dim,  # For MLP architecture
        latent_dim=latent_dim,
        hidden_dims=[256, 128, 64],
        architecture="mlp",
        horizon=horizon,
        feature_dim=feature_dim,
        beta=1.0
    )
    
    # Create trainer
    trainer = LMPPITrainer(
        model=vae_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,
        learning_rate=1e-3,
        device=device,
        save_dir="./lmppi_checkpoints"
    )
    
    # Train model
    metrics = trainer.train(
        num_epochs=50,
        early_stopping_patience=10
    )
    
    # Evaluate reconstruction
    eval_metrics = trainer.evaluate_reconstruction(save_plots=True)
    print("Reconstruction metrics:", eval_metrics)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    return vae_model, trainer


def compare_controllers(vae_model, device="cpu"):
    """Compare LMPPI with standard MPPI."""
    print("Comparing LMPPI vs standard MPPI...")
    
    # Test parameters
    num_test_episodes = 5
    horizon = 20
    num_samples = 100
    
    # Control bounds
    control_min = torch.tensor([-10.0], device=device)
    control_max = torch.tensor([10.0], device=device)
    
    # Create LMPPI controller
    lmppi = LMPPIController(
        vae_model=vae_model,
        state_dim=2,
        control_dim=1,
        cost_fn=pendulum_cost,
        horizon=horizon,
        num_samples=num_samples,
        temperature=1.0,
        control_bounds=(control_min, control_max),
        device=device
    )
    
    # Set dynamics function for accurate rollout
    lmppi.set_dynamics_function(pendulum_dynamics)
    
    # Create standard MPPI controller
    standard_mppi = DiffMPPI(
        state_dim=2,
        control_dim=1,
        dynamics_fn=pendulum_dynamics,
        cost_fn=pendulum_cost,
        horizon=horizon,
        num_samples=num_samples,
        temperature=1.0,
        control_bounds=(control_min, control_max),
        device=device
    )
    
    # Test both controllers
    lmppi_costs = []
    standard_costs = []
    
    for episode in range(num_test_episodes):
        print(f"Test episode {episode + 1}/{num_test_episodes}")
        
        # Random initial state
        initial_state = torch.tensor([
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-5, 5)
        ], device=device).unsqueeze(0)  # Add batch dimension
        
        # Test LMPPI
        lmppi.reset(initial_state)
        lmppi_control_seq = lmppi.solve(initial_state, num_iterations=10)
        lmppi_trajectory = lmppi.rollout(initial_state, lmppi_control_seq)
        
        # Compute LMPPI cost
        lmppi_cost = 0
        for t in range(horizon):
            state = lmppi_trajectory[0, t, :2]
            control = lmppi_control_seq[0, t, :]
            lmppi_cost += pendulum_cost(state, control).item()
        lmppi_costs.append(lmppi_cost)
        
        # Test standard MPPI
        standard_mppi.reset(initial_state)
        standard_control_seq = standard_mppi.solve(initial_state, num_iterations=10)
        standard_trajectory = standard_mppi.rollout(initial_state, standard_control_seq)
        
        # Compute standard MPPI cost
        standard_cost = 0
        for t in range(horizon):
            state = standard_trajectory[0, t, :2]
            control = standard_control_seq[0, t, :]
            standard_cost += pendulum_cost(state, control).item()
        standard_costs.append(standard_cost)
    
    # Print results
    print(f"\nResults over {num_test_episodes} episodes:")
    print(f"LMPPI average cost: {np.mean(lmppi_costs):.4f} ± {np.std(lmppi_costs):.4f}")
    print(f"Standard MPPI average cost: {np.mean(standard_costs):.4f} ± {np.std(standard_costs):.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    episodes = range(1, num_test_episodes + 1)
    plt.plot(episodes, lmppi_costs, 'o-', label='LMPPI', linewidth=2, markersize=6)
    plt.plot(episodes, standard_costs, 's-', label='Standard MPPI', linewidth=2, markersize=6)
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.title('LMPPI vs Standard MPPI Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('./lmppi_checkpoints/controller_comparison.png', dpi=300)
    plt.show()
    
    return lmppi_costs, standard_costs


def main():
    """Main execution function."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs("./lmppi_checkpoints", exist_ok=True)
    
    # Step 1: Generate demonstration data
    demo_trajectories = generate_demonstration_data(
        num_trajectories=1000,
        horizon=50,
        device=device
    )
    
    # Step 2: Train VAE model
    vae_model, trainer = train_vae_model(
        trajectories=demo_trajectories,
        state_dim=2,
        control_dim=1,
        latent_dim=8,
        device=device
    )
    
    # Step 3: Compare controllers
    lmppi_costs, standard_costs = compare_controllers(vae_model, device=device)
    
    print("\nLMPPI example completed successfully!")
    print("Check the './lmppi_checkpoints' directory for saved models and plots.")


if __name__ == "__main__":
    main()
