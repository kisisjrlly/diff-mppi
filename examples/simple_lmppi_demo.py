"""
Simple LMPPI Demo: Quadratic System

A minimal example demonstrating LMPPI on a simple quadratic system.
This shows the core workflow without complex dynamics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from diff_mppi.lmppi import *
from diff_mppi.lmppi.data import create_synthetic_trajectories
from diff_mppi.lmppi.config import create_custom_config


def simple_dynamics(state, control):
    """Simple linear dynamics: x[t+1] = A*x[t] + B*u[t]"""
    A = torch.tensor([[1.0, 0.1], [0.0, 0.9]])
    B = torch.tensor([[0.0], [0.1]])
    
    # Apply dynamics
    next_state = torch.matmul(state, A.T) + torch.matmul(control, B.T)
    return next_state


def quadratic_cost(state, control):
    """Quadratic cost function"""
    Q = torch.tensor([[1.0, 0.0], [0.0, 0.1]])
    R = torch.tensor([[0.1]])
    
    state_cost = torch.sum(state * torch.matmul(state, Q), dim=1)
    control_cost = torch.sum(control * torch.matmul(control, R), dim=1)
    
    return state_cost + control_cost


def main():
    """Run simple LMPPI demo."""
    print("Simple LMPPI Demo")
    print("=" * 30)
    
    # System parameters
    state_dim = 2
    control_dim = 1
    horizon = 15
    
    # Step 1: Generate synthetic training data
    print("1. Generating training data...")
    dataset = create_synthetic_trajectories(
        num_trajectories=200,
        horizon=horizon,
        state_dim=state_dim,
        control_dim=control_dim,
        trajectory_types=["linear", "sinusoidal"],
        noise_std=0.05
    )
    print(f"   Created {len(dataset)} trajectories")
    
    # Step 2: Create and configure VAE
    print("2. Setting up VAE model...")
    config = create_custom_config(
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=horizon,
        latent_dim=6,
        hidden_dims=[64, 32, 16]
    )
    
    vae_model = TrajectoryVAE(config.vae)
    print(f"   VAE: {config.vae.input_dim} -> {config.vae.latent_dim} -> {config.vae.input_dim}")
    
    # Step 3: Train VAE (quick training for demo)
    print("3. Training VAE...")
    train_data, val_data = dataset.split(train_ratio=0.8)
    
    trainer = LMPPITrainer(
        model=vae_model,
        train_dataset=train_data,
        val_dataset=val_data,
        batch_size=32,
        learning_rate=1e-3,
        save_dir="./simple_lmppi_demo"
    )
    
    # Quick training for demo
    trainer.train(num_epochs=20)
    print("   Training completed!")
    
    # Step 4: Test LMPPI control
    print("4. Testing LMPPI control...")
    controller = LMPPIController(
        vae_model=vae_model,
        state_dim=state_dim,
        control_dim=control_dim,
        cost_fn=quadratic_cost,
        horizon=horizon,
        num_samples=50,
        temperature=1.0
    )
    
    # Test control
    initial_state = torch.tensor([[2.0, -1.0]])  # Start away from origin
    print(f"   Initial state: {initial_state.numpy().flatten()}")
    
    # Single control step
    control_action = controller.step(initial_state)
    print(f"   Control action: {control_action.numpy().flatten()}")
    
    # Simulate trajectory
    trajectory = controller.rollout(initial_state)
    states = trajectory[:, :, :state_dim].squeeze().numpy()
    controls = trajectory[:, :, state_dim:].squeeze().numpy()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(states[:, 0], label='x1')
    plt.plot(states[:, 1], label='x2')
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.title('State Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(controls)
    plt.xlabel('Time Step')
    plt.ylabel('Control')
    plt.title('Control Input')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(states[:, 0], states[:, 1], 'b-o', markersize=3)
    plt.plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
    plt.plot(states[-1, 0], states[-1, 1], 'ro', markersize=8, label='End')
    plt.plot(0, 0, 'k*', markersize=10, label='Target')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('State Space Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_lmppi_demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("5. Demo completed!")
    print(f"   Final state: {states[-1]}")
    print(f"   Distance to origin: {np.linalg.norm(states[-1]):.3f}")
    print("\nResults saved to: simple_lmppi_demo_results.png")


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
