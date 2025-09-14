"""
Fixed LMPPI Demo: Simple demonstration with correct data handling

This demonstrates LMPPI with proper data shape handling.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from diff_mppi.lmppi.models import TrajectoryVAE
from diff_mppi.lmppi.controller import LMPPIController
from diff_mppi.lmppi.config import VAEConfig


def create_training_data(num_samples=100, horizon=10, state_dim=2, control_dim=1):
    """Create simple training data for VAE."""
    trajectories = []
    feature_dim = state_dim + control_dim
    
    for _ in range(num_samples):
        # Create random trajectory
        traj = torch.randn(horizon, feature_dim) * 0.5
        # Flatten to 1D for VAE
        traj_flat = traj.flatten()
        trajectories.append(traj_flat)
    
    return torch.stack(trajectories)


def simple_dynamics(state, control):
    """Simple linear dynamics"""
    A = torch.eye(2) + 0.1 * torch.randn(2, 2)
    B = torch.randn(2, 1) * 0.1
    return torch.matmul(state, A.T) + torch.matmul(control, B.T)


def simple_cost(state, control):
    """Simple quadratic cost"""
    state_cost = torch.sum(state**2, dim=-1)
    control_cost = 0.1 * torch.sum(control**2, dim=-1)
    return state_cost + control_cost


def main():
    print("Fixed LMPPI Demo")
    print("=" * 20)
    
    # Parameters
    state_dim = 2
    control_dim = 1
    horizon = 10
    latent_dim = 4
    
    # Step 1: Create training data
    print("1. Creating training data...")
    input_dim = horizon * (state_dim + control_dim)  # Flattened trajectory dimension
    train_data = create_training_data(100, horizon, state_dim, control_dim)
    print(f"   Data shape: {train_data.shape}")
    
    # Step 2: Create and train VAE
    print("2. Creating VAE...")
    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[32, 16],
        architecture="mlp"
    )
    vae = TrajectoryVAE(config)
    print(f"   VAE: {input_dim} -> {latent_dim} -> {input_dim}")
    
    # Simple training loop
    print("3. Training VAE...")
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    for epoch in range(50):
        # Forward pass
        loss, recon_loss, kl_loss = vae.compute_loss(train_data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss={loss:.4f}, Recon={recon_loss:.4f}, KL={kl_loss:.4f}")
    
    # Step 4: Create controller
    print("4. Creating LMPPI Controller...")
    controller = LMPPIController(
        vae_model=vae,
        state_dim=state_dim,
        control_dim=control_dim,
        cost_fn=simple_cost,
        horizon=horizon,
        num_samples=20,
        temperature=0.1,
        device='cpu'
    )
    print("   Controller created successfully")
    
    # Step 5: Test control
    print("5. Testing control...")
    initial_state = torch.tensor([1.0, -0.5])
    
    try:
        # Get control action
        action = controller.step(initial_state)
        print(f"   Control action: {action}")
        print("   ✓ LMPPI control successful!")
    except Exception as e:
        print(f"   ✗ Control failed: {e}")
    
    print("\n🎉 LMPPI demo completed!")


if __name__ == "__main__":
    main()
