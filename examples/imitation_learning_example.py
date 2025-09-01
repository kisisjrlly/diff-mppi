"""
Imitation Learning Example using Diff-MPPI

This example demonstrates the core PI-Net concept: end-to-end differentiable learning
of both dynamics and cost models from expert demonstrations using inverse optimal control.

Based on:
- "Path Integral Networks: End-to-End Differentiable Optimal Control" (Okada et al., 2017)
- "Acceleration of Gradient-Based Path Integral Method" (Okada & Taniguchi, 2018)

Key concepts demonstrated:
1. Learning neural dynamics model from expert trajectories (supervised learning)
2. Learning neural cost function from expert behavior (inverse optimal control)
3. End-to-end differentiable optimization through the MPPI solver
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple

import diff_mppi


class NeuralDynamics(nn.Module):
    """Learnable dynamics model."""
    
    def __init__(self, state_dim: int, control_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict next state given current state and control."""
        x = torch.cat([state, control], dim=1)
        delta_state = self.net(x)
        return state + delta_state  # Residual connection for stability


class NeuralCost(nn.Module):
    """Learnable cost function."""
    
    def __init__(self, state_dim: int, control_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Compute cost for state-control pairs."""
        x = torch.cat([state, control], dim=1)
        cost = self.net(x).squeeze(-1)
        return torch.clamp(cost, min=0.0)  # Ensure non-negative costs


def true_pendulum_dynamics(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Ground truth pendulum dynamics for generating expert data."""
    dt = 0.05
    g, l, m = 9.81, 1.0, 1.0
    damping = 0.1
    
    cos_theta = state[:, 0:1]
    sin_theta = state[:, 1:2]
    theta_dot = state[:, 2:3]
    torque = control[:, 0:1]
    
    theta = torch.atan2(sin_theta, cos_theta)
    theta_ddot = (3.0 * g / (2.0 * l) * torch.sin(theta) + 
                  3.0 / (m * l**2) * torque - 
                  damping * theta_dot)
    
    new_theta_dot = theta_dot + dt * theta_ddot
    new_theta = theta + dt * new_theta_dot
    
    new_cos_theta = torch.cos(new_theta)
    new_sin_theta = torch.sin(new_theta)
    
    return torch.cat([new_cos_theta, new_sin_theta, new_theta_dot], dim=1)


def true_pendulum_cost(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Ground truth cost function for generating expert data."""
    cos_theta = state[:, 0]
    theta_dot = state[:, 2]
    torque = control[:, 0]
    
    # Cost components
    angle_cost = (1.0 + cos_theta)**2
    velocity_cost = 0.1 * theta_dot**2
    control_cost = 0.01 * torque**2
    
    return angle_cost + velocity_cost + control_cost


def generate_expert_data(num_trajectories: int = 5, device: str = "cpu") -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate expert demonstration data using true dynamics and cost."""
    print("Generating expert demonstration data...")
    
    # Create expert controller with true dynamics and cost
    expert_controller = diff_mppi.create_mppi_controller(
        state_dim=3,
        control_dim=1,
        dynamics_fn=true_pendulum_dynamics,
        cost_fn=true_pendulum_cost,
        horizon=25,
        num_samples=150,
        temperature=0.1,
        acceleration="adam",
        lr=0.05,
        device=device
    )
    
    expert_trajectories = []
    
    for i in range(num_trajectories):
        # Random initial states (hanging down with small perturbations)
        initial_state = torch.tensor([
            -1.0 + 0.2 * torch.randn(1).item(),  # cos(θ) near -1
            0.0 + 0.2 * torch.randn(1).item(),   # sin(θ) near 0
            0.0 + 0.5 * torch.randn(1).item()    # θ̇ small random
        ], device=device)
        
        # Generate expert control sequence
        expert_controls = expert_controller.solve(initial_state, num_iterations=12)
        
        # Generate corresponding trajectory
        expert_trajectory = expert_controller.rollout(initial_state, expert_controls)
        
        expert_trajectories.append((expert_trajectory, expert_controls))
        
        final_angle = torch.atan2(expert_trajectory[-1, 1], expert_trajectory[-1, 0]) * 180 / np.pi
        print(f"  Expert trajectory {i+1}: final angle = {final_angle:.1f}°")
    
    return expert_trajectories


def train_dynamics_model(
    dynamics_model: NeuralDynamics, 
    optimizer: optim.Optimizer,
    expert_trajectories: List[Tuple[torch.Tensor, torch.Tensor]]
) -> float:
    """Train dynamics model using supervised learning on expert data."""
    dynamics_model.train()
    total_loss = 0.0
    num_samples = 0
    
    for trajectory, controls in expert_trajectories:
        optimizer.zero_grad()
        
        # Prepare data: states[:-1] -> states[1:]
        current_states = trajectory[:-1]  # [T, state_dim]
        next_states_true = trajectory[1:]  # [T, state_dim]
        
        # Predict next states
        next_states_pred = dynamics_model(current_states, controls)
        
        # MSE loss
        loss = nn.MSELoss()(next_states_pred, next_states_true)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(current_states)
        num_samples += len(current_states)
    
    return total_loss / num_samples


def train_cost_model(
    cost_model: NeuralCost,
    optimizer: optim.Optimizer,
    dynamics_model: NeuralDynamics,
    expert_trajectories: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str = "cpu"
) -> float:
    """Train cost model using inverse optimal control loss."""
    cost_model.train()
    dynamics_model.eval()  # Keep dynamics frozen during cost training
    
    total_loss = 0.0
    num_trajectories = 0
    
    for expert_trajectory, expert_controls in expert_trajectories:
        optimizer.zero_grad()
        
        initial_state = expert_trajectory[0]
        horizon = len(expert_controls)
        
        # Create temporary MPPI controller with current models
        temp_controller = diff_mppi.create_mppi_controller(
            state_dim=3,
            control_dim=1,
            dynamics_fn=dynamics_model,
            cost_fn=cost_model,
            horizon=horizon,
            num_samples=50,  # Fewer samples for faster training
            temperature=0.2,
            device=device
        )
        
        # Get MPPI solution with current cost model
        mppi_controls = temp_controller.solve(initial_state, num_iterations=3)
        
        # Compute trajectory costs
        expert_cost = compute_trajectory_cost(cost_model, expert_trajectory, expert_controls)
        
        # Rollout MPPI trajectory and compute its cost
        mppi_trajectory = temp_controller.rollout(initial_state, mppi_controls)
        mppi_cost = compute_trajectory_cost(cost_model, mppi_trajectory, mppi_controls)
        
        # Inverse optimal control loss: expert should have lower cost
        # Margin loss encourages expert cost to be lower than MPPI cost by margin
        margin = 1.0
        loss = torch.relu(expert_cost - mppi_cost + margin)
        
        if loss.item() > 0:  # Only backprop if there's a violation
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_trajectories += 1
    
    return total_loss / num_trajectories


def compute_trajectory_cost(
    cost_model: NeuralCost, 
    trajectory: torch.Tensor, 
    controls: torch.Tensor
) -> torch.Tensor:
    """Compute total cost for a trajectory."""
    states = trajectory[:-1]  # Exclude final state (no control applied)
    step_costs = cost_model(states, controls)
    return torch.sum(step_costs)


def evaluate_learned_models(
    dynamics_model: NeuralDynamics,
    cost_model: NeuralCost,
    expert_trajectories: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str = "cpu"
) -> dict:
    """Evaluate learned models by comparing with expert performance."""
    print("\nEvaluating learned models...")
    
    dynamics_model.eval()
    cost_model.eval()
    
    # Create controller with learned models
    learned_controller = diff_mppi.create_mppi_controller(
        state_dim=3,
        control_dim=1,
        dynamics_fn=dynamics_model,
        cost_fn=cost_model,
        horizon=25,
        num_samples=100,
        temperature=0.1,
        acceleration="adam",
        device=device
    )
    
    results = {
        "expert_final_angles": [],
        "learned_final_angles": [],
        "expert_costs": [],
        "learned_costs": []
    }
    
    for i, (expert_traj, expert_controls) in enumerate(expert_trajectories):
        initial_state = expert_traj[0]
        
        # Expert performance
        expert_final_angle = torch.atan2(expert_traj[-1, 1], expert_traj[-1, 0]) * 180 / np.pi
        expert_cost = compute_trajectory_cost(cost_model, expert_traj, expert_controls)
        
        # Learned model performance
        learned_controls = learned_controller.solve(initial_state, num_iterations=10)
        learned_traj = learned_controller.rollout(initial_state, learned_controls)
        learned_final_angle = torch.atan2(learned_traj[-1, 1], learned_traj[-1, 0]) * 180 / np.pi
        learned_cost = compute_trajectory_cost(cost_model, learned_traj, learned_controls)
        
        results["expert_final_angles"].append(expert_final_angle.item())
        results["learned_final_angles"].append(learned_final_angle.item())
        results["expert_costs"].append(expert_cost.item())
        results["learned_costs"].append(learned_cost.item())
        
        print(f"  Trajectory {i+1}: Expert {expert_final_angle:.1f}° | Learned {learned_final_angle:.1f}°")
    
    return results


def plot_learning_progress(dynamics_losses: List[float], cost_losses: List[float]):
    """Plot training progress."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Dynamics loss
    ax1.plot(dynamics_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dynamics MSE Loss")
    ax1.set_title("Dynamics Model Learning")
    ax1.grid(True)
    
    # Cost loss
    ax2.plot(cost_losses)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Inverse Optimal Control Loss")
    ax2.set_title("Cost Model Learning")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("imitation_learning_progress.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparison_results(results: dict, expert_trajectories: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Plot comparison between expert and learned behaviors."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Final angle comparison
    ax = axes[0, 0]
    x = np.arange(len(results["expert_final_angles"]))
    width = 0.35
    ax.bar(x - width/2, results["expert_final_angles"], width, label='Expert', alpha=0.7)
    ax.bar(x + width/2, results["learned_final_angles"], width, label='Learned', alpha=0.7)
    ax.set_xlabel("Trajectory")
    ax.set_ylabel("Final Angle (degrees)")
    ax.set_title("Final Angle Comparison")
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Cost comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, results["expert_costs"], width, label='Expert', alpha=0.7)
    ax.bar(x + width/2, results["learned_costs"], width, label='Learned', alpha=0.7)
    ax.set_xlabel("Trajectory")
    ax.set_ylabel("Total Cost")
    ax.set_title("Trajectory Cost Comparison")
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Example trajectory comparison
    ax = axes[1, 0]
    expert_traj = expert_trajectories[0][0].cpu()
    expert_angles = torch.atan2(expert_traj[:, 1], expert_traj[:, 0]) * 180 / np.pi
    ax.plot(expert_angles.numpy(), 'b-', label='Expert', linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Example Trajectory")
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Learning statistics
    ax = axes[1, 1]
    expert_mean = np.mean(results["expert_final_angles"])
    learned_mean = np.mean(results["learned_final_angles"])
    expert_std = np.std(results["expert_final_angles"])
    learned_std = np.std(results["learned_final_angles"])
    
    categories = ['Expert', 'Learned']
    means = [expert_mean, learned_mean]
    stds = [expert_std, learned_std]
    
    ax.bar(categories, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel("Final Angle (degrees)")
    ax.set_title("Performance Statistics")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("imitation_learning_results.png", dpi=300, bbox_inches='tight')
    plt.show()


def run_imitation_learning_example():
    """Main function to run the complete imitation learning example."""
    print("=== Diff-MPPI Imitation Learning Example ===")
    print("Demonstrating end-to-end differentiable learning of dynamics and cost models\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Generate expert demonstration data
    expert_trajectories = generate_expert_data(num_trajectories=8, device=device)
    
    # Initialize neural models
    dynamics_model = NeuralDynamics(state_dim=3, control_dim=1, hidden_dim=64).to(device)
    cost_model = NeuralCost(state_dim=3, control_dim=1, hidden_dim=32).to(device)
    
    # Initialize optimizers
    dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)
    cost_optimizer = optim.Adam(cost_model.parameters(), lr=5e-4)
    
    # Training parameters
    num_epochs = 100
    dynamics_losses = []
    cost_losses = []
    
    print(f"\nTraining models for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train dynamics model (supervised learning)
        dynamics_loss = train_dynamics_model(dynamics_model, dynamics_optimizer, expert_trajectories)
        dynamics_losses.append(dynamics_loss)
        
        # Train cost model (inverse optimal control)
        cost_loss = train_cost_model(cost_model, cost_optimizer, dynamics_model, expert_trajectories, device)
        cost_losses.append(cost_loss)
        
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: Dynamics Loss = {dynamics_loss:.6f}, Cost Loss = {cost_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate learned models
    results = evaluate_learned_models(dynamics_model, cost_model, expert_trajectories, device)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("IMITATION LEARNING RESULTS:")
    print("="*60)
    
    expert_mean = np.mean(results["expert_final_angles"])
    learned_mean = np.mean(results["learned_final_angles"])
    angle_diff = abs(expert_mean - learned_mean)
    
    print(f"Expert average final angle:  {expert_mean:.1f}°")
    print(f"Learned average final angle: {learned_mean:.1f}°")
    print(f"Average angle difference:    {angle_diff:.1f}°")
    
    cost_ratio = np.mean(results["learned_costs"]) / np.mean(results["expert_costs"])
    print(f"Cost ratio (learned/expert): {cost_ratio:.2f}")
    
    if angle_diff < 10.0:
        print("✅ Successfully learned to imitate expert behavior!")
    else:
        print("⚠️  Learning partially successful - may need more training epochs")
    
    # Generate plots
    plot_learning_progress(dynamics_losses, cost_losses)
    plot_comparison_results(results, expert_trajectories)
    
    print(f"\nResults saved to:")
    print("- imitation_learning_progress.png")
    print("- imitation_learning_results.png")
    
    return {
        "dynamics_model": dynamics_model,
        "cost_model": cost_model,
        "training_losses": {"dynamics": dynamics_losses, "cost": cost_losses},
        "evaluation_results": results
    }


if __name__ == "__main__":
    results = run_imitation_learning_example()
