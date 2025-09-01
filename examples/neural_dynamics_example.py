"""
Neural Dynamics Learning Example using Diff-MPPI

This example demonstrates learning a neural dynamics model and using it 
for control with diff-mppi. It shows the differentiable nature of the 
path integral networks.

Key concepts:
1. Learning neural dynamics from trajectory data
2. Using learned dynamics for optimal control
3. End-to-end training through the MPPI solver
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

import diff_mppi


class NeuralDynamics(nn.Module):
    """Simple neural network dynamics model."""
    
    def __init__(self, state_dim: int, control_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Predict next state."""
        x = torch.cat([state, control], dim=1)
        delta_state = self.net(x)
        return state + 0.05 * delta_state  # dt * state_derivative


def true_pendulum_dynamics(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Ground truth pendulum dynamics."""
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


def pendulum_cost(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Fixed cost function for pendulum swing-up."""
    cos_theta = state[:, 0]
    theta_dot = state[:, 2]
    torque = control[:, 0]
    
    angle_cost = (1.0 + cos_theta)**2
    velocity_cost = 0.1 * theta_dot**2
    control_cost = 0.01 * torque**2
    
    return angle_cost + velocity_cost + control_cost


def generate_training_data(num_trajectories: int = 20, device: str = "cpu"):
    """Generate training data using random controls."""
    print("Generating training data with random controls...")
    
    training_data = []
    
    for i in range(num_trajectories):
        # Random initial state
        initial_state = torch.tensor([
            -1.0 + 0.4 * torch.randn(1).item(),
            0.0 + 0.4 * torch.randn(1).item(),
            0.0 + 1.0 * torch.randn(1).item()
        ], device=device)
        
        # Random control sequence
        horizon = 20
        controls = 2.0 * torch.randn(horizon, 1, device=device)
        
        # Generate trajectory using true dynamics
        trajectory = [initial_state]
        state = initial_state.unsqueeze(0)
        
        for t in range(horizon):
            control = controls[t:t+1]
            next_state = true_pendulum_dynamics(state, control)
            trajectory.append(next_state.squeeze(0))
            state = next_state
        
        trajectory = torch.stack(trajectory)
        training_data.append((trajectory, controls))
    
    print(f"Generated {num_trajectories} training trajectories")
    return training_data


def train_neural_dynamics(
    dynamics_model: NeuralDynamics,
    training_data: list,
    num_epochs: int = 200,
    device: str = "cpu"
):
    """Train neural dynamics model."""
    print(f"\nTraining neural dynamics model for {num_epochs} epochs...")
    
    optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_samples = 0
        
        for trajectory, controls in training_data:
            optimizer.zero_grad()
            
            # Prepare training pairs
            current_states = trajectory[:-1]
            next_states_true = trajectory[1:]
            
            # Predict next states
            next_states_pred = dynamics_model(current_states, controls)
            
            # MSE loss
            loss = nn.MSELoss()(next_states_pred, next_states_true)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(current_states)
            num_samples += len(current_states)
        
        avg_loss = total_loss / num_samples
        losses.append(avg_loss)
        
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")
    
    return losses


def test_dynamics_accuracy(dynamics_model: NeuralDynamics, device: str = "cpu"):
    """Test learned dynamics accuracy."""
    print("\nTesting dynamics model accuracy...")
    
    # Test on a simple trajectory
    test_state = torch.tensor([[-1.0, 0.0, 0.0]], device=device)
    test_control = torch.tensor([[1.0]], device=device)
    
    # True next state
    true_next = true_pendulum_dynamics(test_state, test_control)
    
    # Predicted next state
    dynamics_model.eval()
    with torch.no_grad():
        pred_next = dynamics_model(test_state, test_control)
    
    error = torch.norm(true_next - pred_next).item()
    print(f"Single-step prediction error: {error:.6f}")
    
    return error


def compare_controllers(dynamics_model: NeuralDynamics, device: str = "cpu"):
    """Compare control performance using true vs learned dynamics."""
    print("\nComparing control performance...")
    
    initial_state = torch.tensor([-1.0, 0.0, 0.0], device=device)
    
    # Controller with true dynamics
    true_controller = diff_mppi.create_mppi_controller(
        state_dim=3,
        control_dim=1,
        dynamics_fn=true_pendulum_dynamics,
        cost_fn=pendulum_cost,
        horizon=25,
        num_samples=100,
        temperature=0.1,
        acceleration="adam",
        device=device
    )
    
    # Controller with learned dynamics
    learned_controller = diff_mppi.create_mppi_controller(
        state_dim=3,
        control_dim=1,
        dynamics_fn=dynamics_model,
        cost_fn=pendulum_cost,
        horizon=25,
        num_samples=100,
        temperature=0.1,
        acceleration="adam",
        device=device
    )
    
    # Solve with both controllers
    print("Solving with true dynamics...")
    start_time = time.time()
    true_controls = true_controller.solve(initial_state, num_iterations=10)
    true_time = time.time() - start_time
    
    print("Solving with learned dynamics...")
    start_time = time.time()
    learned_controls = learned_controller.solve(initial_state, num_iterations=10)
    learned_time = time.time() - start_time
    
    # Evaluate both solutions using TRUE dynamics
    true_trajectory = true_controller.rollout(initial_state, true_controls)
    learned_trajectory_evaluated = true_controller.rollout(initial_state, learned_controls)
    
    # Compute final angles
    true_final_angle = torch.atan2(true_trajectory[-1, 1], true_trajectory[-1, 0]) * 180 / np.pi
    learned_final_angle = torch.atan2(learned_trajectory_evaluated[-1, 1], learned_trajectory_evaluated[-1, 0]) * 180 / np.pi
    
    results = {
        "true_controller": {
            "final_angle": true_final_angle.item(),
            "solve_time": true_time,
            "trajectory": true_trajectory,
            "controls": true_controls
        },
        "learned_controller": {
            "final_angle": learned_final_angle.item(),
            "solve_time": learned_time,
            "trajectory": learned_trajectory_evaluated,
            "controls": learned_controls
        }
    }
    
    print(f"True dynamics final angle:    {true_final_angle:.1f}°")
    print(f"Learned dynamics final angle: {learned_final_angle:.1f}°")
    print(f"Angle difference: {abs(true_final_angle - learned_final_angle):.1f}°")
    
    return results


def plot_results(training_losses, dynamics_error, comparison_results):
    """Plot training and comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training loss
    ax = axes[0, 0]
    ax.plot(training_losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Dynamics Model Training")
    ax.grid(True)
    ax.set_yscale('log')
    
    # Control trajectories comparison
    ax = axes[0, 1]
    true_traj = comparison_results["true_controller"]["trajectory"].cpu()
    learned_traj = comparison_results["learned_controller"]["trajectory"].cpu()
    
    true_angles = torch.atan2(true_traj[:, 1], true_traj[:, 0]) * 180 / np.pi
    learned_angles = torch.atan2(learned_traj[:, 1], learned_traj[:, 0]) * 180 / np.pi
    
    ax.plot(true_angles.numpy(), 'b-', label='True Dynamics', linewidth=2)
    ax.plot(learned_angles.numpy(), 'r--', label='Learned Dynamics', linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Trajectory Comparison")
    ax.legend()
    ax.grid(True)
    
    # Control sequences
    ax = axes[1, 0]
    true_controls = comparison_results["true_controller"]["controls"].cpu()
    learned_controls = comparison_results["learned_controller"]["controls"].cpu()
    
    ax.plot(true_controls[:, 0].numpy(), 'b-', label='True Dynamics', linewidth=2)
    ax.plot(learned_controls[:, 0].numpy(), 'r--', label='Learned Dynamics', linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Control (Torque)")
    ax.set_title("Control Sequences")
    ax.legend()
    ax.grid(True)
    
    # Performance summary
    ax = axes[1, 1]
    categories = ['Final Angle Error', 'Solve Time Ratio']
    
    angle_error = abs(comparison_results["true_controller"]["final_angle"] - 
                     comparison_results["learned_controller"]["final_angle"])
    time_ratio = (comparison_results["learned_controller"]["solve_time"] / 
                  comparison_results["true_controller"]["solve_time"])
    
    values = [angle_error, time_ratio]
    colors = ['red' if angle_error > 10 else 'green', 'orange']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics")
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("neural_dynamics_results.png", dpi=300, bbox_inches='tight')
    plt.show()


def run_neural_dynamics_example():
    """Main function to run the neural dynamics learning example."""
    print("=== Neural Dynamics Learning with Diff-MPPI ===")
    print("Learning pendulum dynamics and using for optimal control\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Generate training data
    training_data = generate_training_data(num_trajectories=30, device=device)
    
    # Initialize and train neural dynamics
    dynamics_model = NeuralDynamics(state_dim=3, control_dim=1, hidden_dim=64).to(device)
    training_losses = train_neural_dynamics(dynamics_model, training_data, num_epochs=300, device=device)
    
    # Test dynamics accuracy
    dynamics_error = test_dynamics_accuracy(dynamics_model, device=device)
    
    # Compare controllers
    comparison_results = compare_controllers(dynamics_model, device=device)
    
    # Plot results
    plot_results(training_losses, dynamics_error, comparison_results)
    
    # Print summary
    print("\n" + "="*60)
    print("NEURAL DYNAMICS LEARNING RESULTS:")
    print("="*60)
    
    final_loss = training_losses[-1]
    angle_diff = abs(comparison_results["true_controller"]["final_angle"] - 
                    comparison_results["learned_controller"]["final_angle"])
    
    print(f"Final training loss:        {final_loss:.6f}")
    print(f"Single-step error:          {dynamics_error:.6f}")
    print(f"Control angle difference:   {angle_diff:.1f}°")
    
    if angle_diff < 15.0:
        print("✅ Successfully learned dynamics for control!")
    else:
        print("⚠️  Dynamics learning needs improvement")
    
    print(f"\nResults saved to: neural_dynamics_results.png")
    
    return {
        "dynamics_model": dynamics_model,
        "training_losses": training_losses,
        "comparison_results": comparison_results
    }


if __name__ == "__main__":
    results = run_neural_dynamics_example()
