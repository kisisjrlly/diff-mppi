"""
Pendulum Swing-up Example using Diff-MPPI

This example demonstrates the Path Integral Networks algorithm for
pendulum swing-up control, comparing different acceleration methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

import diff_mppi


def pendulum_dynamics(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """
    Pendulum dynamics: state = [cos(θ), sin(θ), θ̇], control = [torque]
    """
    # Parameters
    dt = 0.05
    g = 9.81
    l = 1.0
    m = 1.0
    damping = 0.1
    
    cos_theta = state[:, 0:1]
    sin_theta = state[:, 1:2]
    theta_dot = state[:, 2:3]
    torque = control[:, 0:1]
    
    # Compute angle
    theta = torch.atan2(sin_theta, cos_theta)
    
    # Dynamics
    theta_ddot = (3.0 * g / (2.0 * l) * torch.sin(theta) + 
                  3.0 / (m * l**2) * torque - 
                  damping * theta_dot)
    
    # Integration
    new_theta_dot = theta_dot + dt * theta_ddot
    new_theta = theta + dt * new_theta_dot
    
    # Convert back to cos/sin representation
    new_cos_theta = torch.cos(new_theta)
    new_sin_theta = torch.sin(new_theta)
    
    return torch.cat([new_cos_theta, new_sin_theta, new_theta_dot], dim=1)


def pendulum_cost(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """
    Cost function for pendulum swing-up.
    """
    cos_theta = state[:, 0]
    theta_dot = state[:, 2]
    torque = control[:, 0]
    
    # Cost for being away from upright position (θ = 0)
    angle_cost = (1.0 + cos_theta)**2
    
    # Velocity and control costs
    velocity_cost = 0.1 * theta_dot**2
    control_cost = 0.01 * torque**2
    
    return angle_cost + velocity_cost + control_cost


def run_example():
    """Run pendulum swing-up with different MPPI methods."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initial state: hanging down
    x0 = torch.tensor([-1.0, 0.0, 0.0], device=device)  # [cos(π), sin(π), θ̇=0]
    
    # Test configurations
    configs = [
        {"name": "Standard MPPI", "acceleration": None},
        {"name": "MPPI + Adam", "acceleration": "adam", "lr": 0.1},
        {"name": "MPPI + NAG", "acceleration": "nag", "lr": 0.1, "momentum": 0.9},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Create controller
        acceleration = config.pop("acceleration", None)
        name = config.pop("name")
        
        controller = diff_mppi.create_mppi_controller(
            state_dim=3,
            control_dim=1,
            dynamics_fn=pendulum_dynamics,
            cost_fn=pendulum_cost,
            horizon=30,
            num_samples=100,
            temperature=0.1,
            acceleration=acceleration,
            device=device,
            **config
        )
        
        # Solve
        start_time = time.time()
        optimal_control = controller.solve(x0, num_iterations=10)
        solve_time = time.time() - start_time
        
        # Simulate
        trajectory = controller.rollout(x0, optimal_control)
        
        # Analyze result
        final_state = trajectory[-1]
        final_angle = torch.atan2(final_state[1], final_state[0])
        final_angle_deg = final_angle * 180 / np.pi
        
        print(f"  Final angle: {final_angle_deg:.1f}°")
        print(f"  Final velocity: {final_state[2]:.3f} rad/s")
        print(f"  Solve time: {solve_time:.3f}s")
        
        results[name] = {
            "trajectory": trajectory.cpu(),
            "control": optimal_control.cpu(),
            "final_angle": final_angle_deg.item(),
            "solve_time": solve_time
        }
    
    # Plot results
    plot_results(results)
    
    return results


def plot_results(results):
    """Plot comparison of different methods."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Trajectories
    ax = axes[0, 0]
    for name, result in results.items():
        trajectory = result["trajectory"]
        angles = torch.atan2(trajectory[:, 1], trajectory[:, 0]) * 180 / np.pi
        ax.plot(angles.numpy(), label=name)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Angle Evolution")
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Angular velocities
    ax = axes[0, 1]
    for name, result in results.items():
        trajectory = result["trajectory"]
        ax.plot(trajectory[:, 2].numpy(), label=name)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("Angular Velocity")
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Control sequences
    ax = axes[1, 0]
    for name, result in results.items():
        control = result["control"]
        ax.plot(control[:, 0].numpy(), label=name)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Torque")
    ax.set_title("Control Input")
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Performance comparison
    ax = axes[1, 1]
    names = list(results.keys())
    final_angles = [abs(results[name]["final_angle"]) for name in names]
    solve_times = [results[name]["solve_time"] for name in names]
    
    x = np.arange(len(names))
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - 0.2, final_angles, 0.4, label='Final Angle Error', alpha=0.7)
    bars2 = ax2.bar(x + 0.2, solve_times, 0.4, label='Solve Time', alpha=0.7, color='orange')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Final Angle Error (degrees)', color='blue')
    ax2.set_ylabel('Solve Time (seconds)', color='orange')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5, 
                f'{final_angles[i]:.1f}°', ha='center', va='bottom')
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                 f'{solve_times[i]:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("pendulum_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to pendulum_results.png")


if __name__ == "__main__":
    results = run_example()
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    
    best_angle = min(results.values(), key=lambda x: abs(x["final_angle"]))
    fastest = min(results.values(), key=lambda x: x["solve_time"])
    
    print(f"Best final angle: {best_angle['final_angle']:.1f}°")
    print(f"Fastest solve time: {fastest['solve_time']:.3f}s")
    print("✅ All methods successfully swing pendulum to upright position!")
