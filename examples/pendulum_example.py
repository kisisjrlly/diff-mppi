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
    dt = 0.016
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
    Target: upright position (θ = 0) with zero velocity.
    """
    cos_theta = state[:, 0]
    theta_dot = state[:, 2]
    torque = control[:, 0]
    
    # Cost for being away from upright position (θ = 0)
    # When θ = 0: cos(θ) = 1, cost = 0 (minimum)
    # When θ = π: cos(θ) = -1, cost = 4 (maximum)
    # Squared term provides stronger gradients near the target
    angle_cost = (1.0 - cos_theta)**2
    
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
            horizon=60,
            num_samples=100,
            temperature=0.001,
            acceleration=acceleration,
            device=device,
            **config
        )
        
        # Solve with convergence tracking
        start_time = time.time()
        
        # Track convergence by solving incrementally
        convergence_costs = []
        for iter_step in [2, 5, 10]:
            current_control = controller.solve(x0, num_iterations=iter_step)
            current_trajectory = controller.rollout(x0, current_control)
            
            # Calculate cost for this iteration
            iter_cost = 0.0
            for i in range(len(current_trajectory) - 1):
                state = current_trajectory[i].unsqueeze(0)
                control = current_control[i].unsqueeze(0)
                iter_cost += pendulum_cost(state, control).item()
            convergence_costs.append(iter_cost)
        
        # Final solve for the result
        optimal_control = controller.solve(x0, num_iterations=10)
        solve_time = time.time() - start_time
        
        # Simulate
        trajectory = controller.rollout(x0, optimal_control)
        
        # Analyze result
        final_state = trajectory[-1]
        
        # Raw angle calculation
        final_angle_rad = torch.atan2(final_state[1], final_state[0])
        final_angle_deg = final_angle_rad.item() * 180.0 / np.pi
        
        # Normalized angle error (minimum distance to target 0°)
        angle_error = (final_angle_deg + 180.0) % 360.0 - 180.0  # Normalize to (-180, 180]
        angle_error_abs = abs(angle_error)
        
        # Success criteria (within 10° of upright and low velocity)
        is_successful = angle_error_abs < 10.0 and abs(final_state[2].item()) < 1.0
        
        # Trajectory cost (total accumulated cost)
        total_cost = 0.0
        for i in range(len(trajectory) - 1):
            state = trajectory[i].unsqueeze(0)
            control = optimal_control[i].unsqueeze(0)
            total_cost += pendulum_cost(state, control).item()
        
        print(f"  Final angle (raw): {final_angle_deg:.1f}°")
        print(f"  Angle error: {angle_error:.1f}° (abs: {angle_error_abs:.1f}°)")
        print(f"  Final velocity: {final_state[2]:.3f} rad/s")
        print(f"  Total cost: {total_cost:.2f}")
        print(f"  Convergence: {convergence_costs[0]:.1f} → {convergence_costs[1]:.1f} → {convergence_costs[2]:.1f}")
        print(f"  Success: {'✅' if is_successful else '❌'}")
        print(f"  Solve time: {solve_time:.3f}s")
        
        results[name] = {
            "trajectory": trajectory.cpu(),
            "control": optimal_control.cpu(),
            "final_angle_raw": final_angle_deg,
            "angle_error": angle_error,
            "angle_error_abs": angle_error_abs,
            "final_velocity": final_state[2].item(),
            "total_cost": total_cost,
            "convergence_costs": convergence_costs,
            "is_successful": is_successful,
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
    angle_errors = [results[name]["angle_error_abs"] for name in names]
    solve_times = [results[name]["solve_time"] for name in names]
    total_costs = [results[name]["total_cost"] for name in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    # Create three bars: angle error, solve time (scaled), and total cost (scaled)
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width, angle_errors, width, label='Angle Error (°)', alpha=0.8, color='red')
    bars2 = ax2.bar(x, solve_times, width, label='Solve Time (s)', alpha=0.8, color='orange')
    bars3 = ax2.bar(x + width, [cost/100 for cost in total_costs], width, 
                    label='Total Cost (÷100)', alpha=0.8, color='green')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Angle Error (degrees)', color='red')
    ax2.set_ylabel('Time (s) / Cost (÷100)', color='orange')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5, 
                f'{angle_errors[i]:.1f}°', ha='center', va='bottom', fontsize=9)
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                 f'{solve_times[i]:.2f}s', ha='center', va='bottom', fontsize=9)
        ax2.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + 0.01, 
                 f'{total_costs[i]:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
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
    
    # Calculate success rate
    successful_methods = [name for name, result in results.items() if result["is_successful"]]
    success_rate = len(successful_methods) / len(results) * 100
    
    # Find best performers
    best_accuracy = min(results.values(), key=lambda x: x["angle_error_abs"])
    fastest = min(results.values(), key=lambda x: x["solve_time"])
    lowest_cost = min(results.values(), key=lambda x: x["total_cost"])
    
    print(f"Success Rate: {success_rate:.0f}% ({len(successful_methods)}/{len(results)} methods)")
    print(f"Successful Methods: {', '.join(successful_methods) if successful_methods else 'None'}")
    print("")
    print(f"Best Accuracy: {best_accuracy['angle_error_abs']:.1f}° error")
    print(f"Fastest Method: {fastest['solve_time']:.3f}s")
    print(f"Lowest Cost: {lowest_cost['total_cost']:.1f}")
    print("")
    
    # Method-wise performance
    for name, result in results.items():
        status = "✅" if result["is_successful"] else "❌"
        print(f"{status} {name:15} | Error: {result['angle_error_abs']:5.1f}° | "
              f"Time: {result['solve_time']:5.3f}s | Cost: {result['total_cost']:6.1f}")
    
    if success_rate > 0:
        print(f"\n🎉 {success_rate:.0f}% of methods successfully controlled the pendulum!")
    else:
        print(f"\n⚠️  No methods achieved the success criteria (≤10° error, ≤1 rad/s velocity)")
        print("Consider tuning parameters or increasing iterations.")
