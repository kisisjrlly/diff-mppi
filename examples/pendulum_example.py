"""
Pendulum MPC Example using Diff-MPPI

This example demonstrates Model Predictive Control (MPC) using the Diff-MPPI
algorithm for real-time pendulum swing-up control with receding horizon.
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


def mpc_controller(
    initial_state: torch.Tensor,
    controller: diff_mppi.DiffMPPI,
    mpc_steps: int = 200,
    dt: float = 0.05,
    num_iterations: int = 100,
    verbose: bool = True
) -> dict:
    """
    Run MPC control loop.
    
    Args:
        initial_state: Starting state [cos(θ), sin(θ), θ̇]
        controller: MPPI controller instance
        mpc_steps: Number of MPC steps to simulate
        dt: Time step for MPC loop
        num_iterations: Number of MPPI iterations per MPC step
        verbose: Print progress
    
    Returns:
        Dictionary containing trajectory, controls, and metrics
    """
    device = initial_state.device
    current_state = initial_state.clone()
    
    # Storage for MPC trajectory
    states_history = [current_state.cpu().clone()]
    controls_history = []
    solve_times = []
    costs_history = []
    
    if verbose:
        print(f"Starting MPC control loop for {mpc_steps} steps...")
        print(f"Initial state: θ={torch.atan2(current_state[1], current_state[0])*180/np.pi:.1f}°, "
              f"θ̇={current_state[2]:.3f}")
    
    for step in range(mpc_steps):
        # Solve MPPI optimization at current state
        start_time = time.time()
        
        # Get optimal control sequence from MPPI
        optimal_control_sequence = controller.solve(
            current_state.unsqueeze(0), 
            num_iterations=num_iterations,
            verbose=False
        ).squeeze(0)
        
        solve_time = time.time() - start_time
        solve_times.append(solve_time)
        
        # MPC: Execute only the FIRST control action
        current_control = optimal_control_sequence[0:1, :]  # Shape: [1, control_dim]
        
        # Apply control to get next state
        current_state = pendulum_dynamics(
            current_state.unsqueeze(0), 
            current_control
        ).squeeze(0)
        
        # Calculate instantaneous cost
        step_cost = pendulum_cost(
            current_state.unsqueeze(0), 
            current_control
        ).item()
        costs_history.append(step_cost)
        
        # Store trajectory
        states_history.append(current_state.cpu().clone())
        controls_history.append(current_control.cpu().clone())
        
        # Progress reporting
        if verbose and (step + 1) % 50 == 0:
            current_angle = torch.atan2(current_state[1], current_state[0]) * 180 / np.pi
            print(f"Step {step+1}/{mpc_steps}: θ={current_angle:.1f}°, "
                  f"θ̇={current_state[2]:.3f}, cost={step_cost:.4f}, "
                  f"solve_time={solve_time:.4f}s")
        
        # Early termination check (successful stabilization)
        current_angle_rad = torch.atan2(current_state[1], current_state[0])
        angle_error = abs(current_angle_rad * 180 / np.pi)
        if angle_error < 5.0 and abs(current_state[2]) < 0.5:
            if verbose:
                print(f"Successfully stabilized at step {step+1}!")
            break
    
    # Convert to tensors
    states_trajectory = torch.stack(states_history)
    controls_trajectory = torch.stack(controls_history) if controls_history else torch.empty(0, 1)
    
    # Compute final metrics
    final_state = states_trajectory[-1]
    final_angle_rad = torch.atan2(final_state[1], final_state[0])
    final_angle_deg = final_angle_rad.item() * 180.0 / np.pi
    angle_error = (final_angle_deg + 180.0) % 360.0 - 180.0
    angle_error_abs = abs(angle_error)
    
    total_cost = sum(costs_history)
    avg_solve_time = np.mean(solve_times)
    is_successful = angle_error_abs < 10.0 and abs(final_state[2].item()) < 1.0
    
    return {
        "states": states_trajectory,
        "controls": controls_trajectory,
        "costs": costs_history,
        "solve_times": solve_times,
        "final_angle_deg": final_angle_deg,
        "angle_error_abs": angle_error_abs,
        "final_velocity": final_state[2].item(),
        "total_cost": total_cost,
        "avg_solve_time": avg_solve_time,
        "is_successful": is_successful,
        "steps_taken": len(states_history) - 1
    }


def run_example():
    """Run pendulum MPC with different MPPI methods."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initial state: hanging down
    x0 = torch.tensor([-1.0, 0.0, 0.0], device=device)  # [cos(π), sin(π), θ̇=0]
    
    # MPC configurations to test
    configs = [
        {"name": "MPC-Standard", "acceleration": None, "horizon": 40, "num_samples": 1000, "num_iterations": 10, "mpc_steps": 200},
        {"name": "MPC-Adam", "acceleration": "adam", "lr": 0.1, "horizon": 40, "num_samples": 1000, "num_iterations": 10, "mpc_steps": 200},
        {"name": "MPC-NAG", "acceleration": "nag", "lr": 0.1, "momentum": 0.9, "horizon": 40, "num_samples": 1000, "num_iterations": 10, "mpc_steps": 200},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"{'='*50}")
        
        # Create controller
        acceleration = config.pop("acceleration", None)
        name = config.pop("name")
        num_iterations = config.pop("num_iterations", 100)  # Extract num_iterations from config
        mpc_steps = config.pop("mpc_steps", 200)  # Extract mpc_steps from config
        
        controller = diff_mppi.create_mppi_controller(
            state_dim=3,
            control_dim=1,
            dynamics_fn=pendulum_dynamics,
            cost_fn=pendulum_cost,
            temperature=0.01,
            acceleration=acceleration,
            device=device,
            **config
        )
        
        # Run MPC
        start_time = time.time()
        mpc_result = mpc_controller(
            initial_state=x0,
            controller=controller,
            mpc_steps=mpc_steps,
            num_iterations=num_iterations,
            verbose=True
        )
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\nMPC Summary for {name}:")
        print(f"  Steps taken: {mpc_result['steps_taken']}")
        print(f"  Final angle: {mpc_result['final_angle_deg']:.1f}°")
        print(f"  Angle error: {mpc_result['angle_error_abs']:.1f}°")
        print(f"  Final velocity: {mpc_result['final_velocity']:.3f} rad/s")
        print(f"  Total cost: {mpc_result['total_cost']:.2f}")
        print(f"  Average solve time: {mpc_result['avg_solve_time']:.4f}s")
        print(f"  Total MPC time: {total_time:.2f}s")
        print(f"  Success: {'✅' if mpc_result['is_successful'] else '❌'}")
        
        # Convert to the old format for plotting compatibility
        trajectory = mpc_result["states"]
        control = mpc_result["controls"]
        
        results[name] = {
            "trajectory": trajectory,
            "control": control,
            "final_angle_raw": mpc_result["final_angle_deg"],
            "angle_error": mpc_result["final_angle_deg"],
            "angle_error_abs": mpc_result["angle_error_abs"],
            "final_velocity": mpc_result["final_velocity"],
            "total_cost": mpc_result["total_cost"],
            "convergence_costs": [mpc_result["total_cost"]],  # Simplified for plotting
            "is_successful": mpc_result["is_successful"],
            "solve_time": total_time,
            "avg_solve_time": mpc_result["avg_solve_time"],
            "steps_taken": mpc_result["steps_taken"]
        }
    
    # Plot results
    plot_results(results)
    
    return results


def plot_results(results):
    """Plot comparison of different MPC methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Angle trajectories (MPC closed-loop evolution)
    ax = axes[0, 0]
    for i, (name, result) in enumerate(results.items()):
        trajectory = result["trajectory"]
        angles = torch.atan2(trajectory[:, 1], trajectory[:, 0]) * 180 / np.pi
        time_vec = np.arange(len(angles)) * 0.05  # dt = 0.05
        ax.plot(time_vec, angles.numpy(), label=name, linewidth=2, color=colors[i % len(colors)])
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Target (0°)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("MPC: Angle Evolution (Closed-loop)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Angular velocities
    ax = axes[0, 1]
    for i, (name, result) in enumerate(results.items()):
        trajectory = result["trajectory"]
        time_vec = np.arange(len(trajectory)) * 0.05
        ax.plot(time_vec, trajectory[:, 2].numpy(), label=name, linewidth=2, color=colors[i % len(colors)])
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Target (0 rad/s)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("MPC: Angular Velocity Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Applied control sequences (step-by-step MPC actions)
    ax = axes[1, 0]
    for i, (name, result) in enumerate(results.items()):
        control = result["control"]
        if len(control) > 0:
            time_vec = np.arange(len(control)) * 0.05
            ax.plot(time_vec, control[:, 0].numpy(), label=name, linewidth=2, color=colors[i % len(colors)])
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Applied Torque")
    ax.set_title("MPC: Applied Control Actions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison
    ax = axes[1, 1]
    names = list(results.keys())
    angle_errors = [results[name]["angle_error_abs"] for name in names]
    avg_solve_times = [results[name]["avg_solve_time"] * 1000 for name in names]  # Convert to ms
    total_costs = [results[name]["total_cost"] for name in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    # Create bars for different metrics
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width, angle_errors, width, label='Angle Error (°)', alpha=0.8, color='red')
    bars2 = ax2.bar(x, avg_solve_times, width, label='Avg Solve Time (ms)', alpha=0.8, color='orange')
    bars3 = ax2.bar(x + width, [cost/100 for cost in total_costs], width, 
                    label='Total Cost (÷100)', alpha=0.8, color='green')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Angle Error (degrees)', color='red')
    ax2.set_ylabel('Time (ms) / Cost (÷100)', color='orange')
    ax.set_title('MPC Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    
    # Add value labels on bars
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5, 
                f'{angle_errors[i]:.1f}°', ha='center', va='bottom', fontsize=9)
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1, 
                 f'{avg_solve_times[i]:.1f}ms', ha='center', va='bottom', fontsize=9)
        ax2.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + 0.1, 
                 f'{total_costs[i]:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("pendulum_mpc_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nMPC results saved to pendulum_mpc_results.png")


if __name__ == "__main__":
    results = run_example()
    
    # Print summary
    print("\n" + "="*50)
    print("MPC SUMMARY:")
    print("="*50)
    
    # Calculate success rate
    successful_methods = [name for name, result in results.items() if result["is_successful"]]
    success_rate = len(successful_methods) / len(results) * 100
    
    # Find best performers
    best_accuracy = min(results.values(), key=lambda x: x["angle_error_abs"])
    fastest_avg = min(results.values(), key=lambda x: x["avg_solve_time"])
    lowest_cost = min(results.values(), key=lambda x: x["total_cost"])
    
    print(f"Success Rate: {success_rate:.0f}% ({len(successful_methods)}/{len(results)} methods)")
    print(f"Successful Methods: {', '.join(successful_methods) if successful_methods else 'None'}")
    print("")
    print(f"Best Accuracy: {best_accuracy['angle_error_abs']:.1f}° error")
    print(f"Fastest Average Solve: {fastest_avg['avg_solve_time']*1000:.1f}ms per step")
    print(f"Lowest Total Cost: {lowest_cost['total_cost']:.1f}")
    print("")
    
    # Method-wise performance
    print("Method Performance:")
    print("-" * 80)
    print(f"{'Method':<15} | {'Status':<6} | {'Steps':<5} | {'Error':<7} | {'AvgTime':<8} | {'Cost':<8}")
    print("-" * 80)
    for name, result in results.items():
        status = "✅" if result["is_successful"] else "❌"
        print(f"{name:<15} | {status:<6} | {result['steps_taken']:<5} | "
              f"{result['angle_error_abs']:5.1f}° | {result['avg_solve_time']*1000:6.1f}ms | "
              f"{result['total_cost']:6.1f}")
    
    if success_rate > 0:
        print(f"\n🎉 {success_rate:.0f}% of MPC methods successfully controlled the pendulum!")
        print("MPC demonstrates real-time receding horizon control with MPPI optimization.")
    else:
        print(f"\n⚠️  No MPC methods achieved the success criteria (≤10° error, ≤1 rad/s velocity)")
        print("Consider tuning MPC parameters: horizon, num_samples, temperature, or num_iterations.")