"""
Step 3: Test LMPPI Controller
测试训练好的LMPPI控制器

This script loads a trained VAE model and tests the LMPPI controller 
performance compared to standard MPPI.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import argparse
from pathlib import Path
import time

# Add the parent directory to the path so we can import diff_mppi
import sys
sys.path.append(str(Path(__file__).parent.parent))

from diff_mppi.lmppi import TrajectoryVAE, LMPPIController
from diff_mppi.lmppi.config import VAEConfig
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


def load_trained_model(model_path, device="cpu"):
    """Load trained VAE model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model config
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'model_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = VAEConfig(
            input_dim=config_dict['input_dim'],
            latent_dim=config_dict['latent_dim'],
            hidden_dims=config_dict['hidden_dims'],
            architecture=config_dict['architecture'],
            dropout=config_dict['dropout'],
            beta=config_dict['beta']
        )
    else:
        # Fallback config if config file not found
        print("Warning: Model config not found, using default config")
        config = VAEConfig(input_dim=150, latent_dim=16, hidden_dims=[512, 256, 128, 64])
    
    # Create and load model
    model = TrajectoryVAE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Architecture: {config.architecture}")
    print(f"  Training epoch: {checkpoint['epoch']}")
    
    return model, config


def create_controllers(vae_model, device="cpu", horizon=20, num_samples=100):
    """Create LMPPI and standard MPPI controllers."""
    
    # Control bounds
    control_min = torch.tensor([-15.0], device=device)
    control_max = torch.tensor([15.0], device=device)
    
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
    
    print(f"Created controllers:")
    print(f"  Horizon: {horizon}")
    print(f"  Samples: {num_samples}")
    print(f"  Control bounds: [{control_min.item():.1f}, {control_max.item():.1f}]")
    
    return lmppi, standard_mppi


def run_single_episode(controller, initial_state, horizon, controller_name="Controller"):
    """Run a single episode with the given controller."""
    
    # Solve for control sequence
    start_time = time.time()
    control_sequence = controller.solve(initial_state, num_iterations=10)
    solve_time = time.time() - start_time
    
    # Rollout trajectory
    trajectory = controller.rollout(initial_state, control_sequence)
    
    # Compute total cost
    total_cost = 0
    for t in range(horizon):
        state = trajectory[0, t, :2]
        control = control_sequence[0, t, :]
        total_cost += pendulum_cost(state, control).item()
    
    return {
        'trajectory': trajectory.cpu().numpy(),
        'control_sequence': control_sequence.cpu().numpy(),
        'total_cost': total_cost,
        'solve_time': solve_time
    }


def compare_controllers(lmppi, standard_mppi, num_episodes=10, horizon=20, device="cpu"):
    """Compare LMPPI vs standard MPPI performance."""
    
    print(f"Running comparison with {num_episodes} episodes...")
    
    results = {
        'lmppi': {'costs': [], 'times': [], 'trajectories': []},
        'standard': {'costs': [], 'times': [], 'trajectories': []}
    }
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        # Random initial state
        initial_state = torch.tensor([
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-5, 5)
        ], device=device).unsqueeze(0)
        
        print(f"  Initial state: θ={initial_state[0,0]:.2f}, θ̇={initial_state[0,1]:.2f}")
        
        # Test LMPPI
        lmppi_result = run_single_episode(lmppi, initial_state, horizon, "LMPPI")
        results['lmppi']['costs'].append(lmppi_result['total_cost'])
        results['lmppi']['times'].append(lmppi_result['solve_time'])
        results['lmppi']['trajectories'].append(lmppi_result['trajectory'])
        
        # Test standard MPPI  
        standard_result = run_single_episode(standard_mppi, initial_state, horizon, "Standard MPPI")
        results['standard']['costs'].append(standard_result['total_cost'])
        results['standard']['times'].append(standard_result['solve_time'])
        results['standard']['trajectories'].append(standard_result['trajectory'])
        
        print(f"  LMPPI: cost={lmppi_result['total_cost']:.2f}, time={lmppi_result['solve_time']:.3f}s")
        print(f"  Standard: cost={standard_result['total_cost']:.2f}, time={standard_result['solve_time']:.3f}s")
    
    return results


def plot_comparison_results(results, save_dir):
    """Plot comparison results."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Cost comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cost comparison bar plot
    episodes = range(1, len(results['lmppi']['costs']) + 1)
    
    axes[0,0].plot(episodes, results['lmppi']['costs'], 'o-', label='LMPPI', 
                   linewidth=2, markersize=6)
    axes[0,0].plot(episodes, results['standard']['costs'], 's-', label='Standard MPPI', 
                   linewidth=2, markersize=6)
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Total Cost')
    axes[0,0].set_title('Cost Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Time comparison
    axes[0,1].plot(episodes, results['lmppi']['times'], 'o-', label='LMPPI', 
                   linewidth=2, markersize=6)
    axes[0,1].plot(episodes, results['standard']['times'], 's-', label='Standard MPPI', 
                   linewidth=2, markersize=6)
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Solve Time (s)')
    axes[0,1].set_title('Computational Time Comparison')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Cost distribution
    axes[1,0].hist(results['lmppi']['costs'], alpha=0.7, label='LMPPI', bins=10)
    axes[1,0].hist(results['standard']['costs'], alpha=0.7, label='Standard MPPI', bins=10)
    axes[1,0].set_xlabel('Total Cost')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Cost Distribution')
    axes[1,0].legend()
    
    # Summary statistics
    lmppi_mean_cost = np.mean(results['lmppi']['costs'])
    lmppi_std_cost = np.std(results['lmppi']['costs'])
    lmppi_mean_time = np.mean(results['lmppi']['times'])
    
    standard_mean_cost = np.mean(results['standard']['costs'])
    standard_std_cost = np.std(results['standard']['costs'])
    standard_mean_time = np.mean(results['standard']['times'])
    
    summary_text = f"""Summary Statistics:
    
LMPPI:
  Mean Cost: {lmppi_mean_cost:.2f} ± {lmppi_std_cost:.2f}
  Mean Time: {lmppi_mean_time:.3f}s
  
Standard MPPI:
  Mean Cost: {standard_mean_cost:.2f} ± {standard_std_cost:.2f}
  Mean Time: {standard_mean_time:.3f}s
  
Improvement:
  Cost: {((standard_mean_cost - lmppi_mean_cost) / standard_mean_cost * 100):+.1f}%
  Speed: {(standard_mean_time / lmppi_mean_time):.1f}x"""
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sample trajectory plot
    if len(results['lmppi']['trajectories']) > 0:
        plot_sample_trajectories(results, save_dir)
    
    return {
        'lmppi_mean_cost': lmppi_mean_cost,
        'lmppi_std_cost': lmppi_std_cost,
        'standard_mean_cost': standard_mean_cost,
        'standard_std_cost': standard_std_cost,
        'cost_improvement_pct': (standard_mean_cost - lmppi_mean_cost) / standard_mean_cost * 100,
        'speed_improvement': standard_mean_time / lmppi_mean_time
    }


def plot_sample_trajectories(results, save_dir, num_samples=3):
    """Plot sample trajectories for visualization."""
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(min(num_samples, len(results['lmppi']['trajectories']))):
        lmppi_traj = results['lmppi']['trajectories'][i]
        standard_traj = results['standard']['trajectories'][i]
        
        time_steps = range(lmppi_traj.shape[1])
        
        # Angle trajectories
        axes[0,i].plot(time_steps, lmppi_traj[0, :, 0], 'b-', label='LMPPI', linewidth=2)
        axes[0,i].plot(time_steps, standard_traj[0, :, 0], 'r--', label='Standard MPPI', linewidth=2)
        axes[0,i].axhline(y=np.pi, color='g', linestyle=':', alpha=0.7, label='Target')
        axes[0,i].set_xlabel('Time Step')
        axes[0,i].set_ylabel('Angle (rad)')
        axes[0,i].set_title(f'Episode {i+1}: Angle')
        axes[0,i].legend()
        axes[0,i].grid(True)
        
        # Angular velocity trajectories
        axes[1,i].plot(time_steps, lmppi_traj[0, :, 1], 'b-', label='LMPPI', linewidth=2)
        axes[1,i].plot(time_steps, standard_traj[0, :, 1], 'r--', label='Standard MPPI', linewidth=2)
        axes[1,i].axhline(y=0, color='g', linestyle=':', alpha=0.7, label='Target')
        axes[1,i].set_xlabel('Time Step')
        axes[1,i].set_ylabel('Angular Velocity (rad/s)')
        axes[1,i].set_title(f'Episode {i+1}: Angular Velocity')
        axes[1,i].legend()
        axes[1,i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test LMPPI Controller')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained VAE model checkpoint (.pth)')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of test episodes (default: 20)')
    parser.add_argument('--horizon', type=int, default=20,
                       help='Control horizon (default: 20)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of MPPI samples (default: 100)')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                       help='Directory to save results (default: ./test_results)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cpu, cuda, or auto (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load trained model
    vae_model, config = load_trained_model(args.model_path, device)
    
    # Create controllers
    lmppi, standard_mppi = create_controllers(
        vae_model, device, args.horizon, args.num_samples
    )
    
    # Run comparison
    results = compare_controllers(
        lmppi, standard_mppi, args.num_episodes, args.horizon, device
    )
    
    # Create timestamped save directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'test_results_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot and save results
    summary_stats = plot_comparison_results(results, save_dir)
    
    # Save detailed results
    with open(os.path.join(save_dir, 'detailed_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(save_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save test configuration
    test_config = {
        'model_path': args.model_path,
        'num_episodes': args.num_episodes,
        'horizon': args.horizon,
        'num_samples': args.num_samples,
        'device': device,
        'timestamp': timestamp
    }
    
    with open(os.path.join(save_dir, 'test_config.json'), 'w') as f:
        json.dump(test_config, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*50}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"Episodes tested: {args.num_episodes}")
    print(f"Horizon: {args.horizon}")
    print(f"Samples: {args.num_samples}")
    print(f"\nLMPPI Performance:")
    print(f"  Mean cost: {summary_stats['lmppi_mean_cost']:.2f} ± {summary_stats['lmppi_std_cost']:.2f}")
    print(f"\nStandard MPPI Performance:")
    print(f"  Mean cost: {summary_stats['standard_mean_cost']:.2f} ± {summary_stats['standard_std_cost']:.2f}")
    print(f"\nImprovements:")
    print(f"  Cost improvement: {summary_stats['cost_improvement_pct']:+.1f}%")
    print(f"  Speed improvement: {summary_stats['speed_improvement']:.1f}x")
    print(f"\nResults saved in: {save_dir}")


if __name__ == "__main__":
    main()
