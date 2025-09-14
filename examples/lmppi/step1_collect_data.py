"""
Step 1: Data Collection for LMPPI
收集大量轨迹数据并保存到本地磁盘

This script generates and saves large amounts of trajectory data for LMPPI training.
You can run this multiple times to collect extensive datasets.
"""

import torch
import numpy as np
import pickle
import os
import argparse
from pathlib import Path
import time
from tqdm import tqdm

# Add the parent directory to the path so we can import diff_mppi
import sys
sys.path.append(str(Path(__file__).parent.parent))


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


def generate_pd_trajectory(horizon, device="cpu"):
    """Generate a single trajectory using PD controller with noise."""
    # Random initial state
    theta_0 = np.random.uniform(-np.pi, np.pi)
    theta_dot_0 = np.random.uniform(-5, 5)
    
    state = torch.tensor([theta_0, theta_dot_0], device=device, dtype=torch.float32)
    
    # Storage for trajectory
    states = [state.clone()]
    controls = []
    
    # Random PD controller gains for diversity
    kp = np.random.uniform(5, 40)
    kd = np.random.uniform(0.5, 8)
    
    # Random noise level
    noise_level = np.random.uniform(0.1, 1.0)
    
    for t in range(horizon):
        # Current state
        theta, theta_dot = state[0], state[1]
        
        # Target is upright
        target_theta = np.pi
        
        # PD control with random variations
        angle_error = torch.atan2(torch.sin(theta - target_theta), torch.cos(theta - target_theta))
        velocity_error = -theta_dot  # Target velocity is 0
        
        torque = kp * angle_error + kd * velocity_error
        
        # Add noise for diversity
        torque = torque + torch.randn(1, device=device).squeeze() * noise_level
        
        # Clamp control
        torque = torch.clamp(torque, -15, 15)
        
        control = torch.tensor([torque], device=device)
        controls.append(control.clone())
        
        # Dynamics
        next_state = pendulum_dynamics(state.unsqueeze(0), control.unsqueeze(0)).squeeze(0)
        states.append(next_state.clone())
        state = next_state
    
    # Convert to trajectory format [horizon, state_dim + control_dim]
    states_tensor = torch.stack(states[:-1])  # Remove last state to match control length
    controls_tensor = torch.stack(controls)
    trajectory = torch.cat([states_tensor, controls_tensor], dim=1)
    
    return trajectory.cpu().numpy()


def generate_random_trajectory(horizon, device="cpu"):
    """Generate a random trajectory for exploration."""
    # Random initial state
    theta_0 = np.random.uniform(-np.pi, np.pi)
    theta_dot_0 = np.random.uniform(-8, 8)
    
    state = torch.tensor([theta_0, theta_dot_0], device=device, dtype=torch.float32)
    
    # Storage for trajectory
    states = [state.clone()]
    controls = []
    
    for t in range(horizon):
        # Random control with some smoothness
        if t == 0:
            torque = torch.randn(1, device=device) * 5
        else:
            # Add some correlation with previous control for smoothness
            prev_torque = controls[-1]
            torque = 0.7 * prev_torque + 0.3 * torch.randn(1, device=device) * 5
        
        # Clamp control
        torque = torch.clamp(torque, -15, 15)
        
        control = torque
        controls.append(control.clone())
        
        # Dynamics
        next_state = pendulum_dynamics(state.unsqueeze(0), control.unsqueeze(0)).squeeze(0)
        states.append(next_state.clone())
        state = next_state
    
    # Convert to trajectory format [horizon, state_dim + control_dim]
    states_tensor = torch.stack(states[:-1])  # Remove last state to match control length
    controls_tensor = torch.stack(controls)
    trajectory = torch.cat([states_tensor, controls_tensor], dim=1)
    
    return trajectory.cpu().numpy()


def collect_trajectories(num_trajectories, horizon, device="cpu", strategy_mix=None):
    """
    Collect trajectories using different strategies.
    
    Args:
        num_trajectories: Total number of trajectories to collect
        horizon: Length of each trajectory
        device: Computing device
        strategy_mix: Dict with strategy ratios, e.g., {'pd': 0.7, 'random': 0.3}
    """
    if strategy_mix is None:
        strategy_mix = {'pd': 0.8, 'random': 0.2}
    
    print(f"Collecting {num_trajectories} trajectories...")
    print(f"Strategy mix: {strategy_mix}")
    
    trajectories = []
    
    # Calculate number of trajectories for each strategy
    num_pd = int(num_trajectories * strategy_mix.get('pd', 0))
    num_random = int(num_trajectories * strategy_mix.get('random', 0))
    
    # Ensure we get exactly the requested number
    remaining = num_trajectories - num_pd - num_random
    num_pd += remaining
    
    print(f"Generating {num_pd} PD trajectories and {num_random} random trajectories...")
    
    # Generate PD trajectories
    for i in tqdm(range(num_pd), desc="PD trajectories"):
        trajectory = generate_pd_trajectory(horizon, device)
        trajectories.append(trajectory)
    
    # Generate random trajectories
    for i in tqdm(range(num_random), desc="Random trajectories"):
        trajectory = generate_random_trajectory(horizon, device)
        trajectories.append(trajectory)
    
    # Shuffle trajectories
    np.random.shuffle(trajectories)
    
    return trajectories


def save_dataset(trajectories, save_dir, filename_prefix="pendulum_data"):
    """Save trajectory dataset with metadata."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create metadata
    metadata = {
        'num_trajectories': len(trajectories),
        'horizon': trajectories[0].shape[0],
        'state_dim': 2,
        'control_dim': 1,
        'feature_dim': trajectories[0].shape[1],
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'description': 'Pendulum swing-up trajectories generated using PD and random controllers'
    }
    
    # Save trajectories
    timestamp = metadata['timestamp']
    traj_filename = f"{filename_prefix}_{timestamp}.pkl"
    meta_filename = f"{filename_prefix}_{timestamp}_metadata.pkl"
    
    traj_path = os.path.join(save_dir, traj_filename)
    meta_path = os.path.join(save_dir, meta_filename)
    
    with open(traj_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Saved {len(trajectories)} trajectories to:")
    print(f"  Data: {traj_path}")
    print(f"  Metadata: {meta_path}")
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Number of trajectories: {metadata['num_trajectories']}")
    print(f"  Trajectory length: {metadata['horizon']}")
    print(f"  State dimension: {metadata['state_dim']}")
    print(f"  Control dimension: {metadata['control_dim']}")
    print(f"  Total data points: {metadata['num_trajectories'] * metadata['horizon']}")
    
    return traj_path, meta_path


def main():
    parser = argparse.ArgumentParser(description='Collect trajectory data for LMPPI')
    parser.add_argument('--num_trajectories', type=int, default=5000, 
                       help='Number of trajectories to collect (default: 5000)')
    parser.add_argument('--horizon', type=int, default=50,
                       help='Length of each trajectory (default: 50)')
    parser.add_argument('--save_dir', type=str, default='./data',
                       help='Directory to save data (default: ./data)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cpu, cuda, or auto (default: auto)')
    parser.add_argument('--pd_ratio', type=float, default=0.8,
                       help='Ratio of PD controller trajectories (default: 0.8)')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Process trajectories in batches of this size (default: 1000)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Collecting {args.num_trajectories} trajectories with horizon {args.horizon}")
    
    # Set strategy mix
    strategy_mix = {
        'pd': args.pd_ratio,
        'random': 1.0 - args.pd_ratio
    }
    
    # Collect data in batches if requested number is large
    if args.num_trajectories > args.batch_size:
        print(f"Collecting data in batches of {args.batch_size}...")
        
        all_trajectories = []
        remaining = args.num_trajectories
        batch_num = 1
        
        while remaining > 0:
            current_batch_size = min(args.batch_size, remaining)
            print(f"\n--- Batch {batch_num}: {current_batch_size} trajectories ---")
            
            batch_trajectories = collect_trajectories(
                num_trajectories=current_batch_size,
                horizon=args.horizon,
                device=device,
                strategy_mix=strategy_mix
            )
            
            all_trajectories.extend(batch_trajectories)
            remaining -= current_batch_size
            batch_num += 1
            
            print(f"Collected {len(all_trajectories)}/{args.num_trajectories} trajectories")
        
        trajectories = all_trajectories
    else:
        # Collect all at once
        trajectories = collect_trajectories(
            num_trajectories=args.num_trajectories,
            horizon=args.horizon,
            device=device,
            strategy_mix=strategy_mix
        )
    
    # Save dataset
    traj_path, meta_path = save_dataset(trajectories, args.save_dir)
    
    print(f"\nData collection completed successfully!")
    print(f"You can now train the VAE using: python step2_train_vae.py --data_path {traj_path}")


if __name__ == "__main__":
    main()
