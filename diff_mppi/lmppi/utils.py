"""
Utility Functions for LMPPI

This module provides various utility functions for LMPPI implementation:

1. Trajectory conversion utilities
2. Evaluation metrics
3. Visualization helpers
4. Model loading/saving utilities
5. Configuration management

These utilities support the main LMPPI workflow and provide convenient
interfaces for common operations.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle

from .models import TrajectoryVAE


def trajectory_to_tensor(
    trajectory: Union[np.ndarray, List[np.ndarray]], 
    state_dim: int,
    control_dim: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Convert trajectory data to standardized tensor format.
    
    Args:
        trajectory: Trajectory data
        state_dim: State space dimension
        control_dim: Control space dimension
        device: Device for tensor
        
    Returns:
        Trajectory tensor [horizon, state_dim + control_dim]
    """
    if isinstance(trajectory, list):
        trajectory = np.array(trajectory)
    
    trajectory_tensor = torch.from_numpy(trajectory).float()
    
    # Ensure correct feature dimension
    expected_features = state_dim + control_dim
    if trajectory_tensor.shape[-1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {trajectory_tensor.shape[-1]}")
    
    return trajectory_tensor.to(device)


def tensor_to_trajectory(
    tensor: torch.Tensor,
    state_dim: int,
    control_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tensor to separate state and control trajectories.
    
    Args:
        tensor: Trajectory tensor [..., horizon, state_dim + control_dim]
        state_dim: State space dimension
        control_dim: Control space dimension
        
    Returns:
        states: State trajectory [..., horizon, state_dim]
        controls: Control trajectory [..., horizon, control_dim]
    """
    if tensor.dim() < 2:
        raise ValueError("Tensor must have at least 2 dimensions")
    
    tensor_np = tensor.detach().cpu().numpy()
    
    states = tensor_np[..., :state_dim]
    controls = tensor_np[..., state_dim:state_dim + control_dim]
    
    return states, controls


def evaluate_reconstruction(
    model: TrajectoryVAE,
    original_trajectories: torch.Tensor,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality of VAE model.
    
    Args:
        model: Trained VAE model
        original_trajectories: Original trajectory data
        device: Device for computation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    original_trajectories = original_trajectories.to(device)
    
    with torch.no_grad():
        # Encode and decode
        mu, logvar = model.encode(original_trajectories)
        z = model.reparameterize(mu, logvar)
        reconstructed = model.decode(z)
        
        # Compute metrics
        mse = torch.mean((original_trajectories - reconstructed) ** 2).item()
        mae = torch.mean(torch.abs(original_trajectories - reconstructed)).item()
        
        # Per-feature metrics
        feature_mse = torch.mean(
            (original_trajectories - reconstructed) ** 2, dim=(0, 1)
        ).cpu().numpy()
        
        # Latent space metrics
        latent_mean = torch.mean(mu, dim=0).cpu().numpy()
        latent_std = torch.std(mu, dim=0).cpu().numpy()
        latent_norm = torch.norm(mu, dim=1).mean().item()
        
        # Variance explained
        total_var = torch.var(original_trajectories, dim=(0, 1)).sum().item()
        residual_var = torch.var(original_trajectories - reconstructed, dim=(0, 1)).sum().item()
        var_explained = 1 - (residual_var / total_var) if total_var > 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'feature_mse': feature_mse.tolist(),
        'latent_mean_norm': float(np.linalg.norm(latent_mean)),
        'latent_std_mean': np.mean(latent_std),
        'latent_norm_mean': latent_norm,
        'variance_explained': var_explained
    }


def plot_latent_space(
    model: TrajectoryVAE,
    dataset: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    device: str = "cpu"
):
    """
    Visualize latent space representation.
    
    Args:
        model: Trained VAE model
        dataset: Dataset to encode
        labels: Optional labels for coloring points
        save_path: Path to save plot
        device: Device for computation
    """
    model.eval()
    model = model.to(device)
    dataset = dataset.to(device)
    
    with torch.no_grad():
        mu, _ = model.encode(dataset)
        latent_codes = mu.cpu().numpy()
    
    # Create visualization
    num_plots = min(3, model.latent_dim)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Ensure axes is always a list for consistent indexing
    if num_plots == 1:
        axes = [axes]
    elif not isinstance(axes, list):
        axes = list(axes)
    
    if model.latent_dim >= 2:
        # 2D scatter plot
        ax = axes[0]
        scatter = ax.scatter(
            latent_codes[:, 0], 
            latent_codes[:, 1], 
            c=labels, 
            alpha=0.6,
            cmap='viridis' if labels is not None else None
        )
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title('Latent Space (2D)')
        ax.grid(True)
        
        if labels is not None:
            plt.colorbar(scatter, ax=ax)
    
    if model.latent_dim >= 3 and len(axes) > 1:
        # 3D projection
        ax = axes[1]
        scatter = ax.scatter(
            latent_codes[:, 0],
            latent_codes[:, 2],
            c=labels,
            alpha=0.6,
            cmap='viridis' if labels is not None else None
        )
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 3')
        ax.set_title('Latent Space (1st vs 3rd dim)')
        ax.grid(True)
    
    if model.latent_dim > 1 and len(axes) > 2:
        # Distribution plot
        ax = axes[2]
        for i in range(min(4, model.latent_dim)):
            ax.hist(latent_codes[:, i], alpha=0.5, label=f'Dim {i+1}', bins=30)
        ax.set_xlabel('Latent Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Latent Distributions')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent space plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_reconstruction_comparison(
    model: TrajectoryVAE,
    trajectories: torch.Tensor,
    state_dim: int,
    control_dim: int,
    num_examples: int = 4,
    save_path: Optional[str] = None,
    device: str = "cpu"
):
    """
    Plot comparison between original and reconstructed trajectories.
    
    Args:
        model: Trained VAE model
        trajectories: Original trajectories
        state_dim: State space dimension
        control_dim: Control space dimension
        num_examples: Number of examples to plot
        save_path: Path to save plot
        device: Device for computation
    """
    model.eval()
    model = model.to(device)
    trajectories = trajectories.to(device)
    
    # Select random examples
    indices = torch.randperm(len(trajectories))[:min(num_examples, len(trajectories))]
    selected_trajectories = trajectories[indices]
    
    with torch.no_grad():
        reconstructed, _, _ = model(selected_trajectories)
    
    # Convert to numpy
    original_np = selected_trajectories.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    
    # Create plots
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Split trajectories
        orig_states, orig_controls = tensor_to_trajectory(
            torch.from_numpy(original_np[i]), state_dim, control_dim
        )
        recon_states, recon_controls = tensor_to_trajectory(
            torch.from_numpy(reconstructed_np[i]), state_dim, control_dim
        )
        
        # Plot states
        axes[i, 0].plot(orig_states[:, 0], 'b-', label='Original', linewidth=2)
        axes[i, 0].plot(recon_states[:, 0], 'r--', label='Reconstructed', linewidth=2)
        axes[i, 0].set_title(f'Example {i+1}: State Dim 1')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Plot controls
        axes[i, 1].plot(orig_controls[:, 0], 'b-', label='Original', linewidth=2)
        axes[i, 1].plot(recon_controls[:, 0], 'r--', label='Reconstructed', linewidth=2)
        axes[i, 1].set_title(f'Example {i+1}: Control Dim 1')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
        
        # Plot additional dimensions if available
        if state_dim > 1:
            axes[i, 2].plot(orig_states[:, 1], 'b-', label='Original', linewidth=2)
            axes[i, 2].plot(recon_states[:, 1], 'r--', label='Reconstructed', linewidth=2)
            axes[i, 2].set_title(f'Example {i+1}: State Dim 2')
            axes[i, 2].legend()
            axes[i, 2].grid(True)
        else:
            axes[i, 2].axis('off')
        
        if control_dim > 1:
            axes[i, 3].plot(orig_controls[:, 1], 'b-', label='Original', linewidth=2)
            axes[i, 3].plot(recon_controls[:, 1], 'r--', label='Reconstructed', linewidth=2)
            axes[i, 3].set_title(f'Example {i+1}: Control Dim 2')
            axes[i, 3].legend()
            axes[i, 3].grid(True)
        else:
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_model_config(
    model: TrajectoryVAE,
    config_path: str,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save model configuration to JSON file.
    
    Args:
        model: VAE model
        config_path: Path to save configuration
        additional_info: Additional information to save
    """
    config = {
        'model_type': 'TrajectoryVAE',
        'input_dim': model.encoder.input_dim,
        'latent_dim': model.latent_dim,
        'architecture': model.architecture,
        'beta': model.beta,
        'hidden_dims': getattr(model.encoder, 'hidden_dims', None),
        'horizon': getattr(model.encoder, 'horizon', None),
        'feature_dim': getattr(model.encoder, 'feature_dim', None)
    }
    
    if additional_info:
        config.update(additional_info)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model configuration saved to {config_path}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu"
) -> Tuple[TrajectoryVAE, Dict[str, Any]]:
    """
    Load VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model and checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    
    # Create model (you may need to adjust based on your exact config structure)
    model = TrajectoryVAE(
        input_dim=model_config.get('input_dim', 100),
        latent_dim=model_config.get('latent_dim', 8),
        architecture=model_config.get('architecture', 'mlp'),
        beta=model_config.get('beta', 1.0)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    
    return model, checkpoint


def compute_trajectory_diversity(trajectories: torch.Tensor) -> Dict[str, float]:
    """
    Compute diversity metrics for a set of trajectories.
    
    Args:
        trajectories: Batch of trajectories [batch_size, horizon, features]
        
    Returns:
        Dictionary of diversity metrics
    """
    batch_size, horizon, features = trajectories.shape
    
    # Pairwise distances
    trajectories_flat = trajectories.view(batch_size, -1)
    
    # Compute pairwise L2 distances
    distances = torch.cdist(trajectories_flat, trajectories_flat, p=2)
    
    # Remove diagonal (self-distances)
    mask = ~torch.eye(batch_size, dtype=torch.bool)
    distances = distances[mask]
    
    # Compute metrics
    mean_distance = torch.mean(distances).item()
    std_distance = torch.std(distances).item()
    min_distance = torch.min(distances).item()
    max_distance = torch.max(distances).item()
    
    # Trajectory variance
    trajectory_variance = torch.var(trajectories, dim=0).mean().item()
    
    return {
        'mean_pairwise_distance': mean_distance,
        'std_pairwise_distance': std_distance,
        'min_pairwise_distance': min_distance,
        'max_pairwise_distance': max_distance,
        'trajectory_variance': trajectory_variance
    }


def interpolate_in_latent_space(
    model: TrajectoryVAE,
    trajectory1: torch.Tensor,
    trajectory2: torch.Tensor,
    num_steps: int = 10,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Interpolate between two trajectories in latent space.
    
    Args:
        model: Trained VAE model
        trajectory1: First trajectory
        trajectory2: Second trajectory
        num_steps: Number of interpolation steps
        device: Device for computation
        
    Returns:
        Interpolated trajectories [num_steps, horizon, features]
    """
    model.eval()
    model = model.to(device)
    trajectory1 = trajectory1.to(device).unsqueeze(0)
    trajectory2 = trajectory2.to(device).unsqueeze(0)
    
    with torch.no_grad():
        # Encode both trajectories
        mu1, _ = model.encode(trajectory1)
        mu2, _ = model.encode(trajectory2)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps, device=device).unsqueeze(1)
        
        # Interpolate in latent space
        interpolated_z = (1 - alphas) * mu1 + alphas * mu2
        
        # Decode interpolated latent codes
        interpolated_trajectories = model.decode(interpolated_z)
    
    return interpolated_trajectories


def validate_trajectory_feasibility(
    trajectories: torch.Tensor,
    dynamics_fn: Optional[Callable] = None,
    control_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    state_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Dict[str, Any]:
    """
    Validate feasibility of generated trajectories.
    
    Args:
        trajectories: Generated trajectories
        dynamics_fn: Optional dynamics function for consistency check
        control_bounds: Optional control bounds (min, max)
        state_bounds: Optional state bounds (min, max)
        
    Returns:
        Feasibility metrics
    """
    results = {}
    
    # Extract states and controls (assume first part is states)
    # This is a simplified assumption - adjust based on your trajectory format
    half_features = trajectories.shape[-1] // 2
    states = trajectories[..., :half_features]
    controls = trajectories[..., half_features:]
    
    # Check control bounds
    if control_bounds is not None:
        control_min, control_max = control_bounds
        control_violations = torch.sum(
            (controls < control_min) | (controls > control_max)
        ).item()
        results['control_bound_violations'] = control_violations
        results['control_bound_violation_rate'] = control_violations / controls.numel()
    
    # Check state bounds
    if state_bounds is not None:
        state_min, state_max = state_bounds
        state_violations = torch.sum(
            (states < state_min) | (states > state_max)
        ).item()
        results['state_bound_violations'] = state_violations
        results['state_bound_violation_rate'] = state_violations / states.numel()
    
    # Check dynamics consistency (if dynamics function provided)
    if dynamics_fn is not None:
        dynamics_errors = []
        for i in range(trajectories.shape[0]):
            traj_states = states[i]
            traj_controls = controls[i]
            
            for t in range(len(traj_states) - 1):
                predicted_next_state = dynamics_fn(traj_states[t], traj_controls[t])
                actual_next_state = traj_states[t + 1]
                error = torch.norm(predicted_next_state - actual_next_state).item()
                dynamics_errors.append(error)
        
        results['dynamics_consistency_error_mean'] = np.mean(dynamics_errors)
        results['dynamics_consistency_error_std'] = np.std(dynamics_errors)
        results['dynamics_consistency_error_max'] = np.max(dynamics_errors)
    
    # Basic trajectory statistics
    results['trajectory_length_mean'] = float(torch.norm(
        torch.diff(trajectories, dim=1), dim=-1
    ).mean())
    
    results['trajectory_smoothness'] = float(torch.norm(
        torch.diff(trajectories, n=2, dim=1), dim=-1
    ).mean())
    
    return results
