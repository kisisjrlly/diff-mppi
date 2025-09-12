"""
Data Processing Module for LMPPI

This module provides data handling utilities for trajectory datasets used
in LMPPI training and evaluation:

1. TrajectoryDataset: PyTorch dataset for trajectory data
2. TrajectoryDataLoader: Specialized data loader
3. Data preprocessing and augmentation utilities
4. Dataset generation helpers

Supports various trajectory formats and provides efficient batching
for VAE training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import os
import pickle
import json
from pathlib import Path


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory data.
    
    Handles loading, preprocessing, and serving trajectory data for VAE training.
    Supports various input formats and provides data augmentation options.
    """
    
    def __init__(
        self,
        trajectories: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        state_dim: int,
        control_dim: int,
        horizon: Optional[int] = None,
        normalize: bool = True,
        augment: bool = False,
        augment_noise_std: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize trajectory dataset.
        
        Args:
            trajectories: Trajectory data in various formats:
                - torch.Tensor: [num_trajectories, horizon, state_dim + control_dim]
                - np.ndarray: Same shape as tensor
                - List[np.ndarray]: List of individual trajectories
            state_dim: Dimension of state space
            control_dim: Dimension of control space
            horizon: Expected horizon length (inferred if None)
            normalize: Whether to normalize trajectories
            augment: Whether to apply data augmentation
            augment_noise_std: Standard deviation for augmentation noise
            device: Device for tensor storage
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.feature_dim = state_dim + control_dim
        self.normalize = normalize
        self.augment = augment
        self.augment_noise_std = augment_noise_std
        self.device = device
        
        # Process input trajectories
        self.trajectories = self._process_trajectories(trajectories, horizon)
        self.horizon = self.trajectories.shape[1]
        
        # Compute normalization statistics
        if self.normalize:
            self.mean = torch.mean(self.trajectories, dim=(0, 1), keepdim=True)
            self.std = torch.std(self.trajectories, dim=(0, 1), keepdim=True)
            # Avoid division by zero
            self.std = torch.clamp(self.std, min=1e-8)
        else:
            self.mean = torch.zeros(1, 1, self.feature_dim)
            self.std = torch.ones(1, 1, self.feature_dim)
    
    def _process_trajectories(
        self, 
        trajectories: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        horizon: Optional[int]
    ) -> torch.Tensor:
        """Process input trajectories into standard tensor format."""
        
        if isinstance(trajectories, list):
            # Convert list of arrays to tensor
            if horizon is None:
                horizon = max(len(traj) for traj in trajectories)
            
            processed_trajs = []
            for traj in trajectories:
                if len(traj) != horizon:
                    # Pad or truncate to horizon length
                    if len(traj) < horizon:
                        # Pad with last value
                        padding = np.tile(traj[-1:], (horizon - len(traj), 1))
                        traj = np.concatenate([traj, padding], axis=0)
                    else:
                        # Truncate
                        traj = traj[:horizon]
                
                processed_trajs.append(traj)
            
            trajectories = np.stack(processed_trajs, axis=0)
        
        # Convert to tensor
        if isinstance(trajectories, np.ndarray):
            trajectories = torch.from_numpy(trajectories).float()
        elif not isinstance(trajectories, torch.Tensor):
            raise ValueError(f"Unsupported trajectory type: {type(trajectories)}")
        
        # Validate shape
        if trajectories.dim() != 3:
            raise ValueError(f"Expected 3D trajectories [num_traj, horizon, features], got {trajectories.shape}")
        
        if trajectories.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {trajectories.shape[2]}")
        
        return trajectories.to(self.device)
    
    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get trajectory by index."""
        trajectory = self.trajectories[idx].clone()
        
        # Apply normalization
        if self.normalize:
            trajectory = (trajectory - self.mean) / self.std
        
        # Apply augmentation
        if self.augment and self.training:
            noise = torch.randn_like(trajectory) * self.augment_noise_std
            trajectory = trajectory + noise
        
        return trajectory
    
    def get_unnormalized(self, idx: int) -> torch.Tensor:
        """Get unnormalized trajectory."""
        return self.trajectories[idx].clone()
    
    def denormalize(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """Convert normalized trajectory back to original scale."""
        if not self.normalize:
            return normalized_trajectory
        
        return normalized_trajectory * self.std + self.mean
    
    def normalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalize a trajectory using dataset statistics."""
        if not self.normalize:
            return trajectory
        
        return (trajectory - self.mean) / self.std
    
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get dataset statistics."""
        # Compute min/max properly for multi-dimensional reduction
        flat_trajectories = self.trajectories.view(-1, self.feature_dim)
        return {
            'mean': self.mean,
            'std': self.std,
            'min': torch.min(flat_trajectories, dim=0, keepdim=True)[0].unsqueeze(0),
            'max': torch.max(flat_trajectories, dim=0, keepdim=True)[0].unsqueeze(0)
        }
    
    def train(self):
        """Set dataset to training mode (enables augmentation)."""
        self.training = True
    
    def eval(self):
        """Set dataset to evaluation mode (disables augmentation)."""
        self.training = False
    
    def split(self, train_ratio: float = 0.8) -> Tuple['TrajectoryDataset', 'TrajectoryDataset']:
        """
        Split dataset into training and validation sets.
        
        Args:
            train_ratio: Ratio of data for training
            
        Returns:
            train_dataset, val_dataset
        """
        num_train = int(len(self) * train_ratio)
        indices = torch.randperm(len(self))
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        train_trajectories = self.trajectories[train_indices]
        val_trajectories = self.trajectories[val_indices]
        
        train_dataset = TrajectoryDataset(
            trajectories=train_trajectories,
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            normalize=False,  # Already processed
            augment=self.augment,
            augment_noise_std=self.augment_noise_std,
            device=self.device
        )
        
        val_dataset = TrajectoryDataset(
            trajectories=val_trajectories,
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            normalize=False,  # Already processed
            augment=False,  # No augmentation for validation
            device=self.device
        )
        
        # Share normalization statistics
        if self.normalize:
            train_dataset.normalize = True
            train_dataset.mean = self.mean
            train_dataset.std = self.std
            
            val_dataset.normalize = True
            val_dataset.mean = self.mean
            val_dataset.std = self.std
        
        return train_dataset, val_dataset
    
    def save(self, path: str):
        """Save dataset to disk."""
        data = {
            'trajectories': self.trajectories.cpu().numpy(),
            'state_dim': self.state_dim,
            'control_dim': self.control_dim,
            'horizon': self.horizon,
            'normalize': self.normalize,
            'mean': self.mean.cpu().numpy() if self.normalize else None,
            'std': self.std.cpu().numpy() if self.normalize else None,
            'augment': self.augment,
            'augment_noise_std': self.augment_noise_std
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Dataset saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> 'TrajectoryDataset':
        """Load dataset from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        dataset = cls(
            trajectories=data['trajectories'],
            state_dim=data['state_dim'],
            control_dim=data['control_dim'],
            normalize=False,  # Will be set manually
            augment=data['augment'],
            augment_noise_std=data['augment_noise_std'],
            device=device
        )
        
        # Restore normalization if it was used
        if data['normalize'] and data['mean'] is not None:
            dataset.normalize = True
            dataset.mean = torch.from_numpy(data['mean']).to(device)
            dataset.std = torch.from_numpy(data['std']).to(device)
        
        print(f"Dataset loaded from {path}")
        return dataset


class TrajectoryDataLoader:
    """
    Specialized data loader for trajectory datasets.
    
    Provides additional functionality for trajectory-specific batching
    and preprocessing.
    """
    
    def __init__(
        self,
        dataset: TrajectoryDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False
    ):
        """
        Initialize trajectory data loader.
        
        Args:
            dataset: TrajectoryDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory (for GPU transfer)
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.loader)


def create_synthetic_trajectories(
    num_trajectories: int,
    horizon: int,
    state_dim: int,
    control_dim: int,
    trajectory_types: List[str] = ["linear", "sinusoidal", "spiral"],
    noise_std: float = 0.1,
    device: str = "cpu"
) -> TrajectoryDataset:
    """
    Create synthetic trajectory dataset for testing and development.
    
    Args:
        num_trajectories: Number of trajectories to generate
        horizon: Time horizon for each trajectory
        state_dim: State space dimension
        control_dim: Control space dimension
        trajectory_types: Types of trajectories to generate
        noise_std: Standard deviation of noise to add
        device: Device for computation
        
    Returns:
        TrajectoryDataset containing synthetic trajectories
    """
    trajectories = []
    
    for i in range(num_trajectories):
        # Choose random trajectory type
        traj_type = np.random.choice(trajectory_types)
        
        # Generate time vector
        t = np.linspace(0, 2 * np.pi, horizon)
        
        # Generate trajectory based on type
        if traj_type == "linear":
            # Linear trajectory
            start = np.random.randn(state_dim) * 2
            end = np.random.randn(state_dim) * 2
            states = np.outer(t / t[-1], end - start) + start
            
        elif traj_type == "sinusoidal":
            # Sinusoidal trajectory
            freq = np.random.uniform(0.5, 2.0)
            amp = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi, state_dim)
            states = amp * np.sin(freq * t.reshape(-1, 1) + phase)
            
        elif traj_type == "spiral":
            # Spiral trajectory
            radius = np.linspace(0.1, 2.0, horizon)
            angle = t * np.random.uniform(1, 3)
            if state_dim >= 2:
                states = np.zeros((horizon, state_dim))
                states[:, 0] = radius * np.cos(angle)
                states[:, 1] = radius * np.sin(angle)
                if state_dim > 2:
                    states[:, 2:] = np.random.randn(horizon, state_dim - 2) * 0.1
            else:
                states = radius.reshape(-1, 1)
        
        else:
            # Random walk
            states = np.cumsum(np.random.randn(horizon, state_dim) * 0.1, axis=0)
        
        # Generate controls (simple proportional control towards target)
        controls = np.zeros((horizon, control_dim))
        for t_idx in range(horizon - 1):
            target = states[t_idx + 1]
            current = states[t_idx]
            if control_dim <= state_dim:
                controls[t_idx, :] = (target - current)[:control_dim]
            else:
                controls[t_idx, :state_dim] = target - current
                controls[t_idx, state_dim:] = np.random.randn(control_dim - state_dim) * 0.1
        
        # Last control is zero
        controls[-1, :] = 0
        
        # Combine states and controls
        trajectory = np.concatenate([states, controls], axis=1)
        
        # Add noise
        trajectory += np.random.randn(*trajectory.shape) * noise_std
        
        trajectories.append(trajectory)
    
    # Create dataset
    return TrajectoryDataset(
        trajectories=trajectories,
        state_dim=state_dim,
        control_dim=control_dim,
        normalize=True,
        device=device
    )


def load_trajectory_data(
    data_path: str,
    state_dim: int,
    control_dim: int,
    file_format: str = "auto"
) -> TrajectoryDataset:
    """
    Load trajectory data from various file formats.
    
    Args:
        data_path: Path to data file
        state_dim: State space dimension
        control_dim: Control space dimension
        file_format: File format ("auto", "npz", "csv", "pkl", "mat")
        
    Returns:
        TrajectoryDataset containing loaded trajectories
    """
    path = Path(data_path)
    
    if file_format == "auto":
        file_format = path.suffix[1:]  # Remove the dot
    
    if file_format == "npz":
        data = np.load(data_path)
        if 'trajectories' in data:
            trajectories = data['trajectories']
        else:
            # Assume first array is trajectories
            trajectories = data[list(data.keys())[0]]
            
    elif file_format == "csv":
        # Assume CSV format: each row is a flattened trajectory
        data = np.loadtxt(data_path, delimiter=',')
        horizon = data.shape[1] // (state_dim + control_dim)
        trajectories = data.reshape(-1, horizon, state_dim + control_dim)
        
    elif file_format == "pkl":
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'trajectories' in data:
            trajectories = data['trajectories']
        elif isinstance(data, (np.ndarray, list)):
            trajectories = data
        else:
            raise ValueError(f"Unsupported data format in pickle file: {type(data)}")
            
    elif file_format == "mat":
        from scipy.io import loadmat
        data = loadmat(data_path)
        # Assume trajectories are stored under 'trajectories' key
        if 'trajectories' in data:
            trajectories = data['trajectories']
        else:
            # Try first non-metadata key
            keys = [k for k in data.keys() if not k.startswith('__')]
            if keys:
                trajectories = data[keys[0]]
            else:
                raise ValueError("No trajectory data found in .mat file")
        
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    return TrajectoryDataset(
        trajectories=trajectories,
        state_dim=state_dim,
        control_dim=control_dim,
        normalize=True
    )


def visualize_trajectories(
    dataset: TrajectoryDataset,
    num_trajectories: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize sample trajectories from dataset.
    
    Args:
        dataset: TrajectoryDataset to visualize
        num_trajectories: Number of trajectories to plot
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    num_trajectories = min(num_trajectories, len(dataset))
    indices = np.random.choice(len(dataset), num_trajectories, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices[:4]):
        trajectory = dataset.get_unnormalized(idx).cpu().numpy()
        
        # Split into states and controls
        states = trajectory[:, :dataset.state_dim]
        controls = trajectory[:, dataset.state_dim:]
        
        # Plot states
        axes[0].plot(states[:, 0], label=f'Traj {i+1}' if i < 4 else None)
        axes[1].plot(controls[:, 0], label=f'Traj {i+1}' if i < 4 else None)
        
        # If multi-dimensional, plot additional dimensions
        if dataset.state_dim > 1:
            axes[2].plot(states[:, 1], label=f'Traj {i+1}' if i < 4 else None)
        if dataset.control_dim > 1:
            axes[3].plot(controls[:, 1], label=f'Traj {i+1}' if i < 4 else None)
    
    axes[0].set_title('State Dimension 1')
    axes[0].set_xlabel('Time Step')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title('Control Dimension 1')
    axes[1].set_xlabel('Time Step')
    axes[1].legend()
    axes[1].grid(True)
    
    if dataset.state_dim > 1:
        axes[2].set_title('State Dimension 2')
        axes[2].set_xlabel('Time Step')
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[2].axis('off')
    
    if dataset.control_dim > 1:
        axes[3].set_title('Control Dimension 2')
        axes[3].set_xlabel('Time Step')
        axes[3].legend()
        axes[3].grid(True)
    else:
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
