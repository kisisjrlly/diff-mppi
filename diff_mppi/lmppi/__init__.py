"""
Latent Space Model Predictive Path Integral (LMPPI) Control

A PyTorch-based implementation of LMPPI that uses neural networks to learn
a low-dimensional latent space representation of feasible trajectories,
enabling efficient optimization in the latent space rather than the
high-dimensional control sequence space.

Core Components:
- VAE-based trajectory encoder/decoder for latent space learning
- LMPPI controller for online control using the learned latent space
- Training utilities for offline learning of the latent representation
- Data processing utilities for trajectory datasets

Based on the principle of replacing high-dimensional control sequence sampling
with low-dimensional latent space sampling, dramatically reducing computational
complexity while maintaining trajectory feasibility.
"""

from .models import TrajectoryVAE, TrajectoryEncoder, TrajectoryDecoder
from .controller import LMPPIController
from .trainer import LMPPITrainer
from .data import TrajectoryDataset, TrajectoryDataLoader, create_synthetic_trajectories
from .utils import trajectory_to_tensor, tensor_to_trajectory, evaluate_reconstruction
from .config import VAEConfig, ControllerConfig, LMPPIConfig

__version__ = "1.0.0"
__all__ = [
    "TrajectoryVAE", 
    "TrajectoryEncoder", 
    "TrajectoryDecoder",
    "LMPPIController",
    "LMPPITrainer", 
    "TrajectoryDataset",
    "TrajectoryDataLoader",
    "create_synthetic_trajectories",
    "trajectory_to_tensor",
    "tensor_to_trajectory", 
    "evaluate_reconstruction",
    "VAEConfig",
    "ControllerConfig", 
    "LMPPIConfig"
]
