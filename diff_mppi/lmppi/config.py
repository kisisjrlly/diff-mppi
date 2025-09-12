"""
Configuration Management for LMPPI

This module provides configuration classes and utilities for managing
LMPPI hyperparameters and settings. It includes default configurations
for common use cases and validation utilities.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import json
import yaml
from pathlib import Path


@dataclass
class VAEConfig:
    """Configuration for TrajectoryVAE model."""
    
    # Model architecture
    input_dim: int
    latent_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    architecture: str = "mlp"  # "mlp", "lstm", "cnn"
    dropout: float = 0.1
    beta: float = 1.0  # KL divergence weight
    
    # For LSTM/CNN architectures
    horizon: Optional[int] = None
    feature_dim: Optional[int] = None
    
    def validate(self):
        """Validate configuration parameters."""
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        
        if self.architecture not in ["mlp", "lstm", "cnn"]:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        if self.architecture in ["lstm", "cnn"] and (self.horizon is None or self.feature_dim is None):
            raise ValueError(f"Architecture '{self.architecture}' requires horizon and feature_dim")
        
        if self.beta < 0:
            raise ValueError("beta must be non-negative")


@dataclass 
class ControllerConfig:
    """Configuration for LMPPIController."""
    
    # System dimensions
    state_dim: int
    control_dim: int
    horizon: int = 20
    
    # MPPI parameters
    num_samples: int = 100
    temperature: float = 1.0
    latent_noise_scale: float = 1.0
    
    # Control bounds
    control_min: Optional[List[float]] = None
    control_max: Optional[List[float]] = None
    
    # Device
    device: str = "cpu"
    
    def validate(self):
        """Validate configuration parameters."""
        if self.state_dim <= 0 or self.control_dim <= 0:
            raise ValueError("state_dim and control_dim must be positive")
        
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        
        if self.control_min is not None and len(self.control_min) != self.control_dim:
            raise ValueError("control_min length must match control_dim")
        
        if self.control_max is not None and len(self.control_max) != self.control_dim:
            raise ValueError("control_max length must match control_dim")


@dataclass
class TrainingConfig:
    """Configuration for LMPPI training."""
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 50
    
    # Data
    train_ratio: float = 0.8
    normalize_data: bool = True
    augment_data: bool = True
    augment_noise_std: float = 0.01
    
    # Training monitoring
    log_interval: int = 100
    validation_interval: int = 1000
    checkpoint_interval: int = 5000
    early_stopping_patience: int = 20
    
    # Paths
    save_dir: str = "./lmppi_checkpoints"
    
    def validate(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")


@dataclass
class LMPPIConfig:
    """Complete LMPPI configuration."""
    
    vae: VAEConfig
    controller: ControllerConfig  
    training: TrainingConfig
    
    # Metadata
    name: str = "lmppi_config"
    description: str = ""
    
    def validate(self):
        """Validate entire configuration."""
        self.vae.validate()
        self.controller.validate()
        self.training.validate()
        
        # Cross-validation
        expected_input_dim = self.controller.horizon * (
            self.controller.state_dim + self.controller.control_dim
        )
        
        if self.vae.architecture == "mlp" and self.vae.input_dim != expected_input_dim:
            raise ValueError(
                f"VAE input_dim ({self.vae.input_dim}) should match "
                f"horizon * (state_dim + control_dim) = {expected_input_dim}"
            )
        
        if self.vae.horizon is not None and self.vae.horizon != self.controller.horizon:
            raise ValueError("VAE horizon must match controller horizon")
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        
        # Convert to dictionary
        config_dict = self.to_dict()
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LMPPIConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        
        return dataclass_to_dict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LMPPIConfig':
        """Create configuration from dictionary."""
        vae_config = VAEConfig(**config_dict['vae'])
        controller_config = ControllerConfig(**config_dict['controller'])
        training_config = TrainingConfig(**config_dict['training'])
        
        return cls(
            vae=vae_config,
            controller=controller_config,
            training=training_config,
            name=config_dict.get('name', 'lmppi_config'),
            description=config_dict.get('description', '')
        )


# Predefined configurations for common use cases

def pendulum_config() -> LMPPIConfig:
    """Configuration for pendulum swing-up problem."""
    return LMPPIConfig(
        vae=VAEConfig(
            input_dim=20 * 3,  # horizon * (state_dim + control_dim)
            latent_dim=6,
            hidden_dims=[128, 64, 32],
            architecture="mlp",
            beta=0.5
        ),
        controller=ControllerConfig(
            state_dim=2,  # [theta, theta_dot]
            control_dim=1,  # [torque]
            horizon=20,
            num_samples=50,
            temperature=1.0,
            control_min=[-10.0],
            control_max=[10.0]
        ),
        training=TrainingConfig(
            batch_size=64,
            learning_rate=2e-3,
            num_epochs=100,
            early_stopping_patience=15
        ),
        name="pendulum_config",
        description="Configuration for pendulum swing-up control"
    )


def quadrotor_config() -> LMPPIConfig:
    """Configuration for quadrotor navigation."""
    return LMPPIConfig(
        vae=VAEConfig(
            input_dim=30 * 16,  # horizon * (state_dim + control_dim)
            latent_dim=12,
            hidden_dims=[512, 256, 128],
            architecture="lstm",
            horizon=30,
            feature_dim=16,
            beta=1.0
        ),
        controller=ControllerConfig(
            state_dim=12,  # [pos, vel, quat, omega]
            control_dim=4,   # [thrust, tau_x, tau_y, tau_z]
            horizon=30,
            num_samples=200,
            temperature=0.5,
            control_min=[0.0, -5.0, -5.0, -5.0],
            control_max=[20.0, 5.0, 5.0, 5.0]
        ),
        training=TrainingConfig(
            batch_size=32,
            learning_rate=1e-3,
            num_epochs=200,
            augment_noise_std=0.005
        ),
        name="quadrotor_config",
        description="Configuration for quadrotor navigation"
    )


def robotic_arm_config(num_joints: int = 7) -> LMPPIConfig:
    """Configuration for robotic arm manipulation."""
    return LMPPIConfig(
        vae=VAEConfig(
            input_dim=25 * (num_joints * 2 + num_joints),  # horizon * (states + controls)
            latent_dim=16,
            hidden_dims=[1024, 512, 256],
            architecture="lstm",
            horizon=25,
            feature_dim=num_joints * 3,  # [q, q_dot, tau]
            beta=0.8
        ),
        controller=ControllerConfig(
            state_dim=num_joints * 2,  # [q, q_dot]
            control_dim=num_joints,    # [tau]
            horizon=25,
            num_samples=150,
            temperature=0.8,
            control_min=[-50.0] * num_joints,
            control_max=[50.0] * num_joints
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=5e-4,
            num_epochs=300,
            augment_data=True,
            augment_noise_std=0.01
        ),
        name=f"robotic_arm_{num_joints}dof_config",
        description=f"Configuration for {num_joints}-DOF robotic arm manipulation"
    )


def autonomous_driving_config() -> LMPPIConfig:
    """Configuration for autonomous driving."""
    return LMPPIConfig(
        vae=VAEConfig(
            input_dim=40 * 7,  # horizon * (state_dim + control_dim)
            latent_dim=20,
            hidden_dims=[512, 256, 128, 64],
            architecture="cnn",
            horizon=40,
            feature_dim=7,  # [x, y, theta, v, a, steering, throttle]
            beta=1.2
        ),
        controller=ControllerConfig(
            state_dim=5,   # [x, y, theta, v, a]
            control_dim=2, # [steering, throttle]
            horizon=40,
            num_samples=300,
            temperature=0.3,
            control_min=[-0.5, -1.0],  # [steering, throttle]
            control_max=[0.5, 1.0]
        ),
        training=TrainingConfig(
            batch_size=24,
            learning_rate=8e-4,
            num_epochs=500,
            validation_interval=500,
            checkpoint_interval=2000
        ),
        name="autonomous_driving_config",
        description="Configuration for autonomous driving control"
    )


# Configuration registry for easy access
CONFIG_REGISTRY = {
    "pendulum": pendulum_config,
    "quadrotor": quadrotor_config, 
    "robotic_arm": robotic_arm_config,
    "autonomous_driving": autonomous_driving_config
}


def get_config(name: str, **kwargs) -> LMPPIConfig:
    """
    Get predefined configuration by name.
    
    Args:
        name: Configuration name
        **kwargs: Additional arguments for configuration
        
    Returns:
        LMPPIConfig instance
    """
    if name not in CONFIG_REGISTRY:
        available = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    
    config_fn = CONFIG_REGISTRY[name]
    return config_fn(**kwargs)


def list_configs() -> List[str]:
    """List all available predefined configurations."""
    return list(CONFIG_REGISTRY.keys())


def create_custom_config(
    state_dim: int,
    control_dim: int,
    horizon: int = 20,
    latent_dim: int = 8,
    **kwargs
) -> LMPPIConfig:
    """
    Create a custom configuration with sensible defaults.
    
    Args:
        state_dim: State space dimension
        control_dim: Control space dimension
        horizon: Planning horizon
        latent_dim: Latent space dimension
        **kwargs: Additional configuration overrides
        
    Returns:
        LMPPIConfig instance
    """
    # Calculate input dimension for MLP
    feature_dim = state_dim + control_dim
    input_dim = horizon * feature_dim
    
    # Determine model size based on problem complexity
    total_dim = state_dim + control_dim
    if total_dim <= 5:
        hidden_dims = [128, 64, 32]
    elif total_dim <= 15:
        hidden_dims = [256, 128, 64]
    else:
        hidden_dims = [512, 256, 128]
    
    # Base configuration
    config = LMPPIConfig(
        vae=VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            architecture="mlp"
        ),
        controller=ControllerConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=horizon
        ),
        training=TrainingConfig(),
        name="custom_config",
        description=f"Custom config for {state_dim}D state, {control_dim}D control"
    )
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config.vae, key):
            setattr(config.vae, key, value)
        elif hasattr(config.controller, key):
            setattr(config.controller, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)
    
    config.validate()
    return config
