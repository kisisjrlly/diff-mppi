"""
Test Suite for LMPPI Implementation

This module contains comprehensive tests for all LMPPI components:
- VAE model architectures and training
- Controller functionality 
- Data handling and preprocessing
- Configuration management
- Integration tests

Run with: python -m pytest test_lmppi.py
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Import LMPPI components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from diff_mppi.lmppi import (
    TrajectoryVAE, TrajectoryEncoder, TrajectoryDecoder,
    LMPPIController, LMPPITrainer,
    TrajectoryDataset,
    trajectory_to_tensor, tensor_to_trajectory
)
from diff_mppi.lmppi.data import create_synthetic_trajectories
from diff_mppi.lmppi.config import (
    VAEConfig, ControllerConfig, TrainingConfig, LMPPIConfig,
    pendulum_config, get_config
)


class TestTrajectoryVAE:
    """Test VAE model components."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample trajectory data."""
        batch_size = 10
        horizon = 15
        state_dim = 4
        control_dim = 2
        feature_dim = state_dim + control_dim
        
        # MLP format (flattened)
        mlp_data = torch.randn(batch_size, horizon * feature_dim)
        
        # Structured format
        structured_data = torch.randn(batch_size, horizon, feature_dim)
        
        return {
            'mlp': mlp_data,
            'structured': structured_data,
            'batch_size': batch_size,
            'horizon': horizon,
            'state_dim': state_dim,
            'control_dim': control_dim,
            'feature_dim': feature_dim
        }
    
    def test_encoder_mlp(self, sample_data):
        """Test MLP encoder."""
        encoder = TrajectoryEncoder(
            input_dim=sample_data['horizon'] * sample_data['feature_dim'],
            latent_dim=8,
            hidden_dims=[64, 32],
            architecture="mlp"
        )
        
        mu, logvar = encoder(sample_data['mlp'])
        
        assert mu.shape == (sample_data['batch_size'], 8)
        assert logvar.shape == (sample_data['batch_size'], 8)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(logvar).any()
    
    def test_encoder_lstm(self, sample_data):
        """Test LSTM encoder."""
        encoder = TrajectoryEncoder(
            input_dim=sample_data['horizon'] * sample_data['feature_dim'],
            latent_dim=8,
            hidden_dims=[64, 32],
            architecture="lstm",
            horizon=sample_data['horizon'],
            feature_dim=sample_data['feature_dim']
        )
        
        mu, logvar = encoder(sample_data['structured'])
        
        assert mu.shape == (sample_data['batch_size'], 8)
        assert logvar.shape == (sample_data['batch_size'], 8)
    
    def test_decoder_mlp(self, sample_data):
        """Test MLP decoder."""
        latent_dim = 8
        output_dim = sample_data['horizon'] * sample_data['feature_dim']
        
        decoder = TrajectoryDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=[32, 64],
            architecture="mlp"
        )
        
        z = torch.randn(sample_data['batch_size'], latent_dim)
        reconstruction = decoder(z)
        
        assert reconstruction.shape == (sample_data['batch_size'], output_dim)
    
    def test_vae_forward(self, sample_data):
        """Test complete VAE forward pass."""
        vae = TrajectoryVAE(
            input_dim=sample_data['horizon'] * sample_data['feature_dim'],
            latent_dim=8,
            hidden_dims=[64, 32],
            architecture="mlp"
        )
        
        reconstruction, mu, logvar = vae(sample_data['mlp'])
        
        assert reconstruction.shape == sample_data['mlp'].shape
        assert mu.shape == (sample_data['batch_size'], 8)
        assert logvar.shape == (sample_data['batch_size'], 8)
    
    def test_vae_loss(self, sample_data):
        """Test VAE loss computation."""
        vae = TrajectoryVAE(
            input_dim=sample_data['horizon'] * sample_data['feature_dim'],
            latent_dim=8,
            architecture="mlp"
        )
        
        reconstruction, mu, logvar = vae(sample_data['mlp'])
        total_loss, recon_loss, kl_loss = vae.compute_loss(
            sample_data['mlp'], reconstruction, mu, logvar
        )
        
        assert total_loss.item() > 0
        assert recon_loss.item() > 0
        assert kl_loss.item() >= 0  # KL can be 0
    
    def test_vae_sampling(self, sample_data):
        """Test VAE sampling."""
        vae = TrajectoryVAE(
            input_dim=sample_data['horizon'] * sample_data['feature_dim'],
            latent_dim=8,
            architecture="mlp"
        )
        
        samples = vae.sample(num_samples=5)
        expected_shape = (5, sample_data['horizon'] * sample_data['feature_dim'])
        assert samples.shape == expected_shape


class TestTrajectoryDataset:
    """Test trajectory dataset functionality."""
    
    def test_dataset_creation(self):
        """Test creating dataset from various input formats."""
        # Test with numpy array
        trajectories_np = np.random.randn(50, 20, 6)
        dataset = TrajectoryDataset(
            trajectories=trajectories_np,
            state_dim=4,
            control_dim=2
        )
        assert len(dataset) == 50
        assert dataset.horizon == 20
        
        # Test with list of arrays
        traj_list = [np.random.randn(20, 6) for _ in range(30)]
        dataset2 = TrajectoryDataset(
            trajectories=traj_list,
            state_dim=4,
            control_dim=2
        )
        assert len(dataset2) == 30
    
    def test_dataset_normalization(self):
        """Test dataset normalization."""
        trajectories = np.random.randn(100, 15, 5) * 10 + 5
        
        dataset = TrajectoryDataset(
            trajectories=trajectories,
            state_dim=3,
            control_dim=2,
            normalize=True
        )
        
        # Check normalization statistics
        stats = dataset.get_statistics()
        normalized_mean = torch.mean(dataset.trajectories, dim=(0, 1))
        assert torch.allclose(normalized_mean, stats['mean'].squeeze(), atol=1e-5)
    
    def test_dataset_split(self):
        """Test train/validation split."""
        trajectories = np.random.randn(100, 10, 4)
        dataset = TrajectoryDataset(
            trajectories=trajectories,
            state_dim=2,
            control_dim=2
        )
        
        train_data, val_data = dataset.split(train_ratio=0.8)
        
        assert len(train_data) == 80
        assert len(val_data) == 20
        assert len(train_data) + len(val_data) == len(dataset)
    
    def test_synthetic_trajectories(self):
        """Test synthetic trajectory generation."""
        dataset = create_synthetic_trajectories(
            num_trajectories=50,
            horizon=25,
            state_dim=3,
            control_dim=2
        )
        
        assert len(dataset) == 50
        assert dataset.horizon == 25
        assert dataset.feature_dim == 5
        
        # Check trajectory shapes
        traj = dataset[0]
        assert traj.shape == (25, 5)


class TestLMPPIController:
    """Test LMPPI controller functionality."""
    
    @pytest.fixture
    def simple_vae(self):
        """Create a simple trained VAE for testing."""
        vae = TrajectoryVAE(
            input_dim=20 * 6,  # horizon * (state + control)
            latent_dim=4,
            hidden_dims=[32, 16],
            architecture="mlp"
        )
        return vae
    
    def simple_dynamics(self, state, control):
        """Simple linear dynamics for testing."""
        A = torch.eye(state.shape[-1])
        B = torch.ones(state.shape[-1], control.shape[-1]) * 0.1
        return torch.matmul(state.unsqueeze(-2), A).squeeze(-2) + torch.matmul(control.unsqueeze(-2), B.T).squeeze(-2)
    
    def simple_cost(self, state, control):
        """Simple quadratic cost function."""
        return torch.sum(state**2, dim=-1) + 0.1 * torch.sum(control**2, dim=-1)
    
    def test_controller_creation(self, simple_vae):
        """Test controller initialization."""
        controller = LMPPIController(
            vae_model=simple_vae,
            state_dim=4,
            control_dim=2,
            cost_fn=self.simple_cost,
            horizon=20,
            num_samples=10
        )
        
        assert controller.state_dim == 4
        assert controller.control_dim == 2
        assert controller.horizon == 20
    
    def test_controller_solve(self, simple_vae):
        """Test controller solve method."""
        controller = LMPPIController(
            vae_model=simple_vae,
            state_dim=4,
            control_dim=2,
            cost_fn=self.simple_cost,
            horizon=20,
            num_samples=10
        )
        
        initial_state = torch.randn(1, 4)
        control_sequence = controller.solve(initial_state, num_iterations=3)
        
        assert control_sequence.shape == (1, 20, 2)
        assert not torch.isnan(control_sequence).any()
    
    def test_controller_step(self, simple_vae):
        """Test MPC-style stepping."""
        controller = LMPPIController(
            vae_model=simple_vae,
            state_dim=4,
            control_dim=2,
            cost_fn=self.simple_cost,
            horizon=20,
            num_samples=10
        )
        
        state = torch.randn(1, 4)
        control = controller.step(state)
        
        assert control.shape == (1, 2)
        assert not torch.isnan(control).any()
    
    def test_reference_trajectory(self, simple_vae):
        """Test setting reference trajectory."""
        controller = LMPPIController(
            vae_model=simple_vae,
            state_dim=4,
            control_dim=2,
            cost_fn=self.simple_cost,
            horizon=20,
            num_samples=10
        )
        
        ref_trajectory = torch.randn(20, 6)  # horizon x (state + control)
        controller.set_reference_trajectory(ref_trajectory)
        
        # Check that base latent was updated
        assert controller.base_latent.shape == (1, 4)
        assert not torch.isnan(controller.base_latent).any()


class TestLMPPITrainer:
    """Test LMPPI training functionality."""
    
    @pytest.fixture
    def training_setup(self):
        """Create training setup."""
        # Generate small dataset
        trajectories = create_synthetic_trajectories(
            num_trajectories=50,
            horizon=10,
            state_dim=2,
            control_dim=1
        )
        
        train_data, val_data = trajectories.split(train_ratio=0.8)
        
        # Create simple model
        vae = TrajectoryVAE(
            input_dim=30,  # 10 * 3
            latent_dim=4,
            hidden_dims=[16, 8],
            architecture="mlp"
        )
        
        return vae, train_data, val_data
    
    def test_trainer_creation(self, training_setup):
        """Test trainer initialization."""
        vae, train_data, val_data = training_setup
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = LMPPITrainer(
                model=vae,
                train_dataset=train_data,
                val_dataset=val_data,
                batch_size=8,
                save_dir=temp_dir
            )
            
            assert trainer.model == vae
            assert trainer.train_dataset == train_data
            assert trainer.val_dataset == val_data
    
    def test_short_training(self, training_setup):
        """Test a short training run."""
        vae, train_data, val_data = training_setup
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = LMPPITrainer(
                model=vae,
                train_dataset=train_data,
                val_dataset=val_data,
                batch_size=8,
                save_dir=temp_dir,
                log_interval=10,
                validation_interval=20
            )
            
            # Short training
            metrics = trainer.train(num_epochs=2)
            
            assert 'train_loss' in metrics
            assert len(metrics['train_loss']) > 0
            assert all(loss >= 0 for loss in metrics['train_loss'])


class TestConfiguration:
    """Test configuration management."""
    
    def test_vae_config(self):
        """Test VAE configuration."""
        config = VAEConfig(
            input_dim=100,
            latent_dim=8,
            architecture="mlp"
        )
        
        config.validate()
        assert config.input_dim == 100
        assert config.latent_dim == 8
    
    def test_controller_config(self):
        """Test controller configuration."""
        config = ControllerConfig(
            state_dim=4,
            control_dim=2,
            horizon=20
        )
        
        config.validate()
        assert config.state_dim == 4
        assert config.control_dim == 2
    
    def test_full_config(self):
        """Test complete LMPPI configuration."""
        config = pendulum_config()
        config.validate()
        
        assert config.vae.latent_dim == 6
        assert config.controller.state_dim == 2
        assert config.controller.control_dim == 1
    
    def test_config_save_load(self):
        """Test configuration save/load."""
        config = pendulum_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            
            # Save and load
            config.save(config_path)
            loaded_config = LMPPIConfig.load(config_path)
            
            # Check equality
            assert loaded_config.vae.latent_dim == config.vae.latent_dim
            assert loaded_config.controller.state_dim == config.controller.state_dim
            assert loaded_config.name == config.name
    
    def test_predefined_configs(self):
        """Test predefined configurations."""
        # Test pendulum config
        pendulum = get_config("pendulum")
        pendulum.validate()
        assert pendulum.controller.state_dim == 2
        
        # Test quadrotor config
        quadrotor = get_config("quadrotor")
        quadrotor.validate()
        assert quadrotor.controller.state_dim == 12
        
        # Test robotic arm config
        arm = get_config("robotic_arm", num_joints=6)
        arm.validate()
        assert arm.controller.state_dim == 12  # 6 * 2 (position + velocity)


class TestUtilities:
    """Test utility functions."""
    
    def test_trajectory_conversion(self):
        """Test trajectory tensor conversion."""
        # Create test trajectory
        trajectory_np = np.random.randn(20, 5)
        
        # Convert to tensor
        trajectory_tensor = trajectory_to_tensor(
            trajectory_np, state_dim=3, control_dim=2
        )
        
        assert trajectory_tensor.shape == (20, 5)
        assert isinstance(trajectory_tensor, torch.Tensor)
        
        # Convert back
        states, controls = tensor_to_trajectory(
            trajectory_tensor, state_dim=3, control_dim=2
        )
        
        assert states.shape == (20, 3)
        assert controls.shape == (20, 2)
        assert isinstance(states, np.ndarray)
        assert isinstance(controls, np.ndarray)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_simple(self):
        """Test complete LMPPI workflow with simple system."""
        # Generate data
        dataset = create_synthetic_trajectories(
            num_trajectories=100,
            horizon=15,
            state_dim=2,
            control_dim=1
        )
        
        train_data, val_data = dataset.split()
        
        # Create and train VAE
        vae = TrajectoryVAE(
            input_dim=45,  # 15 * 3
            latent_dim=4,
            hidden_dims=[32, 16],
            architecture="mlp"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = LMPPITrainer(
                model=vae,
                train_dataset=train_data,
                val_dataset=val_data,
                batch_size=16,
                save_dir=temp_dir
            )
            
            # Quick training
            trainer.train(num_epochs=3)
            
            # Create controller
            def test_cost(state, control):
                return torch.sum(state**2, dim=-1) + torch.sum(control**2, dim=-1)
            
            controller = LMPPIController(
                vae_model=vae,
                state_dim=2,
                control_dim=1,
                cost_fn=test_cost,
                horizon=15,
                num_samples=20
            )
            
            # Test control
            initial_state = torch.randn(1, 2)
            control = controller.step(initial_state)
            
            assert control.shape == (1, 1)
            assert not torch.isnan(control).any()
    
    def test_configuration_workflow(self):
        """Test workflow using configuration."""
        config = pendulum_config()
        
        # Modify for testing
        config.training.num_epochs = 2
        config.training.batch_size = 16
        config.controller.num_samples = 20
        
        config.validate()
        
        # Generate compatible data
        dataset = create_synthetic_trajectories(
            num_trajectories=50,
            horizon=config.controller.horizon,
            state_dim=config.controller.state_dim,
            control_dim=config.controller.control_dim
        )
        
        # This tests that configuration is compatible with data generation
        assert dataset.horizon == config.controller.horizon
        assert dataset.feature_dim == (config.controller.state_dim + config.controller.control_dim)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
