"""
Training Module for LMPPI VAE Models

This module provides comprehensive training utilities for learning latent
trajectory representations using Variational Autoencoders. It includes:

1. LMPPITrainer: Main training class with full training loop
2. Training metrics and logging
3. Model checkpointing and loading
4. Validation and evaluation utilities

The trainer supports various optimization strategies and hyperparameter
configurations for robust VAE training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt

from .models import TrajectoryVAE
from .data import TrajectoryDataset


class LMPPITrainer:
    """
    Trainer class for LMPPI VAE models.
    
    Provides comprehensive training functionality including:
    - Training loop with validation
    - Metrics tracking and logging
    - Model checkpointing
    - Learning rate scheduling
    - Early stopping
    """
    
    def __init__(
        self,
        model: TrajectoryVAE,
        train_dataset: TrajectoryDataset,
        val_dataset: Optional[TrajectoryDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
        validation_interval: int = 1000,
        checkpoint_interval: int = 5000
    ):
        """
        Initialize LMPPI trainer.
        
        Args:
            model: TrajectoryVAE model to train
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device for computation
            save_dir: Directory for saving checkpoints
            log_interval: Steps between logging
            validation_interval: Steps between validation
            checkpoint_interval: Steps between checkpoints
        """
        self.model = model.to(device)
        self.device = device
        
        # Datasets and dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=(device == "cuda")
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=(device == "cuda")
            )
        else:
            self.val_loader = None
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging and checkpointing
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.checkpoint_interval = checkpoint_interval
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 20,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training metrics dictionary
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on {len(self.train_dataset)} samples")
        if self.val_dataset:
            print(f"Validation on {len(self.val_dataset)} samples")
        print(f"Device: {self.device}")
        
        early_stopping_counter = 0
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            epoch_start_time = time.time()
            train_metrics = self._train_epoch()
            
            # Validation phase
            if self.val_loader and (epoch % (self.validation_interval // len(self.train_loader)) == 0):
                val_metrics = self._validate()
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['val_loss'])
                
                # Early stopping check
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    early_stopping_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        os.path.join(self.save_dir, "best_model.pth"),
                        is_best=True
                    )
                else:
                    early_stopping_counter += 1
                
                # Log validation metrics
                self._log_metrics(val_metrics, "validation")
                
                # Early stopping
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Periodic checkpointing
            if epoch % (self.checkpoint_interval // len(self.train_loader)) == 0:
                checkpoint_path = os.path.join(
                    self.save_dir, f"checkpoint_epoch_{epoch}.pth"
                )
                self.save_checkpoint(checkpoint_path)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}: {epoch_time:.2f}s, "
                  f"Train Loss: {train_metrics['train_loss']:.6f}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.save_dir, "final_model.pth")
        )
        
        return dict(self.metrics)
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            trajectories = batch.to(self.device)
            
            # Forward pass
            reconstruction, mu, logvar = self.model(trajectories)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = self.model.compute_loss(
                trajectories, reconstruction, mu, logvar
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_metrics['train_loss'] += total_loss.item()
            epoch_metrics['train_recon_loss'] += recon_loss.item()
            epoch_metrics['train_kl_loss'] += kl_loss.item()
            
            self.current_step += 1
            
            # Periodic logging
            if self.current_step % self.log_interval == 0:
                step_metrics = {
                    'step': self.current_step,
                    'train_loss': total_loss.item(),
                    'train_recon_loss': recon_loss.item(),
                    'train_kl_loss': kl_loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self._log_metrics(step_metrics, "step")
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            self.metrics[key].append(epoch_metrics[key])
        
        return dict(epoch_metrics)
    
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_metrics = defaultdict(float)
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                trajectories = batch.to(self.device)
                
                # Forward pass
                reconstruction, mu, logvar = self.model(trajectories)
                
                # Compute loss
                total_loss, recon_loss, kl_loss = self.model.compute_loss(
                    trajectories, reconstruction, mu, logvar
                )
                
                # Track metrics
                val_metrics['val_loss'] += total_loss.item()
                val_metrics['val_recon_loss'] += recon_loss.item()
                val_metrics['val_kl_loss'] += kl_loss.item()
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
            self.metrics[key].append(val_metrics[key])
        
        return dict(val_metrics)
    
    def _log_metrics(self, metrics: Dict[str, float], phase: str):
        """Log metrics."""
        if phase == "step":
            print(f"Step {metrics['step']}: "
                  f"Loss={metrics['train_loss']:.6f}, "
                  f"Recon={metrics['train_recon_loss']:.6f}, "
                  f"KL={metrics['train_kl_loss']:.6f}, "
                  f"LR={metrics['lr']:.2e}")
        elif phase == "validation":
            print(f"Validation: "
                  f"Loss={metrics['val_loss']:.6f}, "
                  f"Recon={metrics['val_recon_loss']:.6f}, "
                  f"KL={metrics['val_kl_loss']:.6f}")
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'metrics': dict(self.metrics),
            'model_config': {
                'input_dim': self.model.encoder.input_dim,
                'latent_dim': self.model.latent_dim,
                'architecture': self.model.architecture,
                'beta': self.model.beta
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        if is_best:
            print("New best model saved!")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_step = checkpoint['current_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'metrics' in checkpoint:
            self.metrics = defaultdict(list, checkpoint['metrics'])
        
        print(f"Checkpoint loaded: {path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")
    
    def evaluate_reconstruction(
        self, 
        num_samples: int = 100,
        save_plots: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality on validation set.
        
        Args:
            num_samples: Number of samples to evaluate
            save_plots: Whether to save reconstruction plots
            
        Returns:
            Evaluation metrics
        """
        if self.val_loader is None:
            print("No validation set available for evaluation")
            return {}
        
        self.model.eval()
        
        all_original = []
        all_reconstructed = []
        all_latent = []
        
        with torch.no_grad():
            samples_collected = 0
            for batch in self.val_loader:
                if samples_collected >= num_samples:
                    break
                
                trajectories = batch.to(self.device)
                reconstruction, mu, logvar = self.model(trajectories)
                
                # Sample from latent distribution
                z = self.model.reparameterize(mu, logvar)
                
                all_original.append(trajectories.cpu())
                all_reconstructed.append(reconstruction.cpu())
                all_latent.append(z.cpu())
                
                samples_collected += trajectories.size(0)
        
        # Concatenate all samples
        original = torch.cat(all_original, dim=0)[:num_samples]
        reconstructed = torch.cat(all_reconstructed, dim=0)[:num_samples]
        latent = torch.cat(all_latent, dim=0)[:num_samples]
        
        # Compute metrics
        mse = torch.mean((original - reconstructed) ** 2).item()
        mae = torch.mean(torch.abs(original - reconstructed)).item()
        
        # Latent space statistics
        latent_mean = torch.mean(latent, dim=0)
        latent_std = torch.std(latent, dim=0)
        
        metrics = {
            'reconstruction_mse': mse,
            'reconstruction_mae': mae,
            'latent_mean_norm': torch.norm(latent_mean).item(),
            'latent_std_mean': torch.mean(latent_std).item()
        }
        
        print("Reconstruction Evaluation:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        # Save plots if requested
        if save_plots:
            self._plot_reconstructions(original, reconstructed, latent)
        
        return metrics
    
    def _plot_reconstructions(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        latent: torch.Tensor,
        num_plots: int = 4
    ):
        """Plot reconstruction examples and latent space visualization."""
        num_plots = min(num_plots, original.size(0))
        
        # Reconstruction examples
        fig, axes = plt.subplots(num_plots, 2, figsize=(12, 3 * num_plots))
        if num_plots == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_plots):
            # Original trajectory
            if original.dim() == 3:  # [batch, horizon, features]
                orig_traj = original[i].numpy()
                recon_traj = reconstructed[i].numpy()
            else:  # Flattened
                horizon = self.model.encoder.horizon or 20
                feature_dim = original.size(1) // horizon
                orig_traj = original[i].view(horizon, feature_dim).numpy()
                recon_traj = reconstructed[i].view(horizon, feature_dim).numpy()
            
            # Plot original
            axes[i, 0].plot(orig_traj)
            axes[i, 0].set_title(f"Original Trajectory {i+1}")
            axes[i, 0].set_xlabel("Time Step")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].grid(True)
            
            # Plot reconstruction
            axes[i, 1].plot(recon_traj)
            axes[i, 1].set_title(f"Reconstructed Trajectory {i+1}")
            axes[i, 1].set_xlabel("Time Step")
            axes[i, 1].set_ylabel("Value")
            axes[i, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "reconstructions.png"))
        plt.close()
        
        # Latent space visualization (2D projection if latent_dim > 2)
        if latent.size(1) >= 2:
            plt.figure(figsize=(8, 6))
            latent_np = latent.numpy()
            
            if latent.size(1) == 2:
                plt.scatter(latent_np[:, 0], latent_np[:, 1], alpha=0.6)
                plt.xlabel("Latent Dimension 1")
                plt.ylabel("Latent Dimension 2")
            else:
                # Use first two dimensions
                plt.scatter(latent_np[:, 0], latent_np[:, 1], alpha=0.6)
                plt.xlabel("Latent Dimension 1")
                plt.ylabel("Latent Dimension 2")
                plt.title(f"Latent Space (2D projection, full dim: {latent.size(1)})")
            
            plt.title("Latent Space Distribution")
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "latent_space.png"))
            plt.close()
        
        print(f"Plots saved to {self.save_dir}/")
    
    def plot_training_curves(self):
        """Plot training curves."""
        if not self.metrics:
            print("No metrics to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total loss
        if 'train_loss' in self.metrics:
            axes[0].plot(self.metrics['train_loss'], label='Train Loss')
        if 'val_loss' in self.metrics:
            axes[0].plot(self.metrics['val_loss'], label='Val Loss')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Reconstruction loss
        if 'train_recon_loss' in self.metrics:
            axes[1].plot(self.metrics['train_recon_loss'], label='Train Recon')
        if 'val_recon_loss' in self.metrics:
            axes[1].plot(self.metrics['val_recon_loss'], label='Val Recon')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # KL loss
        if 'train_kl_loss' in self.metrics:
            axes[2].plot(self.metrics['train_kl_loss'], label='Train KL')
        if 'val_kl_loss' in self.metrics:
            axes[2].plot(self.metrics['val_kl_loss'], label='Val KL')
        axes[2].set_title('KL Divergence Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_curves.png"))
        plt.close()
        
        print(f"Training curves saved to {self.save_dir}/training_curves.png")
