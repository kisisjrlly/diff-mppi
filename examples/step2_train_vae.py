"""
Step 2: Train VAE for LMPPI
训练变分自编码器 (VAE) 用于轨迹表示学习

This script loads saved trajectory data and trains a VAE model for latent space representation.
Supports long training sessions with comprehensive monitoring and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add the parent directory to the path so we can import diff_mppi
import sys
sys.path.append(str(Path(__file__).parent.parent))

from diff_mppi.lmppi import TrajectoryVAE, TrajectoryDataset
from diff_mppi.lmppi.config import VAEConfig


def load_dataset(data_path, metadata_path=None):
    """Load trajectory dataset from file."""
    print(f"Loading dataset from: {data_path}")
    
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print("Loaded metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Trajectory shape: {trajectories[0].shape}")
    
    return trajectories, metadata


def create_data_loaders(trajectories, train_ratio=0.8, batch_size=32, device="cpu"):
    """Create train and validation data loaders."""
    print("Creating data loaders...")
    
    # Convert to tensor and flatten trajectories for MLP architecture
    trajectory_tensors = []
    for traj in trajectories:
        # traj shape: [horizon, state_dim + control_dim]
        # Flatten to: [horizon * (state_dim + control_dim)]
        flattened = torch.tensor(traj, dtype=torch.float32).flatten()
        trajectory_tensors.append(flattened)
    
    data_tensor = torch.stack(trajectory_tensors)
    print(f"Data tensor shape: {data_tensor.shape}")
    
    # Split into train/validation
    num_samples = len(data_tensor)
    num_train = int(num_samples * train_ratio)
    
    # Shuffle indices
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_data = data_tensor[train_indices]
    val_data = data_tensor[val_indices]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create data loaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_data.shape[1]


def create_vae_model(input_dim, latent_dim=16, hidden_dims=None, architecture="mlp"):
    """Create VAE model with specified architecture."""
    if hidden_dims is None:
        hidden_dims = [512, 256, 128, 64]
    
    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        architecture=architecture,
        dropout=0.1,
        beta=1.0
    )
    
    model = TrajectoryVAE(config)
    
    print(f"Created VAE model:")
    print(f"  Input dim: {input_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Architecture: {architecture}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    for batch_data, in tqdm(train_loader, desc="Training", leave=False):
        batch_data = batch_data.to(device)
        
        # Forward pass
        loss, recon_loss, kl_loss = model.compute_loss(batch_data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }


def validate_epoch(model, val_loader, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data, in tqdm(val_loader, desc="Validation", leave=False):
            batch_data = batch_data.to(device)
            
            # Forward pass
            loss, recon_loss, kl_loss = model.compute_loss(batch_data)
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir, is_best=False):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint at epoch {epoch}")
    
    # Save periodic checkpoint
    if epoch % 50 == 0:
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)


def plot_training_curves(train_metrics, val_metrics, save_dir):
    """Plot and save training curves."""
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total loss
    axes[0].plot(epochs, train_metrics['loss'], 'b-', label='Train', alpha=0.7)
    axes[0].plot(epochs, val_metrics['loss'], 'r-', label='Validation', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Reconstruction loss
    axes[1].plot(epochs, train_metrics['recon_loss'], 'b-', label='Train', alpha=0.7)
    axes[1].plot(epochs, val_metrics['recon_loss'], 'r-', label='Validation', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # KL loss
    axes[2].plot(epochs, train_metrics['kl_loss'], 'b-', label='Train', alpha=0.7)
    axes[2].plot(epochs, val_metrics['kl_loss'], 'r-', label='Validation', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Loss')
    axes[2].set_title('KL Divergence Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def train_vae(model, train_loader, val_loader, save_dir, num_epochs=200, 
              learning_rate=1e-3, patience=20, device="cpu"):
    """Train VAE model with comprehensive monitoring."""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
    )
    
    # Training metrics
    train_metrics = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    val_metrics = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    final_epoch = num_epochs  # Default final epoch
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")
    
    for epoch in range(1, num_epochs + 1):
        final_epoch = epoch  # Update final epoch in each iteration
        start_time = time.time()
        
        # Train
        train_epoch_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_epoch_metrics = validate_epoch(model, val_loader, device)
        
        # Record metrics
        for key in train_metrics.keys():
            train_metrics[key].append(train_epoch_metrics[key])
            val_metrics[key].append(val_epoch_metrics[key])
        
        # Scheduler step
        scheduler.step(val_epoch_metrics['loss'])
        
        # Check for best model
        is_best = val_epoch_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_epoch_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, 
                       {'train': train_metrics, 'val': val_metrics}, 
                       save_dir, is_best=is_best)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:3d}/{num_epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_epoch_metrics['loss']:.4f} | "
              f"Val Loss: {val_epoch_metrics['loss']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Plot curves periodically
        if epoch % 10 == 0:
            plot_training_curves(train_metrics, val_metrics, save_dir)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Final plots and metrics
    plot_training_curves(train_metrics, val_metrics, save_dir)
    
    # Save final metrics
    final_metrics = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'total_epochs': final_epoch,
        'final_lr': optimizer.param_groups[0]['lr']
    }
    
    with open(os.path.join(save_dir, 'training_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nTraining completed after {final_epoch} epochs")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    return model, final_metrics


def main():
    parser = argparse.ArgumentParser(description='Train VAE for LMPPI')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to trajectory data file (.pkl)')
    parser.add_argument('--save_dir', type=str, default='./trained_models',
                       help='Directory to save trained model (default: ./trained_models)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--latent_dim', type=int, default=16,
                       help='Latent space dimension (default: 16)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', 
                       default=[512, 256, 128, 64],
                       help='Hidden layer dimensions (default: 512 256 128 64)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training data ratio (default: 0.8)')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (default: 30)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cpu, cuda, or auto (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load dataset
    metadata_path = args.data_path.replace('.pkl', '_metadata.pkl')
    trajectories, metadata = load_dataset(args.data_path, metadata_path)
    
    # Create data loaders
    train_loader, val_loader, input_dim = create_data_loaders(
        trajectories, 
        train_ratio=args.train_ratio, 
        batch_size=args.batch_size,
        device=device
    )
    
    # Create model
    model, config = create_vae_model(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    )
    
    # Create timestamped save directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'vae_training_{timestamp}')
    
    # Save config
    os.makedirs(save_dir, exist_ok=True)
    config_dict = {
        'input_dim': config.input_dim,
        'latent_dim': config.latent_dim,
        'hidden_dims': config.hidden_dims,
        'architecture': config.architecture,
        'dropout': config.dropout,
        'beta': config.beta
    }
    
    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save training arguments
    with open(os.path.join(save_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train model
    trained_model, metrics = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=device
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved in: {save_dir}")
    print(f"Best model: {os.path.join(save_dir, 'checkpoint_best.pth')}")
    print(f"\nYou can now test the model using:")
    print(f"python step3_test_lmppi.py --model_path {os.path.join(save_dir, 'checkpoint_best.pth')}")


if __name__ == "__main__":
    main()
