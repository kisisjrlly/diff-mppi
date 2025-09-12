"""
Neural Network Models for LMPPI Latent Space Learning

This module implements the core neural network architectures for learning
latent space representations of feasible trajectories:

1. TrajectoryEncoder: Compresses high-dimensional trajectories to low-dimensional latent vectors
2. TrajectoryDecoder: Reconstructs trajectories from latent vectors  
3. TrajectoryVAE: Complete Variational Autoencoder combining encoder and decoder

The models support various architectures (MLP, LSTM, CNN) to handle different
trajectory characteristics and temporal dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import math


class TrajectoryEncoder(nn.Module):
    """
    Encoder network that compresses trajectories to latent space.
    
    Supports multiple architectures:
    - MLP: Simple feedforward network (flattens trajectory)
    - LSTM: Recurrent network preserving temporal structure
    - CNN: Convolutional network for local pattern extraction
    """
    
    def __init__(
        self,
        input_dim: int,  # trajectory_dim = horizon * (state_dim + control_dim)
        latent_dim: int,
        hidden_dims: list = [512, 256, 128],
        architecture: Literal["mlp", "lstm", "cnn"] = "mlp",
        horizon: Optional[int] = None,
        feature_dim: Optional[int] = None,  # state_dim + control_dim
        dropout: float = 0.1
    ):
        """
        Initialize trajectory encoder.
        
        Args:
            input_dim: Total input dimension (horizon * feature_dim for MLP)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            architecture: Network architecture type
            horizon: Time horizon (required for LSTM/CNN)
            feature_dim: Feature dimension per timestep (required for LSTM/CNN)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.horizon = horizon
        self.feature_dim = feature_dim
        
        if architecture in ["lstm", "cnn"] and (horizon is None or feature_dim is None):
            raise ValueError(f"Architecture '{architecture}' requires horizon and feature_dim")
            
        if architecture == "mlp":
            self._build_mlp_encoder(hidden_dims, dropout)
        elif architecture == "lstm":
            self._build_lstm_encoder(hidden_dims, dropout)
        elif architecture == "cnn":
            self._build_cnn_encoder(hidden_dims, dropout)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def _build_mlp_encoder(self, hidden_dims: list, dropout: float):
        """Build MLP encoder architecture."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layers for VAE (mean and log variance)
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)
    
    def _build_lstm_encoder(self, hidden_dims: list, dropout: float):
        """Build LSTM encoder architecture."""
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # MLP head for further processing
        mlp_layers = []
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.mlp_head = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        
        # Output layers for VAE
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)
    
    def _build_cnn_encoder(self, hidden_dims: list, dropout: float):
        """Build 1D CNN encoder architecture."""
        # 1D Convolutional layers
        conv_layers = []
        in_channels = self.feature_dim
        
        for i, out_channels in enumerate(hidden_dims[:2]):  # Use first 2 dims for conv
            kernel_size = 5 if i == 0 else 3
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate size after convolution
        conv_output_size = hidden_dims[1] * (self.horizon // (2 ** len(hidden_dims[:2])))
        
        # MLP head
        mlp_layers = []
        prev_dim = conv_output_size
        
        for hidden_dim in hidden_dims[2:]:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.mlp_head = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        
        # Output layers for VAE
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input trajectory tensor
               - MLP: [batch_size, input_dim] (flattened trajectory)
               - LSTM/CNN: [batch_size, horizon, feature_dim]
               
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        if self.architecture == "mlp":
            # Flatten if not already flattened
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            features = self.feature_extractor(x)
            
        elif self.architecture == "lstm":
            # x: [batch_size, horizon, feature_dim]
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use final hidden state
            features = self.mlp_head(h_n[-1])  # Use last layer's hidden state
            
        elif self.architecture == "cnn":
            # x: [batch_size, horizon, feature_dim] -> [batch_size, feature_dim, horizon]
            x = x.transpose(1, 2)
            conv_out = self.conv_layers(x)
            # Flatten for MLP
            features = self.mlp_head(conv_out.view(conv_out.size(0), -1))
        
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return mu, logvar


class TrajectoryDecoder(nn.Module):
    """
    Decoder network that reconstructs trajectories from latent vectors.
    
    Architecture mirrors the encoder to ensure compatibility.
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,  # trajectory_dim = horizon * (state_dim + control_dim)
        hidden_dims: list = [128, 256, 512],
        architecture: Literal["mlp", "lstm", "cnn"] = "mlp",
        horizon: Optional[int] = None,
        feature_dim: Optional[int] = None,  # state_dim + control_dim
        dropout: float = 0.1
    ):
        """
        Initialize trajectory decoder.
        
        Args:
            latent_dim: Dimension of latent space
            output_dim: Total output dimension (horizon * feature_dim for MLP)
            hidden_dims: List of hidden layer dimensions (reversed from encoder)
            architecture: Network architecture type
            horizon: Time horizon (required for LSTM/CNN)
            feature_dim: Feature dimension per timestep (required for LSTM/CNN)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.architecture = architecture
        self.horizon = horizon
        self.feature_dim = feature_dim
        
        if architecture in ["lstm", "cnn"] and (horizon is None or feature_dim is None):
            raise ValueError(f"Architecture '{architecture}' requires horizon and feature_dim")
            
        if architecture == "mlp":
            self._build_mlp_decoder(hidden_dims, dropout)
        elif architecture == "lstm":
            self._build_lstm_decoder(hidden_dims, dropout)
        elif architecture == "cnn":
            self._build_cnn_decoder(hidden_dims, dropout)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def _build_mlp_decoder(self, hidden_dims: list, dropout: float):
        """Build MLP decoder architecture."""
        layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.decoder = nn.Sequential(*layers)
    
    def _build_lstm_decoder(self, hidden_dims: list, dropout: float):
        """Build LSTM decoder architecture."""
        # Initial MLP to expand latent vector
        mlp_layers = []
        prev_dim = self.latent_dim
        
        for hidden_dim in hidden_dims[:-1]:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.mlp_head = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        
        # LSTM for temporal generation
        self.lstm_hidden_dim = hidden_dims[-1]
        self.lstm = nn.LSTM(
            input_size=prev_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.lstm_hidden_dim, self.feature_dim)
    
    def _build_cnn_decoder(self, hidden_dims: list, dropout: float):
        """Build CNN decoder architecture."""
        # Initial MLP
        mlp_layers = []
        prev_dim = self.latent_dim
        
        # Calculate target size for reshape
        conv_output_size = hidden_dims[0] * (self.horizon // (2 ** len(hidden_dims[:2])))
        
        for hidden_dim in hidden_dims[2:]:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        mlp_layers.append(nn.Linear(prev_dim, conv_output_size))
        self.mlp_head = nn.Sequential(*mlp_layers)
        
        # Transposed convolution layers
        deconv_layers = []
        in_channels = hidden_dims[0]
        
        for i, out_channels in enumerate(reversed(hidden_dims[1:3])):
            deconv_layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
            
        # Final layer to get correct feature dimension
        deconv_layers.append(
            nn.ConvTranspose1d(in_channels, self.feature_dim, 3, padding=1)
        )
        
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.conv_output_size = conv_output_size
        self.conv_channels = hidden_dims[0]
        self.conv_length = self.horizon // (2 ** len(hidden_dims[:2]))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            Reconstructed trajectory:
            - MLP: [batch_size, output_dim] (flattened)
            - LSTM/CNN: [batch_size, horizon, feature_dim]
        """
        if self.architecture == "mlp":
            return self.decoder(z)
            
        elif self.architecture == "lstm":
            # Expand latent vector through MLP
            expanded = self.mlp_head(z)  # [batch_size, hidden_dim]
            
            # Repeat for each timestep
            lstm_input = expanded.unsqueeze(1).repeat(1, self.horizon, 1)
            
            # Pass through LSTM
            lstm_out, _ = self.lstm(lstm_input)
            
            # Generate output for each timestep
            output = self.output_layer(lstm_out)
            
            return output
            
        elif self.architecture == "cnn":
            # MLP to expand to conv input size
            mlp_out = self.mlp_head(z)
            
            # Reshape for conv layers
            conv_input = mlp_out.view(-1, self.conv_channels, self.conv_length)
            
            # Pass through deconv layers
            conv_out = self.deconv_layers(conv_input)
            
            # Transpose back to [batch_size, horizon, feature_dim]
            output = conv_out.transpose(1, 2)
            
            return output


class TrajectoryVAE(nn.Module):
    """
    Complete Variational Autoencoder for trajectory learning.
    
    Combines encoder and decoder with VAE loss computation and sampling.
    This is the main model used for learning latent trajectory representations.
    """
    
    def __init__(
        self,
        config,  # Can be VAEConfig object or individual parameters for backward compatibility
        input_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        hidden_dims: Optional[list] = None,
        architecture: Optional[Literal["mlp", "lstm", "cnn"]] = None,
        horizon: Optional[int] = None,
        feature_dim: Optional[int] = None,  # state_dim + control_dim
        dropout: Optional[float] = None,
        beta: Optional[float] = None  # KL divergence weight
    ):
        """
        Initialize trajectory VAE.
        
        Args:
            config: VAEConfig object or input_dim for backward compatibility
            input_dim: Input trajectory dimension (optional if config is provided)
            latent_dim: Latent space dimension (optional if config is provided)
            hidden_dims: Hidden layer dimensions (optional if config is provided)
            architecture: Network architecture (optional if config is provided)
            horizon: Time horizon (for LSTM/CNN, optional if config is provided)
            feature_dim: Feature dimension per timestep (for LSTM/CNN, optional if config is provided)
            dropout: Dropout probability (optional if config is provided)
            beta: Weight for KL divergence loss (β-VAE, optional if config is provided)
        """
        super().__init__()
        
        # Handle config object or individual parameters
        if hasattr(config, 'input_dim'):  # It's a config object
            cfg = config
            _input_dim = cfg.input_dim
            _latent_dim = cfg.latent_dim
            _hidden_dims = cfg.hidden_dims
            _architecture = cfg.architecture
            _horizon = cfg.horizon
            _feature_dim = cfg.feature_dim
            _dropout = cfg.dropout
            _beta = cfg.beta
        else:  # Backward compatibility - config is actually input_dim
            _input_dim = config
            _latent_dim = latent_dim if latent_dim is not None else 8
            _hidden_dims = hidden_dims if hidden_dims is not None else [512, 256, 128]
            _architecture = architecture if architecture is not None else "mlp"
            _horizon = horizon
            _feature_dim = feature_dim
            _dropout = dropout if dropout is not None else 0.1
            _beta = beta if beta is not None else 1.0
        
        self.latent_dim = _latent_dim
        self.beta = _beta
        self.architecture = _architecture
        
        # Encoder
        self.encoder = TrajectoryEncoder(
            input_dim=_input_dim,
            latent_dim=_latent_dim,
            hidden_dims=_hidden_dims,
            architecture=_architecture,
            horizon=_horizon,
            feature_dim=_feature_dim,
            dropout=_dropout
        )
        
        # Decoder
        decoder_hidden_dims = list(reversed(_hidden_dims))
        self.decoder = TrajectoryDecoder(
            latent_dim=_latent_dim,
            output_dim=_input_dim,
            hidden_dims=decoder_hidden_dims,
            architecture=_architecture,
            horizon=_horizon,
            feature_dim=_feature_dim,
            dropout=_dropout
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory to latent distribution parameters."""
        return self.encoder(x)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE sampling.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to trajectory."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through VAE.
        
        Args:
            x: Input trajectory
            
        Returns:
            reconstruction: Reconstructed trajectory
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample trajectories from prior distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device for computation
            
        Returns:
            Generated trajectories
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        reconstruction: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None, 
        logvar: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss components.
        
        Args:
            x: Original trajectory
            reconstruction: Reconstructed trajectory (optional, will compute if not provided)
            mu: Latent mean (optional, will compute if not provided)
            logvar: Latent log variance (optional, will compute if not provided)
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # If components not provided, compute them
        if reconstruction is None or mu is None or logvar is None:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decode(z)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic latent representation (using mean).
        
        Args:
            x: Input trajectory
            
        Returns:
            Latent representation (mean)
        """
        mu, _ = self.encode(x)
        return mu
