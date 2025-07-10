"""
MLP model for spectral abundance prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SpectralMLP(nn.Module):
    """
    Multi-layer perceptron for predicting spectral abundances.
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 output_dim: int = 20,
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Number of input features (spectral bins)
            output_dim: Number of output components (spectral lines)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return self.network(x)


class SpectralCNN(nn.Module):
    """
    1D CNN for spectral abundance prediction (extensible alternative).
    """
    
    def __init__(self,
                 input_dim: int = 512,
                 output_dim: int = 20,
                 channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [7, 5, 3],
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.1):
        """
        Initialize the CNN.
        
        Args:
            input_dim: Number of input features (spectral bins)
            output_dim: Number of output components
            channels: List of channel dimensions for conv layers
            kernel_sizes: List of kernel sizes for conv layers
            hidden_dim: Hidden dimension for final MLP
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Convolutional layers
        conv_layers = []
        in_channels = 1
        
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate the dimension after convolution
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_dim)
            conv_output = self.conv_layers(dummy_input)
            conv_output_dim = conv_output.numel()
        
        # Final MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(conv_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Add channel dimension for 1D convolution
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Flatten and apply MLP
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        
        return x


# Model registry for easy extensibility
MODEL_REGISTRY = {
    'mlp': SpectralMLP,
    'cnn': SpectralCNN,
}


def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    Create a model from the registry.
    
    Args:
        model_type: Type of model ('mlp', 'cnn')
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](**kwargs)


def register_model(name: str, model_class: type):
    """
    Register a new model class.
    
    Args:
        name: Name of the model
        model_class: Model class to register
    """
    MODEL_REGISTRY[name] = model_class