"""
Models module for spectral abundance prediction.
"""

from .mlp import SpectralMLP, SpectralCNN, create_model, register_model, MODEL_REGISTRY

__all__ = ['SpectralMLP', 'SpectralCNN', 'create_model', 'register_model', 'MODEL_REGISTRY']