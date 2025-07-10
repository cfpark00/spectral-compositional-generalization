"""
Data loading and saving utilities.
"""

import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict


def save_data(data_dict: Dict[str, torch.Tensor], save_dir: str):
    """Save generated data to files."""
    os.makedirs(save_dir, exist_ok=True)
    
    for key, tensor in data_dict.items():
        torch.save(tensor, os.path.join(save_dir, f"{key}.pt"))
    
    print(f"Data saved to {save_dir}")
    for key, tensor in data_dict.items():
        print(f"  {key}: {tensor.shape}")


def load_data(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pre-generated data from files."""
    train_spectra = torch.load(os.path.join(data_dir, "train_spectra.pt"))
    train_abundances = torch.load(os.path.join(data_dir, "train_abundances.pt"))
    test_spectra = torch.load(os.path.join(data_dir, "test_spectra.pt"))
    test_abundances = torch.load(os.path.join(data_dir, "test_abundances.pt"))
    
    return train_spectra, train_abundances, test_spectra, test_abundances


def create_dataloaders(train_spectra: torch.Tensor, train_abundances: torch.Tensor,
                      test_spectra: torch.Tensor, test_abundances: torch.Tensor,
                      batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders."""
    train_dataset = TensorDataset(train_spectra, train_abundances)
    test_dataset = TensorDataset(test_spectra, test_abundances)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader