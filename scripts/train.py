"""
Clean training script - thin wrapper around src logic.
"""

import argparse
import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import load_multiple_configs, set_seed
from src.data_utils import load_data, create_dataloaders
from src.training import get_device, train_model


def main():
    parser = argparse.ArgumentParser(description="Train spectral abundance prediction model")
    parser.add_argument("--data", type=str, required=True, help="Data config")
    parser.add_argument("--distribution", type=str, required=True, help="Distribution config")
    parser.add_argument("--training", type=str, required=True, help="Training config")
    parser.add_argument("--data-dir", type=str, help="Directory containing pre-generated data")
    parser.add_argument("--exp-dir", type=str, help="Experiment directory (data will be in exp-dir/data)")
    parser.add_argument("--training-seed", type=int, help="Seed for training (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_multiple_configs(args.data, args.distribution, args.training)
    
    # Use training seed from command line if provided, otherwise from config
    training_seed = args.training_seed if args.training_seed is not None else config.get("seed", 42)
    set_seed(training_seed)
    print(f"Training seed: {training_seed}")
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif args.exp_dir:
        data_dir = os.path.join(args.exp_dir, "data")
    else:
        # Default to ./data/generated/experiment_name
        data_dir = os.path.join("./data", "generated", config["experiment_name"])
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Please run generate_data.py first.")
    
    # Determine experiment directory for training outputs
    if args.exp_dir:
        train_exp_dir = args.exp_dir
    else:
        # Default to ./data/results/experiment_name  
        train_exp_dir = os.path.join("./data", "results", config["experiment_name"])
    
    # Create training directory structure
    train_dir = os.path.join(train_exp_dir, "train")
    logs_dir = os.path.join(train_dir, "logs")
    ckpts_dir = os.path.join(train_dir, "ckpts")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpts_dir, exist_ok=True)
    
    # Auto-detect device
    device = get_device(config.get("device", "auto"))
    
    print(f"Training directory: {train_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    
    # Load data
    train_spectra, train_abundances, test_spectra, test_abundances = load_data(data_dir)
    print(f"Training samples: {train_spectra.shape[0]}")
    print(f"Test samples: {test_spectra.shape[0]}")
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        train_spectra, train_abundances, test_spectra, test_abundances, 
        config["batch_size"], config.get("num_workers", 4)
    )
    
    # Train model with new directory structure
    model = train_model(config, train_loader, test_loader, device, train_dir, 
                       test_spectra, test_abundances)


if __name__ == "__main__":
    main()