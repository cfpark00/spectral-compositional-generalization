"""
Clean data generation script using spectral and distribution configs.
"""

import argparse
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.spectra_generator import load_spectral_config, generate_split_data
from src.data_utils import save_data
import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate spectral data")
    
    # New-style arguments
    parser.add_argument("--spectra-config", type=str, 
                       help="Spectral configuration file (e.g., synthetic_spectra.yaml)")
    parser.add_argument("--distribution-config", type=str, 
                       help="Distribution configuration file (e.g., data_distribution.yaml)")
    parser.add_argument("--exp-dir", type=str,
                       help="Experiment directory (data will be saved in exp-dir/data)")
    
    # Legacy arguments for backward compatibility
    parser.add_argument("--data", type=str, help="Legacy: Data config (maps to spectra-config)")
    parser.add_argument("--distribution", type=str, help="Legacy: Distribution config")
    parser.add_argument("--training", type=str, help="Legacy: Training config (ignored)")
    parser.add_argument("--output-dir", type=str, help="Legacy: Output directory")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.data and not args.spectra_config:
        args.spectra_config = args.data
    if args.distribution and not args.distribution_config:
        args.distribution_config = args.distribution
    
    # Validate required arguments
    if not args.spectra_config or not args.distribution_config:
        parser.error("Either --spectra-config and --distribution-config OR --data and --distribution are required")
    
    # Handle output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.exp_dir:
        output_dir = os.path.join(args.exp_dir, 'data')
    else:
        parser.error("Either --exp-dir or --output-dir must be provided")
    
    # Load configurations
    with open(args.spectra_config, 'r') as f:
        spectra_config_dict = yaml.safe_load(f)
    
    with open(args.distribution_config, 'r') as f:
        distribution_config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating data:")
    print(f"  Spectral config: {args.spectra_config}")
    print(f"  Distribution config: {args.distribution_config}")
    print(f"  Output directory: {output_dir}")
    
    # Create spectral configuration
    spectral_config = load_spectral_config(spectra_config_dict)
    
    # Check if using new splits format
    if 'splits' in distribution_config:
        print("\nUsing distribution-based data generation")
        
        # Generate data for all splits
        all_data = generate_split_data(spectral_config, distribution_config['splits'], seed=args.seed)
        
        # Save data for each split
        data_dict = {}
        for split_name, split_data in all_data.items():
            data_dict[f"{split_name}_spectra"] = split_data['spectra']
            data_dict[f"{split_name}_abundances"] = split_data['abundances']
            print(f"  Generated {split_name}: {split_data['spectra'].shape[0]} samples")
        
        save_data(data_dict, output_dir)
        
        # Save metadata
        metadata = {
            "spectra_config": args.spectra_config,
            "distribution_config": args.distribution_config,
            "seed": args.seed,
            "splits": {name: {"n_samples": split_data['spectra'].shape[0]} 
                      for name, split_data in all_data.items()},
            "spectral_params": {
                "n_bins": spectral_config.n_bins,
                "wavelength_min": spectral_config.wavelength_min,
                "wavelength_max": spectral_config.wavelength_max,
                "n_components": len(spectral_config.lines)
            }
        }
        
    else:
        # Legacy format support
        print("\nUsing legacy combination-based data generation")
        from src.spectra_generator import generate_train_test_data
        
        # Generate training data
        train_spectra, train_abundances, _, _ = generate_train_test_data(
            spectral_config,
            distribution_config["train_combinations"],
            distribution_config["train_combinations"],
            n_train=distribution_config["n_train"],
            n_test=1
        )
        
        # Generate test data
        _, _, test_spectra, test_abundances = generate_train_test_data(
            spectral_config,
            distribution_config["train_combinations"],
            distribution_config["test_combinations"],
            n_train=1,
            n_test=distribution_config["n_test"]
        )
        
        # Save data
        data_dict = {
            "train_spectra": train_spectra,
            "train_abundances": train_abundances,
            "test_spectra": test_spectra,
            "test_abundances": test_abundances
        }
        
        save_data(data_dict, output_dir)
        
        # Save metadata
        metadata = {
            "spectra_config": args.spectra_config,
            "distribution_config": args.distribution_config,
            "seed": args.seed,
            "n_train": distribution_config["n_train"],
            "n_test": distribution_config["n_test"],
            "train_combinations": distribution_config["train_combinations"],
            "test_combinations": distribution_config["test_combinations"],
            "spectral_params": {
                "n_bins": spectral_config.n_bins,
                "wavelength_min": spectral_config.wavelength_min,
                "wavelength_max": spectral_config.wavelength_max,
                "n_components": len(spectral_config.lines)
            }
        }
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy config files to exp-dir/configs (if exp-dir was provided)
    if args.exp_dir:
        import shutil
        configs_dir = os.path.join(args.exp_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        shutil.copy(args.spectra_config, os.path.join(configs_dir, "spectra_config.yaml"))
        shutil.copy(args.distribution_config, os.path.join(configs_dir, "distribution_config.yaml"))
        print(f"Config files copied to: {configs_dir}")
    
    print(f"\nData generation completed successfully!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()