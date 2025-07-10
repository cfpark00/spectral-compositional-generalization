"""
Analysis script that fetches training checkpoints and test results,
and saves MSE trajectory data as CSV.
"""

import argparse
import os
import sys
import json
import yaml
import torch
import numpy as np
from glob import glob
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import compute_metrics


def load_analysis_config(config_path: str) -> Dict:
    """Load analysis configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def collect_checkpoint_data(run_dir: str) -> List[Dict]:
    """Collect data from all training checkpoints for a single run."""
    print(f"  Collecting from: {run_dir}")
    
    # Find checkpoints directory
    ckpts_dir = os.path.join(run_dir, "train", "ckpts")
    if not os.path.exists(ckpts_dir):
        # Try old structure
        ckpts_dir = os.path.join(run_dir, "ckpts")
        if not os.path.exists(ckpts_dir):
            raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    
    # Find test targets - look in the experiment's data directory
    # Go up from runs/seed-X to exp/, then into data/
    base_exp_dir = os.path.dirname(os.path.dirname(run_dir))  # exp/
    data_base_dir = os.path.join(base_exp_dir, "data")
    
    # Find any data seed directory (they all have same test targets)
    seed_dirs = glob(os.path.join(data_base_dir, "seed-*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No data seed directories found in {data_base_dir}")
    
    data_dir = seed_dirs[0]  # Use first available
    test_targets_file = os.path.join(data_dir, "test_abundances.pt")
    
    if not os.path.exists(test_targets_file):
        raise FileNotFoundError(f"Test targets not found: {test_targets_file}")
    
    print(f"  Using test data from: {os.path.basename(data_dir)}")
    test_targets = torch.load(test_targets_file, map_location='cpu')
    
    checkpoint_data = []
    
    # Find all checkpoint directories
    ckpt_dirs = sorted(glob(os.path.join(ckpts_dir, "ckpt-*")), 
                      key=lambda x: int(x.split('-')[-1]))
    
    for ckpt_dir in ckpt_dirs:
        stats_file = os.path.join(ckpt_dir, "stats.json")
        predictions_file = os.path.join(ckpt_dir, "test_predictions.csv")
        
        if not os.path.exists(stats_file) or not os.path.exists(predictions_file):
            print(f"    Warning: Missing files in {os.path.basename(ckpt_dir)}, skipping")
            continue
            
        # Load stats
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        # Load predictions from CSV
        import pandas as pd
        predictions_df = pd.read_csv(predictions_file)
        predictions = torch.tensor(predictions_df.values, dtype=torch.float32)
        
        checkpoint_data.append({
            'checkpoint_num': stats['checkpoint_num'],
            'step': stats['step'],
            'epoch': stats['epoch'],
            'train_loss': stats['train_loss'],
            'test_loss': stats['test_loss'],
            'predictions': predictions,
            'targets': test_targets
        })
    
    print(f"  Found {len(checkpoint_data)} valid checkpoints")
    return checkpoint_data


def collect_all_runs_data(runs_dir: str) -> Dict[str, List[Dict]]:
    """Collect checkpoint data from all runs."""
    all_runs_data = {}
    
    # Find all seed directories
    seed_dirs = sorted(glob(os.path.join(runs_dir, "seed-*")))
    
    if not seed_dirs:
        print(f"No seed directories found in {runs_dir}")
        return {}
    
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        print(f"Collecting data for {seed_name}...")
        
        try:
            checkpoint_data = collect_checkpoint_data(seed_dir)
            if checkpoint_data:
                all_runs_data[seed_name] = checkpoint_data
            else:
                print(f"  No valid checkpoints found for {seed_name}")
        except Exception as e:
            print(f"  Warning: Could not collect data for {seed_name}: {e}")
    
    return all_runs_data


def compute_component_mse(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """Compute MSE for each component."""
    # predictions and targets are [n_samples, n_components]
    mse_per_component = torch.mean((predictions - targets) ** 2, dim=0)
    return mse_per_component.numpy()


def save_trajectory_data(all_runs_data: Dict[str, List[Dict]], config: Dict, save_path: str):
    """Save trajectory data as CSV for all runs."""
    x_comp = config['plot_config']['x_component'] - 1
    y_comp = config['plot_config']['y_component'] - 1
    
    # Collect data from all runs
    data_rows = []
    for seed_name, checkpoint_data in all_runs_data.items():
        for ckpt in checkpoint_data:
            component_mse = compute_component_mse(ckpt['predictions'], ckpt['targets'])
            
            data_rows.append({
                'seed': seed_name,
                'checkpoint_num': ckpt['checkpoint_num'],
                'step': ckpt['step'],
                'epoch': ckpt['epoch'],
                'train_loss': ckpt['train_loss'],
                'test_loss': ckpt['test_loss'],
                f'component_{x_comp + 1}_mse': component_mse[x_comp],
                f'component_{y_comp + 1}_mse': component_mse[y_comp]
            })
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(data_rows)
    df.to_csv(save_path, index=False)
    print(f"Trajectory data saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training checkpoints and create MSE trajectory data")
    parser.add_argument("--exp-dir", type=str, help="Single experiment directory (legacy)")
    parser.add_argument("--runs-dir", type=str, help="Runs directory containing seed-* subdirectories")
    parser.add_argument("--config", type=str, default="configs/analysis.yaml", 
                       help="Analysis configuration file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()
    
    # Load analysis configuration
    config = load_analysis_config(args.config)
    
    # Handle legacy single experiment or new multi-run
    if args.exp_dir:
        # Legacy single experiment
        print(f"Analyzing single experiment: {args.exp_dir}")
        checkpoint_data = collect_checkpoint_data(args.exp_dir)
        all_runs_data = {"single-run": checkpoint_data}
        
        output_dir = args.output_dir or os.path.join(args.exp_dir, "analysis")
    elif args.runs_dir:
        # New multi-run analysis
        print(f"Analyzing multiple runs: {args.runs_dir}")
        all_runs_data = collect_all_runs_data(args.runs_dir)
        
        output_dir = args.output_dir or os.path.join(os.path.dirname(args.runs_dir), "analysis")
    else:
        parser.error("Either --exp-dir or --runs-dir must be provided")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if not all_runs_data:
        print("No checkpoint data found!")
        return
    
    # Save trajectory data
    print("Saving trajectory data...")
    trajectory_data_path = os.path.join(output_dir, "trajectory_data.csv")
    save_trajectory_data(all_runs_data, config, trajectory_data_path)
    
    print("\nAnalysis completed!")
    print(f"Results saved in: {output_dir}")
    print("Files created:")
    print(f"  - trajectory_data.csv: Raw trajectory data ({len(all_runs_data)} runs)")


if __name__ == "__main__":
    main()