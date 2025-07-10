"""
Evaluation utilities and metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, Tuple

from .models import create_model


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics."""
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # R-squared
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {"mse": mse, "mae": mae, "r2": r2.item()}


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate model on a dataset."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for spectra, abundances in data_loader:
            spectra = spectra.to(device)
            abundances = abundances.to(device)
            
            predictions = model(spectra)
            all_predictions.append(predictions.cpu())
            all_targets.append(abundances.cpu())
    
    return torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0)


def plot_predictions_vs_targets(predictions: torch.Tensor, targets: torch.Tensor, 
                               save_path: str, title: str):
    """Plot predictions vs targets."""
    n_components = predictions.shape[1]
    n_cols = min(4, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(n_components):
        ax = axes[i] if len(axes) > 1 else axes
        
        pred_i = predictions[:, i].numpy()
        target_i = targets[:, i].numpy()
        
        ax.scatter(target_i, pred_i, alpha=0.6, s=10)
        
        # Perfect prediction line
        min_val = min(pred_i.min(), target_i.min())
        max_val = max(pred_i.max(), target_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect')
        
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title(f'Component {i+1}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_components, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_trained_model(config: dict, model_path: str, device: torch.device) -> nn.Module:
    """Load a trained model."""
    model = create_model(
        config["model_type"],
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_dims=config["hidden_dims"],
        dropout_rate=config["dropout_rate"],
        activation=config["activation"]
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def run_evaluation(config: dict, model: nn.Module, val_loader: DataLoader, 
                  test_loader: DataLoader, device: torch.device, save_dir: str) -> Dict:
    """Run complete evaluation."""
    # Evaluate on validation set (in-distribution)
    val_preds, val_targets = evaluate_model(model, val_loader, device)
    val_metrics = compute_metrics(val_preds, val_targets)
    
    # Evaluate on test set (out-of-distribution)
    test_preds, test_targets = evaluate_model(model, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_targets)
    
    # Create visualizations
    plot_predictions_vs_targets(
        val_preds, val_targets, 
        os.path.join(save_dir, "val_predictions.png"),
        "Validation (In-Distribution) - Predictions vs Targets"
    )
    
    plot_predictions_vs_targets(
        test_preds, test_targets, 
        os.path.join(save_dir, "test_predictions.png"),
        "Test (Out-of-Distribution) - Predictions vs Targets"
    )
    
    # Compute generalization gap
    gen_gap = {
        'mae_gap': test_metrics['mae'] - val_metrics['mae'],
        'mse_gap': test_metrics['mse'] - val_metrics['mse'],
        'r2_gap': val_metrics['r2'] - test_metrics['r2'],
    }
    
    # Compile results
    results = {
        'in_distribution': val_metrics,
        'out_of_distribution': test_metrics,
        'generalization_gap': gen_gap
    }
    
    # Save results
    with open(os.path.join(save_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results