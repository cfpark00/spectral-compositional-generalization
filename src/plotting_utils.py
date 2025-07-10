"""
Plotting utilities for spectral compositional generalization.
Contains all visualization functions for physics and data distributions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json


def plot_diverse_samples(spectral_config, generator, output_dir, n_samples=10):
    """Plot diverse samples with their blackbody baselines."""
    fig, ax = plt.subplots(figsize=(14, 8))
    wavelengths = np.linspace(spectral_config.wavelength_min, 
                             spectral_config.wavelength_max, 
                             spectral_config.n_bins)
    
    # Function to generate proper blackbody at given temperature
    def blackbody_spectrum(wavelengths_nm, temperature_k):
        wavelengths_m = wavelengths_nm * 1e-9
        h = 6.626e-34
        c = 3e8
        k_b = 1.381e-23
        numerator = 2 * h * c**2
        denominator = wavelengths_m**5 * (np.exp(h * c / (wavelengths_m * k_b * temperature_k)) - 1)
        intensity = numerator / denominator
        return intensity / np.max(intensity)
    
    # Generate samples
    for i in range(n_samples):
        # Random temperature
        temp_k = generator._sample_from_config(generator.config.temperature_sampling)
        
        # Random components (up to 3)
        n_active = np.random.randint(1, 4)
        active_components = np.random.choice(len(spectral_config.lines), n_active, replace=False)
        
        # Create abundance vector
        abundances = torch.zeros(len(spectral_config.lines))
        for comp_idx in active_components:
            abundances[comp_idx] = generator._sample_from_config(generator.config.abundance_sampling)
        
        # Generate spectrum
        spectrum = generator.generate_spectrum(abundances)
        
        # Plot baseline blackbody
        bb_baseline = blackbody_spectrum(wavelengths, temp_k)
        ax.plot(wavelengths, bb_baseline, '--', color=f'C{i}', alpha=0.5, linewidth=1)
        
        # Plot spectrum
        ax.plot(wavelengths, spectrum.numpy(), color=f'C{i}', alpha=0.8, linewidth=1.5,
                label=f'{int(temp_k)}K, {n_active} comp')
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Diverse Synthetic Spectra with Blackbody Baselines (dashed)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'diverse_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_spectral_lines(spectral_config, output_dir):
    """Plot the spectral line configurations."""
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    
    # Use higher resolution for plotting to properly show narrow lines
    wavelengths_highres = np.linspace(spectral_config.wavelength_min, 
                                     spectral_config.wavelength_max, 
                                     4096)
    
    for i, line in enumerate(spectral_config.lines):
        # Generate normalized line profile (peak = 1)
        profile = np.exp(-0.5 * ((wavelengths_highres - line.center) / line.width) ** 2)
        ax1.plot(wavelengths_highres, profile, label=f'{line.label} (w={line.width:.1f})', alpha=0.7)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Spectral Line Profiles (All 20 Components)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax1.grid(True, alpha=0.3)
    
    # Plot line centers and strengths
    centers = [line.center for line in spectral_config.lines]
    strengths = [line.strength for line in spectral_config.lines]
    labels = [line.label for line in spectral_config.lines]
    
    colors = ['red' if s < 0 else 'blue' for s in strengths]
    
    # Add vertical lines from 0 to each point
    for center, strength, color in zip(centers, strengths, colors):
        ax2.plot([center, center], [0, strength], color=color, alpha=0.5, linewidth=1.5)
    
    # Plot scatter points on top
    ax2.scatter(centers, strengths, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add labels for all 20 points
    for i in range(len(centers)):
        ax2.annotate(labels[i], (centers[i], strengths[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Strength')
    ax2.set_title('Spectral Line Centers and Strengths')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, right=0.85)
    fig.savefig(os.path.join(output_dir, 'spectral_lines.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_train_samples(data_dict, output_dir, analysis_config_path=None):
    """Plot training samples showing individual components."""
    train_data = data_dict.get('train', None)
    if train_data is None:
        print("No training data found")
        return
        
    # Load component indices from analysis config
    if analysis_config_path and os.path.exists(analysis_config_path):
        import yaml
        with open(analysis_config_path, 'r') as f:
            config = yaml.safe_load(f)
        comp1_idx = config['plot_config']['x_component'] - 1  # Convert to 0-indexed
        comp2_idx = config['plot_config']['y_component'] - 1  # Convert to 0-indexed
    else:
        # Fallback to default
        comp1_idx, comp2_idx = 0, 1
        
    train_spectra = train_data['spectra']
    train_abundances = train_data['abundances']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    wavelengths = np.linspace(300, 1200, train_spectra.shape[1])
    
    # Find samples with no components, comp1 only, and comp2 only
    no_comp_mask = (train_abundances.sum(dim=1) == 0)
    comp1_only_mask = (train_abundances[:, comp1_idx] > 0) & (train_abundances[:, comp2_idx] == 0)
    comp2_only_mask = (train_abundances[:, comp1_idx] == 0) & (train_abundances[:, comp2_idx] > 0)
    
    # Plot 2 samples of each type
    # No components (gray)
    no_comp_indices = torch.where(no_comp_mask)[0]
    if len(no_comp_indices) >= 2:
        for i in range(2):
            idx = no_comp_indices[i]
            ax.plot(wavelengths, train_spectra[idx].numpy(), color='gray', alpha=0.8, linewidth=1.5,
                    label='No components' if i == 0 else '')
    
    # Comp1 only (blue)
    comp1_indices = torch.where(comp1_only_mask)[0]
    if len(comp1_indices) >= 2:
        for i in range(2):
            idx = comp1_indices[i]
            abundance = train_abundances[idx, comp1_idx].item()
            ax.plot(wavelengths, train_spectra[idx].numpy(), color=f'C{i}', alpha=0.8, linewidth=1.5,
                    label=f'comp{comp1_idx+1} only (a={abundance:.2f})')
    
    # Comp2 only (green)
    comp2_indices = torch.where(comp2_only_mask)[0]
    if len(comp2_indices) >= 2:
        for i in range(2):
            idx = comp2_indices[i]
            abundance = train_abundances[idx, comp2_idx].item()
            ax.plot(wavelengths, train_spectra[idx].numpy(), color=f'C{i+2}', alpha=0.8, linewidth=1.5,
                    label=f'comp{comp2_idx+1} only (a={abundance:.2f})')
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Training Distribution (Individual Components Only)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'train_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_test_samples(data_dict, output_dir, analysis_config_path=None):
    """Plot test samples showing combined components."""
    test_data = data_dict.get('test', None)
    if test_data is None:
        print("No test data found")
        return
        
    # Load component indices from analysis config
    if analysis_config_path and os.path.exists(analysis_config_path):
        import yaml
        with open(analysis_config_path, 'r') as f:
            config = yaml.safe_load(f)
        comp1_idx = config['plot_config']['x_component'] - 1  # Convert to 0-indexed
        comp2_idx = config['plot_config']['y_component'] - 1  # Convert to 0-indexed
    else:
        # Fallback to default
        comp1_idx, comp2_idx = 0, 1
        
    test_spectra = test_data['spectra']
    test_abundances = test_data['abundances']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    wavelengths = np.linspace(300, 1200, test_spectra.shape[1])
    
    # Find samples with both comp1 AND comp2
    both_comp_mask = (test_abundances[:, comp1_idx] > 0) & (test_abundances[:, comp2_idx] > 0)
    both_comp_indices = torch.where(both_comp_mask)[0]
    
    # Plot up to 5 samples
    n_to_plot = min(5, len(both_comp_indices))
    for i in range(n_to_plot):
        idx = both_comp_indices[i]
        a1 = test_abundances[idx, comp1_idx].item()
        a2 = test_abundances[idx, comp2_idx].item()
        ax.plot(wavelengths, test_spectra[idx].numpy(), color=f'C{i}', alpha=0.8, linewidth=1.5,
                label=f'comp{comp1_idx+1}+comp{comp2_idx+1} (a1={a1:.2f}, a2={a2:.2f})')
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Test Distribution (Compositional Gap: comp{comp1_idx+1} AND comp{comp2_idx+1} Together)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'test_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_abundance_scatter(data_dict, output_dir, analysis_config_path=None):
    """Plot scatter of train vs test in comp1-comp2 abundance space."""
    # Load component indices from analysis config
    if analysis_config_path and os.path.exists(analysis_config_path):
        import yaml
        with open(analysis_config_path, 'r') as f:
            config = yaml.safe_load(f)
        comp1_idx = config['plot_config']['x_component'] - 1  # Convert to 0-indexed
        comp2_idx = config['plot_config']['y_component'] - 1  # Convert to 0-indexed
    else:
        # Fallback to default
        comp1_idx, comp2_idx = 0, 1
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each split
    colors = {'train': 'blue', 'test': 'red'}
    markers = {'train': 'o', 'test': 's'}
    
    for split_name, split_data in data_dict.items():
        abundances = split_data['abundances']
        comp1_abundances = abundances[:, comp1_idx].numpy()
        comp2_abundances = abundances[:, comp2_idx].numpy()
        
        # Plot with transparency to show density
        ax.scatter(comp1_abundances, comp2_abundances, 
                  c=colors.get(split_name, 'gray'),
                  marker=markers.get(split_name, 'o'),
                  alpha=0.5, s=30, label=split_name)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    # Labels and title
    ax.set_xlabel(f'Component {comp1_idx+1} Abundance', fontsize=12)
    ax.set_ylabel(f'Component {comp2_idx+1} Abundance', fontsize=12)
    ax.set_title(f'Train vs Test Distribution in Abundance Space (comp{comp1_idx+1} vs comp{comp2_idx+1})', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'abundance_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()


def load_generated_data(data_path):
    """Load the generated data and metadata."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Generated data not found at {data_path}")
    
    # Check if we have the new multi-seed structure
    data_files = os.listdir(data_path)
    seed_dirs = [f for f in data_files if f.startswith('seed-') and os.path.isdir(os.path.join(data_path, f))]
    
    if seed_dirs:
        # New multi-seed structure - use first available seed for plotting
        seed_dir = sorted(seed_dirs)[0]
        actual_data_path = os.path.join(data_path, seed_dir)
        print(f"  Using data from {seed_dir} for visualization")
    else:
        # Old single-seed structure
        actual_data_path = data_path
    
    # Load all data files from the actual data path
    data_files = os.listdir(actual_data_path)
    data_dict = {}
    
    # Load train/test data (supporting both old and new formats)
    if 'train_spectra.pt' in data_files:
        # Old format - import here to avoid circular import
        from .data_utils import load_data
        train_spectra, train_abundances, test_spectra, test_abundances = load_data(actual_data_path)
        data_dict = {
            'train': {'spectra': train_spectra, 'abundances': train_abundances},
            'test': {'spectra': test_spectra, 'abundances': test_abundances}
        }
    else:
        # New format - load all splits
        for file in data_files:
            if file.endswith('_spectra.pt'):
                split_name = file.replace('_spectra.pt', '')
                spectra = torch.load(os.path.join(actual_data_path, file))
                abundances = torch.load(os.path.join(actual_data_path, f'{split_name}_abundances.pt'))
                data_dict[split_name] = {'spectra': spectra, 'abundances': abundances}
    
    # Load metadata
    metadata_path = os.path.join(actual_data_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return data_dict, metadata


def plot_training_curves(train_dir, output_dir):
    """Plot MSE curves for train and test from checkpoint stats."""
    ckpts_dir = os.path.join(train_dir, 'ckpts')
    
    if not os.path.exists(ckpts_dir):
        print("  ⚠️  No checkpoints found - skipping training curves")
        return
    
    # Collect checkpoint data
    checkpoint_data = []
    
    # Get all checkpoint directories
    ckpt_dirs = [d for d in os.listdir(ckpts_dir) if d.startswith('ckpt-')]
    ckpt_dirs.sort(key=lambda x: int(x.split('-')[1]))
    
    for ckpt_dir in ckpt_dirs:
        stats_path = os.path.join(ckpts_dir, ckpt_dir, 'stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                checkpoint_data.append(stats)
    
    if not checkpoint_data:
        print("  ⚠️  No checkpoint stats found - skipping training curves")
        return
    
    # Extract data for plotting
    checkpoints = [d['checkpoint_num'] for d in checkpoint_data]
    steps = [d['step'] for d in checkpoint_data]
    train_losses = [d['train_loss'] for d in checkpoint_data]
    test_losses = [d['test_loss'] for d in checkpoint_data]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot by checkpoint number
    ax1.plot(checkpoints, train_losses, 'b-o', label='Train MSE', alpha=0.8)
    ax1.plot(checkpoints, test_losses, 'r-s', label='Test MSE', alpha=0.8)
    ax1.set_xlabel('Checkpoint Number')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Progress by Checkpoint')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot by training step
    ax2.plot(steps, train_losses, 'b-o', label='Train MSE', alpha=0.8)
    ax2.plot(steps, test_losses, 'r-s', label='Test MSE', alpha=0.8)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Progress by Step')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_analysis(train_dir, data_dict, output_dir):
    """Plot analysis of best checkpoint predictions vs ground truth."""
    ckpts_dir = os.path.join(train_dir, 'ckpts')
    
    if not os.path.exists(ckpts_dir):
        print("  ⚠️  No checkpoints found - skipping prediction analysis")
        return
    
    # Find best checkpoint (lowest test loss)
    best_ckpt = None
    best_loss = float('inf')
    
    ckpt_dirs = [d for d in os.listdir(ckpts_dir) if d.startswith('ckpt-')]
    for ckpt_dir in ckpt_dirs:
        stats_path = os.path.join(ckpts_dir, ckpt_dir, 'stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                if stats['test_loss'] < best_loss:
                    best_loss = stats['test_loss']
                    best_ckpt = ckpt_dir
    
    if best_ckpt is None:
        print("  ⚠️  No valid checkpoints found - skipping prediction analysis")
        return
    
    # Load best predictions
    pred_path = os.path.join(ckpts_dir, best_ckpt, 'test_predictions.pt')
    if not os.path.exists(pred_path):
        print("  ⚠️  No predictions found - skipping prediction analysis")
        return
    
    predictions = torch.load(pred_path)
    ground_truth = data_dict['test']['abundances']
    
    # Create scatter plot for first two components
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Component 1
    ax1.scatter(ground_truth[:, 0], predictions[:, 0], alpha=0.6, s=20)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect prediction')
    ax1.set_xlabel('True Abundance (Component 1)')
    ax1.set_ylabel('Predicted Abundance (Component 1)')
    ax1.set_title(f'Component 1 Predictions (Best Checkpoint: {best_ckpt})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    
    # Component 2
    ax2.scatter(ground_truth[:, 1], predictions[:, 1], alpha=0.6, s=20)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect prediction')
    ax2.set_xlabel('True Abundance (Component 2)')
    ax2.set_ylabel('Predicted Abundance (Component 2)')
    ax2.set_title(f'Component 2 Predictions (Best Checkpoint: {best_ckpt})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_mse_trajectory(analysis_data_path, output_dir):
    """Plot MSE trajectory for all runs from analysis CSV data."""
    import pandas as pd
    
    # Load trajectory data
    df = pd.read_csv(analysis_data_path)
    
    # Get component names from columns
    component_cols = [col for col in df.columns if col.startswith('component_') and col.endswith('_mse')]
    if len(component_cols) < 2:
        print("Warning: Not enough component MSE columns found in analysis data")
        return
    
    x_col = component_cols[0]  # First component MSE
    y_col = component_cols[1]  # Second component MSE
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Collect data for mean calculation
    all_trajectories = []
    
    # Plot each seed separately
    seeds = df['seed'].unique()
    for i, seed in enumerate(seeds):
        seed_data = df[df['seed'] == seed]
        
        if len(seed_data) == 0:
            continue
            
        color = colors[i % len(colors)]
        
        # Plot trajectory
        ax.plot(seed_data[x_col], seed_data[y_col], '-', color=color, linewidth=2, alpha=0.5, label=f'{seed} trajectory')
        
        # Mark start and end points
        ax.scatter(seed_data[x_col].iloc[0], seed_data[y_col].iloc[0], color=color, s=100, marker='o', zorder=5, alpha=0.6)
        ax.scatter(seed_data[x_col].iloc[-1], seed_data[y_col].iloc[-1], color=color, s=100, marker='s', zorder=5, alpha=0.6)
        
        # Add checkpoint markers
        ax.scatter(seed_data[x_col], seed_data[y_col], color=color, s=30, alpha=0.2, zorder=3)
        
        # Store for mean calculation
        all_trajectories.append(seed_data[[x_col, y_col, 'checkpoint_num']].set_index('checkpoint_num'))
    
    # Calculate and plot mean trajectory
    if len(all_trajectories) > 1:
        # Align all trajectories by checkpoint number
        aligned_data = pd.concat(all_trajectories, axis=1, keys=range(len(all_trajectories)))
        
        # Calculate mean for each component
        x_mean = aligned_data.xs(x_col, level=1, axis=1).mean(axis=1)
        y_mean = aligned_data.xs(y_col, level=1, axis=1).mean(axis=1)
        
        # Plot mean trajectory
        ax.plot(x_mean, y_mean, 'k--', linewidth=4, alpha=0.8, label='Mean trajectory', zorder=10)
        ax.scatter(x_mean.iloc[0], y_mean.iloc[0], color='black', s=200, marker='o', zorder=15, alpha=0.8, edgecolors='white', linewidth=2)
        ax.scatter(x_mean.iloc[-1], y_mean.iloc[-1], color='black', s=200, marker='s', zorder=15, alpha=0.8, edgecolors='white', linewidth=2)
    
    # Add legend entries for start/end markers
    ax.scatter([], [], color='gray', s=100, marker='o', label='Start (individual runs)', alpha=0.8)
    ax.scatter([], [], color='gray', s=100, marker='s', label='End (individual runs)', alpha=0.8)
    
    # Extract component numbers from column names
    x_comp_num = x_col.split('_')[1]
    y_comp_num = y_col.split('_')[1]
    
    ax.set_xlabel(f'Component {x_comp_num} MSE')
    ax.set_ylabel(f'Component {y_comp_num} MSE')
    ax.set_title('MSE Trajectory During Training (All Runs)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_trajectory.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MSE trajectory plot saved to: {os.path.join(output_dir, 'mse_trajectory.png')}")


def plot_mse_time_series(analysis_data_path, output_dir):
    """Plot MSE for each component as function of training time."""
    import pandas as pd
    
    # Load trajectory data
    df = pd.read_csv(analysis_data_path)
    
    # Get component MSE columns
    component_cols = [col for col in df.columns if col.startswith('component_') and col.endswith('_mse')]
    if len(component_cols) < 2:
        print("Warning: Not enough component MSE columns found in analysis data")
        return
    
    # Use first two components
    comp1_col = component_cols[0]
    comp2_col = component_cols[1]
    
    # Extract component numbers
    comp1_num = comp1_col.split('_')[1]
    comp2_num = comp2_col.split('_')[1]
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for components
    comp1_color = 'blue'
    comp2_color = 'red'
    
    # Plot each seed
    seeds = df['seed'].unique()
    
    # Collect data for ensemble average
    comp1_trajectories = []
    comp2_trajectories = []
    
    # Flag to add legend labels only once
    added_individual_labels = False
    
    for i, seed in enumerate(seeds):
        seed_data = df[df['seed'] == seed]
        
        if len(seed_data) == 0:
            continue
        
        # Use step as x-axis (training time)
        steps = seed_data['step'].values
        
        # Plot individual runs with transparency (label only first one)
        if not added_individual_labels:
            ax.plot(steps, seed_data[comp1_col], '-', color=comp1_color, alpha=0.3, linewidth=1, 
                    label=f'Component {comp1_num} (individual)')
            ax.plot(steps, seed_data[comp2_col], '-', color=comp2_color, alpha=0.3, linewidth=1,
                    label=f'Component {comp2_num} (individual)')
            added_individual_labels = True
        else:
            ax.plot(steps, seed_data[comp1_col], '-', color=comp1_color, alpha=0.3, linewidth=1)
            ax.plot(steps, seed_data[comp2_col], '-', color=comp2_color, alpha=0.3, linewidth=1)
        
        # Store for ensemble average
        comp1_trajectories.append(seed_data[['step', comp1_col]].set_index('step')[comp1_col])
        comp2_trajectories.append(seed_data[['step', comp2_col]].set_index('step')[comp2_col])
    
    # Calculate and plot ensemble averages
    if len(comp1_trajectories) > 1:
        # Align all trajectories by step
        comp1_aligned = pd.concat(comp1_trajectories, axis=1)
        comp2_aligned = pd.concat(comp2_trajectories, axis=1)
        
        # Calculate means
        comp1_mean = comp1_aligned.mean(axis=1)
        comp2_mean = comp2_aligned.mean(axis=1)
        
        # Plot thick ensemble average lines
        ax.plot(comp1_mean.index, comp1_mean.values, '-', color=comp1_color, linewidth=3, 
                label=f'Component {comp1_num} (ensemble avg)', zorder=10)
        ax.plot(comp2_mean.index, comp2_mean.values, '-', color=comp2_color, linewidth=3, 
                label=f'Component {comp2_num} (ensemble avg)', zorder=10)
    
    # Formatting
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Component MSE vs Training Steps', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MSE time series plot saved to: {os.path.join(output_dir, 'mse_time_series.png')}")


def plot_test_abundance_by_mse(analysis_data_path, output_dir, exp_dir, analysis_config_path=None):
    """Plot test abundances colored by ensemble-averaged log MSE."""
    import pandas as pd
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    # Load analysis data to get final MSE values
    df = pd.read_csv(analysis_data_path)
    
    # Load component indices from analysis config
    if analysis_config_path and os.path.exists(analysis_config_path):
        import yaml
        with open(analysis_config_path, 'r') as f:
            config = yaml.safe_load(f)
        comp1_idx = config['plot_config']['x_component'] - 1  # Convert to 0-indexed
        comp2_idx = config['plot_config']['y_component'] - 1  # Convert to 0-indexed
    else:
        # Fallback to default
        comp1_idx, comp2_idx = 0, 1
    
    # Get final checkpoint data for each seed
    seeds = df['seed'].unique()
    final_predictions = {}
    
    for seed in seeds:
        seed_data = df[df['seed'] == seed]
        # Get the last checkpoint
        final_row = seed_data.iloc[-1]
        
        # Find the checkpoint directory
        runs_dir = os.path.join(exp_dir, 'runs', f'seed-{seed}', 'train', 'ckpts')
        ckpt_num = int(final_row['checkpoint_num'])
        ckpt_dir = os.path.join(runs_dir, f'ckpt-{ckpt_num}')
        
        # Load test predictions
        pred_path = os.path.join(ckpt_dir, 'test_predictions.pt')
        if os.path.exists(pred_path):
            final_predictions[seed] = torch.load(pred_path)
    
    if not final_predictions:
        print("Warning: No test predictions found")
        return
    
    # Load test abundances (ground truth) - use any data seed
    data_dir = os.path.join(exp_dir, 'data')
    seed_dirs = [d for d in os.listdir(data_dir) if d.startswith('seed-') and os.path.isdir(os.path.join(data_dir, d))]
    if not seed_dirs:
        print("Warning: No data directories found")
        return
    
    test_abundances_path = os.path.join(data_dir, seed_dirs[0], 'test_abundances.pt')
    test_abundances = torch.load(test_abundances_path)
    
    # Calculate ensemble-averaged MSE for each test sample
    n_samples = test_abundances.shape[0]
    ensemble_mse = np.zeros(n_samples)
    
    for i in range(n_samples):
        sample_mses = []
        for seed, predictions in final_predictions.items():
            # Calculate MSE for this sample across all components
            mse = ((predictions[i] - test_abundances[i]) ** 2).mean().item()
            sample_mses.append(mse)
        # Take ensemble average
        ensemble_mse[i] = np.mean(sample_mses)
    
    # Calculate log MSE (with small epsilon to avoid log(0))
    log_ensemble_mse = np.log10(ensemble_mse + 1e-10)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get abundances for the two components
    comp1_abundances = test_abundances[:, comp1_idx].numpy()
    comp2_abundances = test_abundances[:, comp2_idx].numpy()
    
    # Create colormap
    vmin, vmax = np.percentile(log_ensemble_mse, [5, 95])  # Use percentiles for better color range
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='viridis')
    
    # Create scatter plot
    scatter = ax.scatter(comp1_abundances, comp2_abundances, 
                        c=log_ensemble_mse, cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
                        norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, label='log₁₀(Ensemble Avg MSE)')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    # Labels and title
    ax.set_xlabel(f'Component {comp1_idx+1} Abundance', fontsize=12)
    ax.set_ylabel(f'Component {comp2_idx+1} Abundance', fontsize=12)
    ax.set_title('Test Set Abundances Colored by Ensemble-Averaged MSE', fontsize=14)
    
    # Add text showing number of runs in ensemble
    ax.text(0.02, 0.98, f'Ensemble: {len(final_predictions)} runs', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_abundance_mse.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test abundance MSE plot saved to: {os.path.join(output_dir, 'test_abundance_mse.png')}")