#!/usr/bin/env python3
"""
Comprehensive example runner for spectral compositional generalization.
Creates a complete experiment directory with configs, data, and visualizations.
"""

import os
import sys
import argparse
import yaml
import subprocess
import shutil
from pathlib import Path

# Add project root to path (run.py is now in root)
sys.path.append(os.path.dirname(__file__))


def load_meta_config(config_path):
    """Load meta configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_meta_config(config):
    """Validate that all required fields are present in meta config."""
    required_fields = ['exp_dir', 'spectra_config_path', 'distribution_config_path']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required fields in meta config: {missing_fields}")
    
    # Check that config files exist
    if not os.path.exists(config['spectra_config_path']):
        raise FileNotFoundError(f"Spectra config not found: {config['spectra_config_path']}")
    
    if not os.path.exists(config['distribution_config_path']):
        raise FileNotFoundError(f"Distribution config not found: {config['distribution_config_path']}")
    
    # Handle seed fields - convert to lists if needed
    if 'data_seed' in config:
        if not isinstance(config['data_seed'], list):
            config['data_seed'] = [config['data_seed']]
    else:
        config['data_seed'] = [42]  # default
    
    # Check training config if training is enabled
    if config.get('run_training', True):
        if 'training_config_path' not in config:
            raise ValueError("training_config_path required when run_training is true")
        if not os.path.exists(config['training_config_path']):
            raise FileNotFoundError(f"Training config not found: {config['training_config_path']}")
        
        # Handle training seeds
        if 'training_seed' in config:
            if not isinstance(config['training_seed'], list):
                config['training_seed'] = [config['training_seed']]
        else:
            config['training_seed'] = [123]  # default
        
        # Validate seed configuration: single data seed can be used with multiple training seeds
        if len(config['data_seed']) > 1 and len(config['data_seed']) != len(config['training_seed']):
            raise ValueError(f"When using multiple data seeds ({len(config['data_seed'])}), number of training seeds ({len(config['training_seed'])}) must match")
    
    # Check analysis config if analysis is enabled
    if config.get('run_analysis', True):
        if 'analysis_config_path' not in config:
            raise ValueError("analysis_config_path required when run_analysis is true")
        if not os.path.exists(config['analysis_config_path']):
            raise FileNotFoundError(f"Analysis config not found: {config['analysis_config_path']}")
    
    # Check logical dependencies between flags
    if config.get('run_training', True) and not config.get('generate_data', True):
        raise ValueError("run_training=true requires generate_data=true (training needs data)")
    
    if config.get('run_analysis', True) and not config.get('run_training', True):
        raise ValueError("run_analysis=true requires run_training=true (analysis needs trained model)")
    
    return True


def setup_experiment_directory(exp_dir):
    """Create experiment directory structure."""
    # Create directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'configs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)


def copy_configs(meta_config, exp_dir):
    """Copy configuration files to experiment directory."""
    config_dir = os.path.join(exp_dir, 'configs')
    
    # Copy spectra config
    shutil.copy2(
        meta_config['spectra_config_path'], 
        os.path.join(config_dir, 'spectra_config.yaml')
    )
    
    # Copy distribution config
    shutil.copy2(
        meta_config['distribution_config_path'],
        os.path.join(config_dir, 'distribution_config.yaml')
    )
    
    # Copy training config if it exists
    if 'training_config_path' in meta_config:
        shutil.copy2(
            meta_config['training_config_path'],
            os.path.join(config_dir, 'training.yaml')
        )
    
    # Copy analysis config if it exists
    if 'analysis_config_path' in meta_config:
        shutil.copy2(
            meta_config['analysis_config_path'],
            os.path.join(config_dir, 'analysis.yaml')
        )
    
    # Save meta config
    meta_config_path = os.path.join(config_dir, 'meta_config.yaml')
    with open(meta_config_path, 'w') as f:
        yaml.dump(meta_config, f, default_flow_style=False)


def generate_data(meta_config, exp_dir, project_root):
    """Generate synthetic data using the generate_data.py script."""
    data_seeds = meta_config['data_seed']
    
    for i, data_seed in enumerate(data_seeds):
        print(f"\n  Generating data with seed {data_seed} ({i+1}/{len(data_seeds)})...")
        
        # Create seed-specific data directory
        seed_data_dir = os.path.join(exp_dir, 'data', f'seed-{data_seed}')
        
        cmd = [
            sys.executable,
            os.path.join(project_root, 'scripts', 'generate_data.py'),
            '--spectra-config', meta_config['spectra_config_path'],
            '--distribution-config', meta_config['distribution_config_path'],
            '--output-dir', seed_data_dir,
            '--seed', str(data_seed)
        ]
        
        print("    Running command:")
        print("    " + " ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error generating data:")
            print(result.stderr)
            raise RuntimeError(f"Data generation failed for seed {data_seed}")
        
        print(f"    ✓ Data generated for seed {data_seed}")


def run_training(meta_config, exp_dir, project_root):
    """Run model training using the train.py script."""
    data_seeds = meta_config['data_seed']
    training_seeds = meta_config['training_seed']
    
    # Handle single data seed with multiple training seeds
    if len(data_seeds) == 1:
        # Use the same data seed for all training runs
        data_seed = data_seeds[0]
        for i, training_seed in enumerate(training_seeds):
            print(f"\n  Training with data_seed={data_seed}, training_seed={training_seed} ({i+1}/{len(training_seeds)})...")
            
            # Create seed-specific training directory
            seed_train_dir = os.path.join(exp_dir, 'runs', f'seed-{training_seed}')
            seed_data_dir = os.path.join(exp_dir, 'data', f'seed-{data_seed}')
            
            cmd = [
                sys.executable,
                os.path.join(project_root, 'scripts', 'train.py'),
                '--data', meta_config['spectra_config_path'],
                '--distribution', meta_config['distribution_config_path'],
                '--training', meta_config['training_config_path'],
                '--data-dir', seed_data_dir,
                '--exp-dir', seed_train_dir,
                '--training-seed', str(training_seed)
            ]
            
            print("    Running command:")
            print("    " + " ".join(cmd))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Error during training:")
                print(result.stderr)
                raise RuntimeError(f"Training failed for seed {training_seed}")
            
            print(f"    ✓ Training completed for seed {training_seed}")
    else:
        # Multiple data seeds with matching training seeds
        for i, (data_seed, training_seed) in enumerate(zip(data_seeds, training_seeds)):
            print(f"\n  Training with data_seed={data_seed}, training_seed={training_seed} ({i+1}/{len(training_seeds)})...")
            
            # Create seed-specific training directory
            seed_train_dir = os.path.join(exp_dir, 'runs', f'seed-{training_seed}')
            seed_data_dir = os.path.join(exp_dir, 'data', f'seed-{data_seed}')
            
            cmd = [
                sys.executable,
                os.path.join(project_root, 'scripts', 'train.py'),
                '--data', meta_config['spectra_config_path'],
                '--distribution', meta_config['distribution_config_path'],
                '--training', meta_config['training_config_path'],
                '--data-dir', seed_data_dir,
                '--exp-dir', seed_train_dir,
                '--training-seed', str(training_seed)
            ]
            
            print("    Running command:")
            print("    " + " ".join(cmd))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Error during training:")
                print(result.stderr)
                raise RuntimeError(f"Training failed for seed {training_seed}")
            
            print(f"    ✓ Training completed for seed {training_seed}")


def run_analysis(meta_config, exp_dir, project_root):
    """Run analysis using the analysis.py script."""
    # Create combined analysis directory
    analysis_dir = os.path.join(exp_dir, 'analysis')
    
    # Pass all seed directories to analysis script
    runs_dir = os.path.join(exp_dir, 'runs')
    
    cmd = [
        sys.executable,
        os.path.join(project_root, 'scripts', 'analysis.py'),
        '--runs-dir', runs_dir,
        '--config', meta_config['analysis_config_path'],
        '--output-dir', analysis_dir
    ]
    
    print("    Running command:")
    print("    " + " ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error during analysis:")
        print(result.stderr)
        raise RuntimeError("Analysis failed")
    
    print(result.stdout)


def create_visualizations(meta_config, exp_dir, script_dir):
    """Create all visualizations using plotting utilities."""
    
    # Import plotting utilities
    import yaml
    from src.spectra_generator import load_spectral_config, SpectralGenerator
    from src.plotting_utils import (
        plot_diverse_samples,
        plot_spectral_lines,
        plot_train_samples,
        plot_test_samples,
        plot_abundance_scatter,
        plot_training_curves,
        plot_prediction_analysis,
        plot_mse_trajectory,
        plot_mse_time_series,
        plot_test_abundance_by_mse,
        load_generated_data
    )
    
    # Check what data/results are available
    data_available = meta_config.get('generate_data', True) and os.path.exists(os.path.join(exp_dir, 'data'))
    training_available = meta_config.get('run_training', True) and os.path.exists(os.path.join(exp_dir, 'runs'))
    all_flags_false = not meta_config.get('generate_data', True) and not meta_config.get('run_training', True) and not meta_config.get('run_analysis', True)
    
    if all_flags_false:
        print("Creating Physics Visualizations Only")
        print("===================================")
    else:
        print("Creating Comprehensive Visualizations")
        print("====================================")
    
    # Create plots directory
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Output directory: {plots_dir}")
    
    # Load spectral configuration
    print(f"\nLoading spectral configuration from: {meta_config['spectra_config_path']}")
    with open(meta_config['spectra_config_path'], 'r') as f:
        config_dict = yaml.safe_load(f)
    spectral_config = load_spectral_config(config_dict)
    
    # Always create spectral physics visualizations
    print("\nCreating spectral physics visualizations...")
    generator = SpectralGenerator(spectral_config)
    plot_diverse_samples(spectral_config, generator, plots_dir)
    print("  ✓ Diverse samples plot saved")
    
    plot_spectral_lines(spectral_config, plots_dir)
    print("  ✓ Spectral lines plot saved")
    
    # Skip data distribution plots if all flags are false
    if all_flags_false:
        print("\nSkipping data distribution visualizations (all pipeline flags disabled)")
        print(f"\nPhysics visualizations saved in: {plots_dir}")
        print("Files created:")
        print("  - diverse_samples.png: Random spectra showing the physics")
        print("  - spectral_lines.png: All 20 component configurations")
        return
    
    # Create data distribution visualizations
    data_path = os.path.join(exp_dir, 'data')
    if os.path.exists(data_path):
        print("\nCreating data distribution visualizations...")
        try:
            data_dict, metadata = load_generated_data(data_path)
            
            print(f"Loaded splits:")
            for split_name, split_data in data_dict.items():
                n_samples = split_data['spectra'].shape[0]
                print(f"  {split_name}: {n_samples} samples")
            
            # Get analysis config path if available
            analysis_config_path = meta_config.get('analysis_config_path')
            
            plot_train_samples(data_dict, plots_dir, analysis_config_path)
            print("  ✓ Training samples plot saved")
            
            plot_test_samples(data_dict, plots_dir, analysis_config_path)
            print("  ✓ Test samples plot saved")
            
            plot_abundance_scatter(data_dict, plots_dir, analysis_config_path)
            print("  ✓ Abundance scatter plot saved")
            
        except Exception as e:
            print(f"  ⚠️  Could not create data distribution plots: {e}")
    else:
        print("\n⚠️  No data found - skipping distribution visualizations")
    
    # Create training visualizations if training was run
    runs_path = os.path.join(exp_dir, 'runs')
    if os.path.exists(runs_path):
        print("\nCreating training visualizations...")
        
        # For now, use first run for training curves
        first_run = sorted(os.listdir(runs_path))[0] if os.listdir(runs_path) else None
        if first_run:
            first_run_path = os.path.join(runs_path, first_run)
            plot_training_curves(first_run_path, plots_dir)
            print("  ✓ Training curves plot saved")
            
            if os.path.exists(data_path) and 'data_dict' in locals():
                plot_prediction_analysis(first_run_path, data_dict, plots_dir)
                print("  ✓ Prediction analysis plot saved")
    
    # Create analysis visualizations if analysis data is available
    analysis_path = os.path.join(exp_dir, 'analysis')
    if os.path.exists(analysis_path):
        trajectory_data_path = os.path.join(analysis_path, 'trajectory_data.csv')
        if os.path.exists(trajectory_data_path):
            print("\nCreating analysis visualizations...")
            plot_mse_trajectory(trajectory_data_path, plots_dir)
            print("  ✓ MSE trajectory plot saved")
            plot_mse_time_series(trajectory_data_path, plots_dir)
            print("  ✓ MSE time series plot saved")
            
            # Get analysis config path if available
            analysis_config_path = meta_config.get('analysis_config_path')
            try:
                print(f"  Calling plot_test_abundance_by_mse with:")
                print(f"    trajectory_data_path: {trajectory_data_path}")
                print(f"    plots_dir: {plots_dir}")
                print(f"    exp_dir: {exp_dir}")
                print(f"    analysis_config_path: {analysis_config_path}")
                plot_test_abundance_by_mse(trajectory_data_path, plots_dir, exp_dir, analysis_config_path)
                print("  ✓ Test abundance MSE plot saved")
            except Exception as e:
                import traceback
                print(f"  ⚠️  Could not create test abundance MSE plot: {e}")
                print("  Full traceback:")
                traceback.print_exc()
    
    print(f"\nAll visualizations saved in: {plots_dir}")
    print("Files created:")
    print("  - diverse_samples.png: Random spectra showing the physics")
    print("  - spectral_lines.png: All 20 component configurations")
    if os.path.exists(data_path):
        print("  - train_samples.png: Training distribution samples")
        print("  - test_samples.png: Test distribution samples")
        print("  - abundance_scatter.png: Train vs test in abundance space")
    if os.path.exists(runs_path):
        print("  - training_curves.png: MSE curves for train and test")
        print("  - prediction_analysis.png: Best model predictions vs ground truth")
    if os.path.exists(os.path.join(exp_dir, 'analysis')):
        print("  - mse_trajectory.png: MSE trajectory plot (all runs overlapped)")
        print("  - mse_time_series.png: Component MSE vs training steps")
        print("  - test_abundance_mse.png: Test abundances colored by ensemble MSE")


def main():
    """Main function to run the complete example."""
    parser = argparse.ArgumentParser(
        description="Run spectral compositional generalization example from meta config"
    )
    parser.add_argument(
        "meta_config",
        type=str,
        help="Path to meta configuration YAML file"
    )
    args = parser.parse_args()
    
    print("==============================================")
    print("SPECTRAL COMPOSITIONAL GENERALIZATION EXAMPLE")
    print("==============================================")
    
    # Get directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir  # Now run.py is in root
    
    # Load meta configuration
    print(f"\nLoading meta configuration from: {args.meta_config}")
    meta_config = load_meta_config(args.meta_config)
    
    # Make paths absolute if they're relative (relative to meta config file directory)
    meta_config_dir = os.path.dirname(os.path.abspath(args.meta_config))
    
    if not os.path.isabs(meta_config['exp_dir']):
        meta_config['exp_dir'] = os.path.abspath(os.path.join(meta_config_dir, meta_config['exp_dir']))
    if not os.path.isabs(meta_config['spectra_config_path']):
        meta_config['spectra_config_path'] = os.path.abspath(os.path.join(meta_config_dir, meta_config['spectra_config_path']))
    if not os.path.isabs(meta_config['distribution_config_path']):
        meta_config['distribution_config_path'] = os.path.abspath(os.path.join(meta_config_dir, meta_config['distribution_config_path']))
    if 'training_config_path' in meta_config and not os.path.isabs(meta_config['training_config_path']):
        meta_config['training_config_path'] = os.path.abspath(os.path.join(meta_config_dir, meta_config['training_config_path']))
    if 'analysis_config_path' in meta_config and not os.path.isabs(meta_config['analysis_config_path']):
        meta_config['analysis_config_path'] = os.path.abspath(os.path.join(meta_config_dir, meta_config['analysis_config_path']))
    
    # Validate configuration (after fixing paths)
    validate_meta_config(meta_config)
    
    print("\nConfiguration:")
    print(f"  Experiment directory: {meta_config['exp_dir']}")
    print(f"  Spectra config: {meta_config['spectra_config_path']}")
    print(f"  Distribution config: {meta_config['distribution_config_path']}")
    if 'training_config_path' in meta_config:
        print(f"  Training config: {meta_config['training_config_path']}")
    if 'analysis_config_path' in meta_config:
        print(f"  Analysis config: {meta_config['analysis_config_path']}")
    print(f"  Data seeds: {meta_config['data_seed']}")
    if 'training_seed' in meta_config:
        print(f"  Training seeds: {meta_config['training_seed']}")
    print(f"  Generate data: {meta_config.get('generate_data', True)}")
    print(f"  Run training: {meta_config.get('run_training', True)}")
    print(f"  Run analysis: {meta_config.get('run_analysis', True)}")
    
    # Step 1: Setup experiment directory
    print("\nStep 1: Setting up experiment directory...")
    exp_dir = meta_config['exp_dir']
    if os.path.exists(exp_dir):
        response = input(f"Directory {exp_dir} already exists. Remove and continue? [y/N]: ")
        if response.lower() == 'y':
            shutil.rmtree(exp_dir)
        else:
            print("Exiting...")
            sys.exit(0)
    
    setup_experiment_directory(exp_dir)
    # Create runs directory instead of train
    os.makedirs(os.path.join(exp_dir, 'runs'), exist_ok=True)
    print("  ✓ Experiment directory created")
    
    # Step 2: Copy configs
    print("\nStep 2: Copying configuration files...")
    copy_configs(meta_config, exp_dir)
    print("  ✓ Configs copied to exp/configs/")
    
    step_num = 3
    
    # Step 3: Generate data (if enabled)
    if meta_config.get('generate_data', True):
        print(f"\nStep {step_num}: Generating synthetic data...")
        generate_data(meta_config, exp_dir, project_root)
        print("  ✓ Data generated in exp/data/")
        
        # Plot immediately after data generation
        print(f"\nStep {step_num}a: Creating data visualizations...")
        create_visualizations(meta_config, exp_dir, script_dir)
        step_num += 1
    else:
        print(f"\nStep {step_num}: Skipping data generation (generate_data=false)")
        step_num += 1
    
    # Step 4: Run training (if enabled)
    if meta_config.get('run_training', True):
        print(f"\nStep {step_num}: Running model training...")
        run_training(meta_config, exp_dir, project_root)
        print("  ✓ Training completed in exp/runs/")
        
        # Plot after training
        print(f"\nStep {step_num}a: Updating visualizations with training results...")
        create_visualizations(meta_config, exp_dir, script_dir)
        step_num += 1
    else:
        print(f"\nStep {step_num}: Skipping training (run_training=false)")
        step_num += 1
    
    # Step 5: Run analysis (if enabled)
    if meta_config.get('run_analysis', True):
        print(f"\nStep {step_num}: Running analysis...")
        run_analysis(meta_config, exp_dir, project_root)
        print("  ✓ Analysis completed in exp/analysis/")
        
        # Plot after analysis
        print(f"\nStep {step_num}a: Creating analysis visualizations...")
        create_visualizations(meta_config, exp_dir, script_dir)
        step_num += 1
    else:
        print(f"\nStep {step_num}: Skipping analysis (run_analysis=false)")
        step_num += 1
    
    print("\n==============================================")
    print("EXAMPLE COMPLETED!")
    print("==============================================")
    print("")
    print("Experiment directory structure:")
    print(f"{exp_dir}/")
    print("├── configs/           # Configuration files")
    print("│   ├── synthetic_spectra.yaml")
    print("│   ├── data_distribution.yaml")
    if 'training_config_path' in meta_config:
        print("│   ├── training.yaml")
    print("│   └── meta_config.yaml")
    print("├── data/              # Generated datasets")
    print("│   └── seed-*/        # Data for each seed")
    print("│       ├── train_spectra.pt")
    print("│       ├── train_abundances.pt")
    print("│       ├── test_spectra.pt")
    print("│       ├── test_abundances.pt")
    print("│       └── metadata.json")
    if meta_config.get('run_training', True):
        print("├── runs/              # Training results")
        print("│   └── seed-*/        # Results for each training seed")
        print("│       └── train/     # Training outputs")
        print("│           └── ckpts/ # Model checkpoints")
    if meta_config.get('run_analysis', True):
        print("├── analysis/          # Combined analysis results")
        print("│   └── trajectory_data.csv")
    print("└── plots/             # Visualizations")
    print("    ├── diverse_samples.png      # Random spectra with physics")
    print("    ├── spectral_lines.png       # Component configurations")
    if meta_config.get('generate_data', True):
        print("    ├── train_samples.png        # Training distribution")
        print("    ├── test_samples.png         # Test distribution")
        print("    ├── abundance_scatter.png    # Compositional gap visualization")
    if meta_config.get('run_training', True):
        print("    ├── training_curves.png      # MSE curves during training")
        print("    ├── prediction_analysis.png  # Best model predictions")
    if meta_config.get('run_analysis', True):
        print("    ├── mse_trajectory.png       # MSE trajectory (all runs)")
        print("    ├── mse_time_series.png      # Component MSE vs time")
        print("    └── test_abundance_mse.png   # Test abundances colored by MSE")
    print("")
    print("Key insights:")
    print("  - Training: Individual components only")
    print("  - Testing: Combinations of components")
    print("  - This creates the compositional generalization challenge!")
    print("  - Multiple seeds allow studying training variability")


if __name__ == "__main__":
    main()