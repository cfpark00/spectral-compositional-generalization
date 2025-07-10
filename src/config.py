"""
Simple configuration loading for YAML files.
"""

import yaml
import torch
import numpy as np
import random
import os
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv


def load_env() -> Dict[str, str]:
    """Load environment variables from .env file."""
    # Find and load .env file, starting from current directory up to project root
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
    
    # Return the loaded environment variables as a dict
    return dict(os.environ)


def load_multiple_configs(data_path: str, distribution_path: str, training_path: str) -> dict:
    """Load and merge multiple configuration files."""
    
    # Load all config files
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    with open(distribution_path, 'r') as f:
        distribution_config = yaml.safe_load(f)
    
    with open(training_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Merge all configs (training config is the base)
    config = training_config.copy()
    config.update(data_config)
    
    # Handle new splits structure
    if 'splits' in distribution_config:
        config['splits'] = distribution_config['splits']
        # For backward compatibility, extract train/test info
        if 'train' in distribution_config['splits']:
            config['n_train'] = distribution_config['splits']['train']['n_samples']
        if 'test' in distribution_config['splits']:
            config['n_test'] = distribution_config['splits']['test']['n_samples']
    else:
        # Old format compatibility
        config.update(distribution_config)
    
    # Add input/output dimensions
    config['input_dim'] = config['n_bins']
    config['output_dim'] = config['n_components']
    
    # Set experiment name based on training config
    if 'experiment_name' not in config:
        config['experiment_name'] = 'demo_simple'
    
    # Load environment variables from .env file
    env_vars = load_env()
    
    # Optional: override device from .env if set
    if 'DEVICE' in env_vars:
        config['device'] = env_vars['DEVICE']
    
    return config


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file and merge with .env variables."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables from .env file
    env_vars = load_env()
    
    # Optional: override device from .env if set
    if 'DEVICE' in env_vars:
        config['device'] = env_vars['DEVICE']
    
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dir(config: dict, base_dir: str = "./data") -> str:
    """Create experiment directory and save configuration."""
    results_dir = os.path.join(base_dir, "results")
    experiment_name = config["experiment_name"]
    
    exp_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return exp_dir