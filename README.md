# Spectral Compositional Generalization Tutorial

A tutorial demonstrating compositional generalization using synthetic spectroscopy data. Train neural networks on individual spectral components and test on unseen combinations.

## Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib pyyaml tqdm

# Run the complete pipeline
python run.py meta_config.yaml
```

This creates an experiment in `exp/` with:
- Generated synthetic spectral data
- Trained models across multiple seeds
- Comprehensive analysis and visualizations

## Core Concept

**Compositional Gap**: Train on components individually, test on combinations
- **Training**: Model sees comp10 alone OR comp4 alone  
- **Testing**: Model must predict comp10 AND comp4 together
- **Result**: Tests if networks can generalize compositionally

## Configuration

Edit `meta_config.yaml` to control:
- `data_seed`: Random seed(s) for data generation
- `training_seed`: Random seed(s) for model training (can use multiple with single data seed)
- Pipeline flags: `generate_data`, `run_training`, `run_analysis`

Key configs in `configs/`:
- `synthetic_spectra.yaml`: Physics parameters (20 spectral lines)
- `data_distribution.yaml`: Train/test split definition
- `training.yaml`: Model architecture and training
- `analysis.yaml`: Which components to analyze (x,y axes)

## Outputs

### Data Structure
```
exp/
├── data/seed-*/          # Generated datasets
├── runs/seed-*/          # Training runs  
├── analysis/             # Combined analysis
└── plots/                # All visualizations
```

### Generated Plots
- **Physics**: `spectral_lines.png`, `diverse_samples.png`
- **Data**: `train_samples.png`, `test_samples.png`, `abundance_scatter.png`
- **Training**: `training_curves.png`, `prediction_analysis.png`
- **Analysis**: `mse_trajectory.png`, `mse_time_series.png`, `test_abundance_mse.png`

## Scripts

- `scripts/generate_data.py`: Create synthetic spectra
- `scripts/train.py`: Train neural network
- `scripts/analysis.py`: Analyze checkpoints and compute MSE trajectories

## Extending

1. **New components**: Edit spectral lines in `configs/synthetic_spectra.yaml`
2. **Different gaps**: Modify train/test splits in `configs/data_distribution.yaml`
3. **Model architecture**: Change network in `configs/training.yaml`

## License

MIT