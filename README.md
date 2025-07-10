# Spectral Compositional Generalization Tutorial

This repository contains a tutorial on understanding neural network compositional generalization through controlled synthetic experiments. We use a synthetic spectral abundance model to investigate when and how neural networks can generalize to unseen combinations of components.

## Overview

The tutorial demonstrates how to:
- Create controlled synthetic experiments to probe neural network behavior
- Analyze compositional generalization in neural networks
- Design systematic train/test splits that create compositional gaps
- Visualize and interpret generalization capabilities

## Quick Start

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy
- Matplotlib
- PyYAML
- tqdm
- python-dotenv

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd SpectralCompGenTutorial

# Install dependencies
pip install torch numpy matplotlib pyyaml tqdm python-dotenv

# Setup environment configuration
./setup_env.sh
```

### Running the Demo

The fastest way to see compositional generalization in action:

```bash
./demo.sh
```

This will:
1. **Generate synthetic spectra** with specific component combinations
2. **Train a model** on individual components only (comp1 OR comp2)
3. **Evaluate on combinations** (comp1 AND comp2 together)
4. **Demonstrate the compositional gap** in model performance

## Core Concepts

### Synthetic Spectra Model

Our synthetic model generates stellar-like spectra with:
- **Blackbody radiation**: Planck's law continuum (3000-7000K)
- **Spectral lines**: 20 components with diverse widths (3.2-350nm)
- **Multiplicative modulation**: Lines modify the continuum
- **Realistic noise**: Multiplicative noise (1-15% level)

Each spectrum spans 300-1200nm with 1024 wavelength bins.

### The Physics

```
spectrum = blackbody(T) × ∏(1 + strength[i] × abundance[i] × line_profile[i]) × (1 + noise)
```

Where:
- `blackbody(T)`: Planck's law at temperature T
- `strength[i]`: Fixed component property [-1, 1] (absorption/emission)
- `abundance[i]`: Variable concentration [0, 1] to be predicted
- `line_profile[i]`: Gaussian centered at specific wavelength
- `noise`: Multiplicative noise modeling detector/atmospheric effects

### Compositional Generalization Challenge

We test whether models can:
1. **Train** on spectra with individual components (A only, B only)
2. **Generalize** to spectra with combinations (A AND B together)
3. **Compose** learned representations without seeing combinations

## Repository Structure

```
SpectralCompGenTutorial/
├── configs/                      # Configuration files
│   ├── synthetic_spectra.yaml    # Physics parameters & sampling
│   ├── data_distribution.yaml    # Train/test split definitions
│   └── training.yaml             # Model & training parameters
├── src/                          # Source code
│   ├── config.py                 # Config management with .env support
│   ├── data_utils.py             # Data I/O utilities
│   ├── spectra_generator.py      # Physics simulation engine
│   ├── plotting_utils.py         # Visualization utilities
│   ├── training.py               # Training logic
│   ├── evaluation.py             # Evaluation metrics
│   └── models/
│       └── mlp.py                # Neural network architectures
├── scripts/                      # Executable scripts
│   ├── generate_data.py          # Data generation
│   ├── train.py                  # Model training
│   └── evaluate.py               # Model evaluation
├── example/                      # Comprehensive example
│   ├── visualize_all.py          # Visualization script
│   ├── run.sh                    # Example runner
│   └── README.md                 # Example documentation
├── data/                         # Generated data (created at runtime)
│   ├── generated/                # Synthetic datasets
│   └── results/                  # Experiment results
├── demo.sh                       # Main demo script
├── setup_env.sh                  # Environment setup
├── .env                          # Local configuration
└── README.md                     # This file
```

## Configuration System

The system uses three modular YAML configurations:

### 1. `synthetic_spectra.yaml` - Physics Parameters
```yaml
# Spectral grid
n_bins: 1024
wavelength_min: 300.0  # nm
wavelength_max: 1200.0  # nm

# Sampling distributions
sampling:
  temperature:
    range: [3000, 7000]  # Kelvin
    distribution: "uniform"
  noise:
    range: [0.01, 0.15]  # 1-15% multiplicative
    distribution: "log_uniform"
  abundance:
    range: [0.0, 1.0]
    distribution: "uniform"

# Spectral lines (20 components)
lines:
  - label: "comp1"
    center: 732.0   # nm
    width: 3.2      # nm (Gaussian sigma)
    strength: 0.8   # Emission line
  # ... 19 more components
```

### 2. `data_distribution.yaml` - Compositional Gap Setup
```yaml
# Training: Individual components only
train_combinations:
  - []          # No components (pure blackbody)
  - ["comp1"]   # Component 1 only
  - ["comp2"]   # Component 2 only

# Testing: Unseen combinations
test_combinations:
  - ["comp1", "comp2"]  # Both together (compositional gap!)

# Dataset sizes
n_train: 8000
n_test: 2000
```

### 3. `training.yaml` - Model Configuration
```yaml
model:
  type: "mlp"
  hidden_dims: [256, 128, 64]
  activation: "relu"
  dropout: 0.1

training:
  batch_size: 128
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
```

## Usage Examples

### Basic Pipeline

```bash
# 1. Generate data with compositional gap
python scripts/generate_data.py \
    --spectra-config configs/synthetic_spectra.yaml \
    --distribution-config configs/data_distribution.yaml \
    --exp-dir experiments/my_experiment

# 2. Train model (sees only individual components)
python scripts/train.py \
    --data configs/synthetic_spectra.yaml \
    --distribution configs/data_distribution.yaml \
    --training configs/training.yaml

# 3. Evaluate (tests on unseen combinations)
python scripts/evaluate.py \
    --data configs/synthetic_spectra.yaml \
    --distribution configs/data_distribution.yaml \
    --training configs/training.yaml
```

Note: `generate_data.py` also accepts legacy arguments (`--data`, `--distribution`, `--training`) for backward compatibility.

### Visualize Data and Explore Physics

```bash
cd example
python run.py meta_config.yaml
# or alternatively:
./run.sh
```

This creates a complete experiment in `example/exp/` with:
- `configs/`: Configuration files for reproducibility
- `data/`: Generated synthetic datasets
- `plots/`: Comprehensive visualizations
  - `diverse_samples.png`: Various spectra with blackbody baselines
  - `spectral_lines.png`: All 20 component profiles
  - `train_samples.png`: Training distribution (individual components)
  - `test_samples.png`: Test distribution (combinations)
  - `abundance_scatter.png`: Compositional gap visualization

### Custom Experiments

Create new distribution configs to test different hypotheses:

```yaml
# Experiment: Can model generalize from pairs to triplets?
train_combinations:
  - ["comp1", "comp2"]
  - ["comp3", "comp4"]
  - ["comp5", "comp6"]

test_combinations:
  - ["comp1", "comp2", "comp3"]  # Triplet!
  - ["comp4", "comp5", "comp6"]  # Another triplet!
```

## Key Results

When running the default compositional gap experiment:

1. **Training**: Model sees comp1 alone OR comp2 alone
2. **Testing**: Model must predict comp1 AND comp2 together
3. **Expected outcome**: 
   - Good performance on individual components
   - Degraded but non-zero performance on combinations
   - Demonstrates partial compositional generalization

## Environment Variables

The `.env` file (created by `setup_env.sh`) contains:
```bash
PROJECT_ROOT=/path/to/SpectralCompGenTutorial
DATA_DIR=./data              # Can be changed to fast storage
CUDA_VISIBLE_DEVICES=0       # GPU selection
TORCH_NUM_THREADS=4          # CPU threads
```

## Extending the Tutorial

### Add New Models

In `src/models/mlp.py`:
```python
class TransformerSpectralModel(nn.Module):
    def __init__(self, n_components, d_model=256):
        # Your implementation
        pass

# Use in config:
# model:
#   type: "transformer"
#   d_model: 256
```

### Modify Physics

In `configs/synthetic_spectra.yaml`:
- Add more spectral lines
- Change temperature ranges
- Modify noise distributions
- Adjust wavelength coverage

### Create New Experiments

1. Design hypothesis about generalization
2. Create new `data_distribution.yaml`
3. Run pipeline with new config
4. Analyze results

## Troubleshooting

**Issue**: CUDA out of memory
- Reduce `batch_size` in training.yaml

**Issue**: Poor convergence
- Increase `learning_rate` or reduce `weight_decay`
- Check data normalization

**Issue**: No generalization
- Verify test combinations are truly unseen
- Try different model architectures
- Analyze component correlations

## Educational Goals

This tutorial helps understand:
- **Compositional generalization**: Core challenge in AI
- **Controlled experiments**: Isolating specific behaviors
- **Systematic evaluation**: Beyond aggregate metrics
- **Interpretable testbeds**: Understanding before scaling

## License

MIT License - see LICENSE file

## Acknowledgments

This tutorial was developed for teaching neural network behavior through synthetic experiments. The spectral model provides an intuitive yet challenging testbed for compositional generalization.

---

*"The ability to understand the whole from its parts is fundamental to intelligence."*