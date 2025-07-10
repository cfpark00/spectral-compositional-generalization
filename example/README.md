# Comprehensive Example

This example demonstrates the complete spectral compositional generalization workflow, from understanding the physics to visualizing the compositional gap.

## Overview

This example:
1. **Copies configuration files** for reproducibility
2. **Generates synthetic spectral data** with compositional gap
3. **Creates comprehensive visualizations** of both physics and data distributions

## Quick Start

### Option 1: Using Meta Config (Recommended)

```bash
cd example
python run.py meta_config.yaml
```

### Option 2: Using Shell Script

```bash
cd example
./run.sh
```

Both methods create a complete experiment in `example/exp/` with:
- `configs/`: Copied configuration files
- `data/`: Generated synthetic datasets
- `plots/`: All visualization plots

## Generated Visualizations

### Physics Visualizations

#### `diverse_samples.png`
Shows 10 random spectra demonstrating the physics:
- **Solid lines**: Complete spectra with all physical effects
- **Dashed lines**: Pure blackbody baselines (Planck's law)
- **Labels**: Temperature (K) and number of active components
- **Demonstrates**: How components modulate the blackbody continuum

#### `spectral_lines.png`
Comprehensive view of all 20 spectral components:
- **Top panel**: Normalized line profiles (3.2nm to 350nm widths)
- **Bottom panel**: Component centers and strengths
  - Blue dots: Emission lines (positive strength)
  - Red dots: Absorption lines (negative strength)

### Data Distribution Visualizations

#### `train_samples.png`
Shows the training distribution:
- **Gray lines**: No components (pure blackbody + noise)
- **Blue lines**: Component 1 only
- **Green lines**: Component 2 only
- **Key insight**: Model NEVER sees comp1 AND comp2 together

#### `test_samples.png`
Shows the test distribution:
- **5 samples** all containing BOTH comp1 AND comp2
- **Challenge**: Model must predict both components together

#### `abundance_scatter.png`
Scatter plot in comp1-comp2 abundance space:
- **Blue circles**: Training data (along axes only)
- **Red squares**: Test data (interior region)
- **Visualizes**: The compositional gap clearly

## Directory Structure

After running the example:

```
example/
├── visualize_all.py   # Visualization script
├── run.py             # Python runner script (recommended)
├── run.sh             # Shell runner script
├── meta_config.yaml   # Meta configuration example
├── README.md          # This file
└── exp/               # Generated experiment (created at runtime)
    ├── configs/       # Configuration files
    │   ├── synthetic_spectra.yaml
    │   └── data_distribution.yaml
    ├── data/          # Synthetic datasets
    │   ├── train_spectra.pt
    │   ├── train_abundances.pt
    │   ├── test_spectra.pt
    │   ├── test_abundances.pt
    │   └── metadata.json
    └── plots/         # All visualizations
        ├── diverse_samples.png
        ├── spectral_lines.png
        ├── train_samples.png
        ├── test_samples.png
        └── abundance_scatter.png
```

## The Compositional Gap

This example clearly demonstrates:

1. **Physics Model**: Realistic stellar spectra with 20 components
2. **Training Distribution**: Only individual components
3. **Test Distribution**: Unseen combinations
4. **The Challenge**: Can models learn to compose?

## Configuration

### Meta Configuration (`meta_config.yaml`)

The Python runner uses a meta configuration file that specifies:

```yaml
# Experiment directory (where all outputs will be saved)
exp_dir: "./exp"

# Path to spectral configuration file (physics parameters)
spectra_config_path: "../configs/synthetic_spectra.yaml"

# Path to data distribution configuration file (train/test splits)
distribution_config_path: "../configs/data_distribution.yaml"

# Random seed for data generation (for reproducibility)
data_seed: 42
```

### Data Configuration Files

The example uses two main configuration files:

### `synthetic_spectra.yaml`
- Physics parameters (wavelengths, temperatures, noise)
- Spectral line definitions (20 components)
- Sampling distributions

### `data_distribution.yaml`
- Train/test split definitions
- Component combination rules
- Sample counts per split

## Requirements

- Environment setup complete (`.env` file exists)
- Python packages: matplotlib, numpy, torch, pyyaml

## Customization

### Using Meta Config (Recommended)

1. Copy and modify `meta_config.yaml`:
   - Change `exp_dir` to your desired output location
   - Update config paths to your custom configurations
   - Set a different `data_seed` for varied data
2. Run: `python run.py your_meta_config.yaml`

### Using Shell Script

1. Edit config files in `configs/`
2. Re-run `./run.sh`
3. New experiment will be created in `exp/`

## Educational Value

This example helps understand:
- **Spectral physics**: How synthetic spectra are generated
- **Data distributions**: How to create compositional gaps
- **Visualization**: Multiple ways to view the challenge
- **Complete workflow**: From configs to visualizations

---

*"One example to rule them all, one example to show them,
One example to bring all plots, and in the exp directory bind them."*