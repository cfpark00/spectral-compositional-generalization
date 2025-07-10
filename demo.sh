#!/bin/bash

# Demo orchestration script for spectral compositional generalization tutorial
# This script runs the complete pipeline (requires .env file)

set -e  # Exit on any error

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please run: ./setup_env.sh"
    exit 1
fi

echo "=============================================="
echo "SPECTRAL COMPOSITIONAL GENERALIZATION DEMO"
echo "=============================================="

# Configuration files
DATA_CONFIG="configs/synthetic_spectra.yaml"
DISTRIBUTION_CONFIG="configs/data_distribution.yaml"
TRAINING_CONFIG="configs/training.yaml"
EXPERIMENT_NAME="demo_simple"

echo "Using modular configuration files:"
echo "  Data: $DATA_CONFIG"
echo "  Distribution: $DISTRIBUTION_CONFIG"
echo "  Training: $TRAINING_CONFIG"
echo "Experiment name: $EXPERIMENT_NAME"

# Step 1: Generate data
echo ""
echo "Step 1: Generating synthetic spectral data..."
echo "----------------------------------------------"
python scripts/generate_data.py --data $DATA_CONFIG --distribution $DISTRIBUTION_CONFIG --training $TRAINING_CONFIG

# Step 2: Train model
echo ""
echo "Step 2: Training neural network model..."
echo "----------------------------------------------"
python scripts/train.py --data $DATA_CONFIG --distribution $DISTRIBUTION_CONFIG --training $TRAINING_CONFIG

# Step 3: Evaluate model
echo ""
echo "Step 3: Evaluating compositional generalization..."
echo "----------------------------------------------"
python scripts/evaluate.py --data $DATA_CONFIG --distribution $DISTRIBUTION_CONFIG --training $TRAINING_CONFIG

echo ""
echo "=============================================="
echo "DEMO COMPLETED SUCCESSFULLY!"
echo "=============================================="
echo "Results can be found in: ./data/results/$EXPERIMENT_NAME/"
echo "Check the evaluation_results.json for detailed metrics."
echo "Visualization plots have been saved as PNG files."
echo "=============================================="