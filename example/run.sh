#!/bin/bash

# Simple wrapper script that cleans up and runs the Python runner

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Remove existing experiment directory
EXP_DIR="$SCRIPT_DIR/exp"
if [ -d "$EXP_DIR" ]; then
    rm -rf "$EXP_DIR"
fi

# Run the Python script
python "$SCRIPT_DIR/run.py" "$SCRIPT_DIR/meta_config.yaml"