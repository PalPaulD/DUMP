#!/bin/bash
# Setup script for DUMP environment

set -e  # Exit on error

echo "🚀 Setting up DUMP environment..."
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo " Error: conda not found."
    echo "   Please install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove old environment if it exists
if [ -d "./DUMP_env" ]; then
    echo "  Found existing DUMP_env directory"
    read -p "   Remove it and create fresh environment? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ./DUMP_env
    else
        echo " Aborted. Please remove ./DUMP_env manually or use a different name."
        exit 1
    fi
fi

# Create conda environment
echo " Creating conda environment..."
echo " This installs PyTorch via conda (avoids permission issues)"
conda env create -f environment.yml -p "./DUMP_env"

# Install DUMP package
echo ""
echo "🔧 Installing DUMP package in editable mode..."
"./DUMP_env/bin/pip" install -e .

echo ""
echo " Setup complete!"
echo ""
echo " Now you are ready to run scripts:"
echo "   1. Activate environment:"
echo "      conda activate ./DUMP_env"
echo ""
echo "   2. Fit scalers:"
echo "      python scripts/compute_scalers.py --use_every_n=1"
echo ""
echo "   3. Run training:"
echo "      python scripts/train.py --config configs/example.yaml"
echo ""
