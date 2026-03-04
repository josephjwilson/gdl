#!/bin/bash
# Install script for the Graphormer head specialisation environment.
# Tested with CUDA 12.4 and Python 3.9.

set -e

ENV_NAME="graphormer"

conda create -n $ENV_NAME python=3.9 -y
conda run -n $ENV_NAME pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# PyG + sparse dependencies
conda run -n $ENV_NAME pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
conda run -n $ENV_NAME pip install torch-geometric==2.6.1

# Core dependencies
conda run -n $ENV_NAME pip install \
    numpy==1.26.4 \
    networkx==3.2.1 \
    tqdm \
    scipy \
    matplotlib \
    seaborn \
    pandas \
    ogb==1.3.6 \
    rdkit \
    Cython

# Compile the Cython algos extension
conda run -n $ENV_NAME python setup.py build_ext --inplace

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
