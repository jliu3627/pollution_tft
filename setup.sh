#!/bin/bash

set -e

ENV_DIR="tft_venv"

# Create venv
echo "Creating virtualenv..."
python3.10 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Install packages
echo "Installing packages from requirements.txt..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo -e "To activate the environment: \nsource $ENV_DIR/bin/activate"