#! /bin/bash

# Create a new conda env using python 3.11 and name it assignment_env
if conda env list | grep -q "assignment_env"; then
    echo "Deleting existing env"
    conda remove -n assignment_env --all -y
fi

conda create -n assignment_env python=3.11 -y

# Init conda (if not already done)
source ~/.bash_profile 2>/dev/null || source ~/.bashrc 2>/dev/null || eval "$(conda shell.bash hook)"

# Activate the env
conda activate assignment_env

# Install the requirements
pip install -r requirements.txt

# Make sure price_data.csv is in the same directory as the script
if [ ! -f price_data.csv ]; then
    echo "price_data.csv not found. Please download it first."
    exit 1
fi

echo "Setup complete. You can now run the pipeline using main.py"
