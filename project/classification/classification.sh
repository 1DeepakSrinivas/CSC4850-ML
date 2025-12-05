#!/bin/bash

set -e

# Change to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed"
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment"
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

echo "Activating virtual environment"
source venv/bin/activate

echo "Installing dependencies from requirements.txt"
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in current directory"
    exit 1
fi
pip install -r requirements.txt

echo "All dependencies installed successfully"

echo "Running the classification model (classification.py)"
python classification.py

echo "Classification complete! Check the output directory for results."
echo "Results saved in: output/"
echo "Evaluation metrics saved in: output/evals/"
