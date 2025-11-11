#!/bin/bash

set -e

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
pip install -r requirements.txt

echo "All dependencies installed successfully"

echo "Running the classification model (classification.py)"
cd project/classification
python classification.py    

echo "Running the classification model (classification2.py)"
python classification2.py

echo "Classification complete! Check the output directory for results."
echo "Results saved in: project/classification/output/"
