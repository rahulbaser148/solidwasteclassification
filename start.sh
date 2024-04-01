#!/bin/bash

# Set up virtual environment with Python 3.10
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install packages from requirements.txt
pip install -r requirement.txt

# Run Python script
python app.py
