#!/bin/bash

echo "==================================="
echo "IDS System - Quick Start"
echo "==================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Check if models exist
if [ ! -f "models/ids_model_ensemble.pkl" ]; then
    echo "Models not found. Training models..."
    echo "This will take 3-5 minutes on i3 processor..."
    echo ""
    python train.py
    echo ""
fi

# Run the application
echo "==================================="
echo "Starting IDS Web Application..."
echo "==================================="
echo ""
echo "Open your browser and go to:"
echo "http://127.0.0.1:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
