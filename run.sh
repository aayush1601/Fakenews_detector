#!/bin/bash
# Script to run the FakeNewsDB Flask application

echo "Setting Oracle environment variables..."
export ORACLE_USER=fakenews
export ORACLE_PASSWORD=yourpassword
export ORACLE_DSN=localhost:1521/XEPDB1

# Determine virtual environment path
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ] && [ ! -d ".venv" ]; then
    echo "Creating virtual environment ($VENV_DIR)..."
    python3 -m venv $VENV_DIR
fi

if [ -d ".venv" ]; then
    VENV_DIR=".venv"
fi

# Use the Python executable directly from the virtual environment
PYTHON_EXEC="$VENV_DIR/bin/python"

echo "Installing/verifying dependencies..."
$PYTHON_EXEC -m pip install -q -r requirements.txt

# Check if models exist
if [ ! -f "models/nb.pkl" ]; then
    echo "Models not found. Training ML models (this takes 2-5 minutes)..."
    $PYTHON_EXEC train_models.py
fi

echo "Starting Flask app..."
$PYTHON_EXEC app.py
