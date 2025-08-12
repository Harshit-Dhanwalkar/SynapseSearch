#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

function setup_project() {
    echo "Creating Python virtual environment..."
    if [ -d ".venv" ]; then
        echo "Virtual environment '.venv' already exists. Skipping creation."
    else
        echo "Creating Python virtual environment..."
        python3.11 -m venv .venv
    fi

    echo "Installing dependencies from requirements.txt..."
    .venv/bin/pip install -r requirements.txt

    echo "Setup complete!"
    echo "To activate the virtual environment, please run the following command:"
    echo "  source .venv/bin/activate"
}

function clean_project() {
    echo "Cleaning project..."
    if [ -d ".venv" ]; then
        echo "Removing virtual environment directory (.venv)..."
        rm -rf .venv
    fi
    if [ -f "search_engine.log" ]; then
        echo "Removing log file (search_engine.log)..."
        rm search_engine.log
    fi
    if [ -d "data" ]; then
        echo "Removing data directory (data)..."
        rm -rf data
    fi
    echo "Clean operation complete."
}


if [ -z "$1" ]; then
    echo "Usage: ./setup.sh [setup|clean]"
    echo "  setup: Creates a virtual environment and installs dependencies."
    echo "  clean: Removes the virtual environment, log file, and data directory."
    exit 1
elif [ "$1" == "setup" ]; then
    setup_project
elif [ "$1" == "clean" ]; then
    clean_project
else
    echo "Error: Unknown argument '$1'"
    echo "Usage: ./setup.sh [setup|clean]"
    echo "  setup: Creates a virtual environment and installs dependencies."
    echo "  clean: Removes the virtual environment, log file, and data directory."
    exit 1
fi
