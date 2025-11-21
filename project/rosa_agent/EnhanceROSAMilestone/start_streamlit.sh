#!/bin/bash
# Start Streamlit app for Enhanced ROSA Agent

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Source .env file from workspace root if it exists
if [ -f "../../../.env" ]; then
    set -a
    source ../../../.env
    set +a
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. Please set it in your .env file."
fi

# Run Streamlit app
echo "Starting Streamlit app for Enhanced ROSA Agent..."
echo "The app will be available at http://localhost:8501"
streamlit run streamlit_app.py




