#!/bin/bash
# Start ROSA Agent with NATS
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ../../../.venv/bin/activate ]; then
    source ../../../.venv/bin/activate
fi

# Load .env file from workspace root if it exists
if [ -f ../../../.env ]; then
    set -a
    source ../../../.env
    set +a
    echo "Loaded environment variables from .env"
fi

export USE_NATS=true
export NATS_URL="${NATS_URL:-nats://localhost:4222}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set. Please set it in .env file or export it."
    exit 1
fi

echo "Starting ROSA Agent with NATS (NATS_URL=$NATS_URL)..."
python3 agent.py

