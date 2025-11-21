#!/bin/bash
# Start ROSA Agent with MockMessageBus (no NATS/ROS required)
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
fi

export USE_NATS=false
echo "Starting ROSA Agent with MockMessageBus (no ROS/NATS required)..."
python3 agent.py
