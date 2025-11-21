#!/bin/bash
# Start ROS-NATS adapter
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ../../../.venv/bin/activate ]; then
    source ../../../.venv/bin/activate
fi

# Source ROS if available
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

export NATS_URL="${NATS_URL:-nats://localhost:4222}"
echo "Starting ROS-NATS adapter (NATS_URL=$NATS_URL)..."
python3 ros_nats_adapter.py

