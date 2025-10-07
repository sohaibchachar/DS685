#!/bin/bash

# Simple script to run object detection with automatic database connection
# No need to set environment variables manually!

echo "🚀 Starting Object Detection Node..."
echo "📊 Database will automatically connect to localhost:5432/agents"
echo "🤖 Robot pose tracking enabled"
echo ""

# Navigate to the package directory
cd /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection

# Source ROS environment
source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger

# Source turtlebot workspace (if available)
if [ -f "/workspaces/eng-ai-agents/turtlebot_ws/install/setup.bash" ]; then
    source /workspaces/eng-ai-agents/turtlebot_ws/install/setup.bash
    echo "✅ Sourced turtlebot workspace"
fi

# Source this package
source install/setup.bash

# Start PostgreSQL if not running
sudo service postgresql start 2>/dev/null || echo "PostgreSQL already running"

echo "🎯 Launching Object Detection Node..."
echo "💡 Database connection will be automatic (no environment variables needed)"
echo ""

# Launch the object detection node
ros2 launch turtlebot_object_detection object_detection.launch.py
