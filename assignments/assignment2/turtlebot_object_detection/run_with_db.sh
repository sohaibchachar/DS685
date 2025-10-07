#!/bin/bash

echo "🚀 Starting Object Detection with Automatic Database Connection"
echo "=============================================================="
echo ""

# Navigate to the package directory
cd /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection

# Start PostgreSQL if not running
echo "🔧 Starting PostgreSQL..."
sudo service postgresql start 2>/dev/null || echo "PostgreSQL already running"

# Wait a moment for PostgreSQL to start
sleep 2

# Source ROS environment
echo "🔧 Setting up ROS environment..."
source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger

# Source turtlebot workspace (if available)
if [ -f "/workspaces/eng-ai-agents/turtlebot_ws/install/setup.bash" ]; then
    source /workspaces/eng-ai-agents/turtlebot_ws/install/setup.bash
    echo "✅ Sourced turtlebot workspace"
fi

# Source this package
source install/setup.bash

echo ""
echo "🎯 Launching Object Detection Node..."
echo "💾 Database connection will be automatic (no environment variables needed)"
echo "🤖 Robot pose tracking enabled"
echo "📊 Objects will be stored with 2048-d embeddings"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Launch the object detection node
ros2 launch turtlebot_object_detection object_detection.launch.py
