#!/bin/bash

# Set required environment variables
export AMENT_TRACE_SETUP_FILES=0
export COLCON_TRACE=0
export AMENT_PYTHON_EXECUTABLE=/usr/bin/python3
export COLCON_PREFIX_PATH=/workspaces/eng-ai-agents/assignments/assignment-2

# Source ROS2 environment
source /opt/ros/jazzy/setup.bash

# Source our workspace
source /workspaces/eng-ai-agents/assignments/assignment-2/install/setup.bash

# Run the camera publisher
echo "Starting camera publisher..."
ros2 run my_camera camera_publisher




