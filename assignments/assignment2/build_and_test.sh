#!/bin/bash
"""
Build and test script for assignment 2
"""

set -e

echo "Building turtlebot_object_detection package..."

# Build the package
cd /workspaces/eng-ai-agents/assignments/assignment2
colcon build --packages-select turtlebot_object_detection

echo "Build complete!"
echo ""
echo "To test the package:"
echo "1. Source the workspace:"
echo "   source /workspaces/eng-ai-agents/assignments/assignment2/install/setup.bash"
echo ""
echo "2. Start TurtleBot3 simulation:"
echo "   ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py"
echo ""
echo "3. Run object detection (in another terminal):"
echo "   source /workspaces/eng-ai-agents/turtlebot_ws/install/setup.bash"
echo "   source /workspaces/eng-ai-agents/assignments/assignment2/install/setup.bash"
echo "   ros2 run turtlebot_object_detection object_detection_node"
echo ""
echo "4. View results:"
echo "   ros2 run rqt_image_view rqt_image_view /camera/annotated"
