#!/bin/bash

# Script to run Gazebo and RViz2 in the container environment
# This script sets up the necessary environment variables and starts the applications

echo "Setting up ROS 2 environment for Gazebo and RViz2..."

# Set up display environment
export DISPLAY=:99
export QT_X11_NO_MITSHM=1

# Source ROS 2 environment
set +u
source /opt/ros/jazzy/setup.bash
set -u

# Start virtual display if not already running
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting virtual display (Xvfb)..."
    Xvfb :99 -screen 0 1024x768x24 &
    sleep 2
fi

# Function to run Gazebo
run_gazebo() {
    echo "Starting Gazebo Sim..."
    echo "Available options:"
    echo "  -s: Server only (headless)"
    echo "  -g: GUI only"
    echo "  -r: Run simulation on start"
    echo "  -v 4: Verbose output"
    echo ""
    echo "Example commands:"
    echo "  gz sim -s                    # Headless server"
    echo "  gz sim -g                    # GUI only"
    echo "  gz sim -r                    # Run with simulation"
    echo "  gz sim -v 4                  # Verbose output"
    echo ""
    echo "Starting Gazebo Sim with default settings..."
    gz sim -v 4
}

# Function to run RViz2
run_rviz2() {
    echo "Starting RViz2..."
    echo "Available options:"
    echo "  -d <config>: Load display config"
    echo "  -f <frame>: Set fixed frame"
    echo "  -l: Enable Ogre logging"
    echo ""
    echo "Starting RViz2..."
    rviz2
}

# Function to run both
run_both() {
    echo "Starting both Gazebo and RViz2..."
    echo "Gazebo will run in server mode (headless), RViz2 will show the GUI"
    
    # Start Gazebo in server mode
    echo "Starting Gazebo Sim server..."
    gz sim -s -v 4 &
    GAZEBO_PID=$!
    
    # Wait a moment for Gazebo to start
    sleep 3
    
    # Start RViz2
    echo "Starting RViz2..."
    rviz2
    
    # Clean up when RViz2 exits
    echo "Cleaning up..."
    kill $GAZEBO_PID 2>/dev/null
}

# Main menu
case "${1:-menu}" in
    "gazebo")
        run_gazebo
        ;;
    "rviz")
        run_rviz2
        ;;
    "both")
        run_both
        ;;
    "menu"|*)
        echo "ROS 2 Gazebo and RViz2 Launcher"
        echo "================================"
        echo ""
        echo "Usage: $0 [gazebo|rviz|both]"
        echo ""
        echo "Options:"
        echo "  gazebo  - Run Gazebo Sim only"
        echo "  rviz    - Run RViz2 only"
        echo "  both    - Run Gazebo (server) + RViz2 (GUI)"
        echo "  menu    - Show this menu (default)"
        echo ""
        echo "Examples:"
        echo "  $0 gazebo    # Start Gazebo Sim"
        echo "  $0 rviz      # Start RViz2"
        echo "  $0 both      # Start both applications"
        echo ""
        echo "Environment:"
        echo "  DISPLAY=$DISPLAY"
        echo "  ROS_DISTRO=$ROS_DISTRO"
        echo ""
        ;;
esac
