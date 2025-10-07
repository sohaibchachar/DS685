# TurtleBot Object Detection with Vector Database

This project implements real-time object detection for TurtleBot3 using PyTorch and stores semantic information in a PostgreSQL database with pgvector extension for vector similarity search.

## Prerequisites

### System Requirements
- Ubuntu 20.04/22.04
- ROS 2 Humble/Iron
- Python 3.8+
- NVIDIA GPU (recommended for CUDA acceleration)

### Dependencies
- ROS 2 Humble or Iron
- TurtleBot3 packages
- Gazebo simulation environment
- PostgreSQL with pgvector extension
- PyTorch with CUDA support (optional)

## Installation

### 1. Install ROS 2 and TurtleBot3 Packages

```bash
# Install ROS 2 Humble (if not already installed)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop python3-argcomplete python3-colcon-common-extensions python3-rosdep python3-vcstool

# Initialize rosdep
sudo rosdep init
rosdep update

# Install TurtleBot3 packages
sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description ros-humble-turtlebot3-msgs

# Set TurtleBot3 model
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install PostgreSQL and pgvector

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
sudo apt install postgresql-14-pgvector  # Adjust version based on your PostgreSQL version

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 3. Set up Database

```bash
# Switch to postgres user and create database
sudo -u postgres psql

# In PostgreSQL shell:
CREATE DATABASE agents;
CREATE USER postgres WITH PASSWORD 'postgres';
GRANT ALL PRIVILEGES ON DATABASE agents TO postgres;
\q

# Create .env file for database configuration (optional)
echo "POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agents
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres" > .env
```

### 4. Build the Package

```bash
# Navigate to your workspace
cd /workspaces/eng-ai-agents/assignments/assignment2

# Install Python dependencies
pip install torch torchvision opencv-python numpy Pillow psycopg2-binary

# Build the package
colcon build --packages-select turtlebot_object_detection

# Source the workspace
source install/setup.bash
```

## Launch Instructions

### Step 1: Launch Gazebo Simulation

Open a new terminal and run:

```bash
# Source ROS 2 and workspace
source /opt/ros/humble/setup.bash
source /workspaces/eng-ai-agents/assignments/assignment2/install/setup.bash

# Set TurtleBot3 model
export TURTLEBOT3_MODEL=burger

# Launch Gazebo with TurtleBot3 in empty world
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

### Step 2: Launch RViz Visualization

Open another terminal and run:

```bash
# Source ROS 2 and workspace
source /opt/ros/humble/setup.bash
source /workspaces/eng-ai-agents/assignments/assignment2/install/setup.bash

# Set TurtleBot3 model
export TURTLEBOT3_MODEL=burger

# Launch RViz with TurtleBot3 configuration
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True
```

Alternatively, for a simpler RViz setup:

```bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix --share turtlebot3_navigation2)/rviz/tb3_navigation2.rviz
```

### Step 3: Launch Object Detection Node

Open a third terminal and run:

```bash
# Source ROS 2 and workspace
source /opt/ros/humble/setup.bash
source /workspaces/eng-ai-agents/assignments/assignment2/install/setup.bash

# Launch object detection node
ros2 launch turtlebot_object_detection object_detection.launch.py
```

With custom parameters:

```bash
# Launch with custom confidence threshold
ros2 launch turtlebot_object_detection object_detection.launch.py confidence_threshold:=0.7

# Launch without simulation time
ros2 launch turtlebot_object_detection object_detection.launch.py use_sim_time:=false
```

### Step 4: Verify Vector Database Setup

The object detection node automatically initializes the vector database schema. You can verify the setup:

```bash
# Connect to PostgreSQL database
psql -h localhost -U postgres -d agents

# Check if tables were created
\dt

# View sample data (after running object detection)
SELECT * FROM regions;
SELECT * FROM objects LIMIT 5;
SELECT * FROM object_observations LIMIT 5;

# Exit PostgreSQL
\q
```

## Usage

### Controlling the Robot

In a new terminal, you can control the TurtleBot3:

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Teleop control
ros2 run turtlebot3_teleop teleop_keyboard
```

### Monitoring Object Detection

```bash
# View detection results
ros2 topic echo /object_detections

# View processed camera feed
ros2 run rqt_image_view rqt_image_view
```

### Database Queries

The system stores detected objects with their embeddings for semantic search. Example queries:

```sql
-- Find all objects of a specific class
SELECT * FROM objects WHERE class_name = 'person';

-- Find objects near a specific location
SELECT o.class_name, o.confidence, obs.robot_x, obs.robot_y 
FROM objects o 
JOIN object_observations obs ON o.id = obs.object_id 
WHERE obs.region_id = 1;

-- Semantic similarity search (example with a specific embedding)
SELECT class_name, confidence, embedding <-> '[0.1, 0.2, ...]' AS distance 
FROM objects 
ORDER BY embedding <-> '[0.1, 0.2, ...]' 
LIMIT 5;
```

## Configuration

### Launch Parameters

- `use_sim_time`: Use simulation time (default: true)
- `confidence_threshold`: Minimum confidence for object detection (default: 0.5)

### Environment Variables

Create a `.env` file or set environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=agents
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
```

## Troubleshooting

### Common Issues

1. **CUDA not available**: The system will fall back to CPU processing if CUDA is not available.

2. **Database connection failed**: Ensure PostgreSQL is running and credentials are correct.

3. **Model loading slow**: The first run downloads the pre-trained model (~200MB).

4. **No camera feed**: Ensure Gazebo is running and the camera topic is published.

### Debug Commands

```bash
# Check available topics
ros2 topic list

# Check camera topic
ros2 topic echo /camera/image_raw

# Check node status
ros2 node list
ros2 node info /object_detection_node

# Check database connection
psql -h localhost -U postgres -d agents -c "SELECT version();"
```

## Project Structure

```
turtlebot_object_detection/
├── launch/
│   └── object_detection.launch.py
├── turtlebot_object_detection/
│   ├── __init__.py
│   ├── object_detection_node.py
│   └── semantic_db.py
├── package.xml
├── setup.py
└── README.md
```

## Features

- Real-time object detection using Faster R-CNN
- COCO dataset class support (80 classes)
- Semantic database with vector embeddings
- PostgreSQL + pgvector integration
- ROS 2 integration with TurtleBot3
- Configurable confidence thresholds
- GPU acceleration support

## License

Apache-2.0
