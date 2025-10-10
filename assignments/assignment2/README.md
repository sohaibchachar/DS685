# Assignment 2: TurtleBot Object Detection with Semantic Database

**Assignment Name:** Assignment 2  
**Date of Submission:** October 09, 2025  
**Author:** Sohaib Chachar

---

## 📋 Overview

This assignment implements real-time object detection for TurtleBot4 using Faster R-CNN with ResNet50 backbone. Detected objects are stored in a PostgreSQL database with semantic localization information, enabling spatial queries and object tracking across the robot's environment.

### Key Features
- **Real-time Object Detection**: Detects 80 COCO classes using pre-trained Faster R-CNN
- **Feature Embeddings**: Extracts 2048-dimensional feature vectors using ResNet50 for each detected object
- **Semantic Localization**: Stores robot pose (x, y, θ) and region information with each detection
- **PostgreSQL + pgvector**: Vector database for similarity search and spatial queries
- **ROS2 Integration**: Subscribes to camera feed and AMCL pose for autonomous localization

### What Gets Stored in Database
- **Object Information**: Class name, class ID, confidence score
- **Feature Embeddings**: 2048-dimensional vectors for semantic similarity matching
- **Robot Location**: Robot's X, Y position and orientation (theta) when object was detected
- **Bounding Boxes**: Object location in image space [x1, y1, x2, y2]
- **Region Names**: Auto-generated semantic regions (e.g., `location_5_-3`)
- **Timestamps**: Detection time for temporal queries

---

## 🚀 Instructions to Run

### Prerequisites
- Docker container should be running (`eng-ai-agents-ros`)
- You should be inside the Docker container

### Step 1: Launch TurtleBot Demo World

Open **Terminal 1** and execute the following commands:

```bash
# Navigate to TurtleBot workspace
cd /workspaces/eng-ai-agents/turtlebot_ws

# Source the workspace
source /workspaces/eng-ai-agents/turtlebot_ws/install/setup.bash

# Launch the demo world with Gazebo and RViz
ros2 launch tb_worlds tb_demo_world.launch.py
```

**Note:** This demo world includes COCO-class objects placed on benches for testing object detection.

Wait for Gazebo and RViz to fully load before proceeding to the next step.

---

### Step 2: Run Object Detection Node

Open **Terminal 2** and execute the following commands:

```bash
# Navigate to assignment directory
cd /workspaces/eng-ai-agents/assignments/assignment2

# Build the object detection package
colcon build --packages-select turtlebot_object_detection

# Source the local workspace
source /workspaces/eng-ai-agents/assignments/assignment2/install/setup.bash

# Start PostgreSQL database
sudo service postgresql start

# Run the object detection node
ros2 run turtlebot_object_detection object_detection_node
```

The object detection node will:
- Subscribe to `/camera/image_raw` topic
- Subscribe to `/amcl_pose` for robot localization
- Detect objects in real-time
- Store detections in PostgreSQL database
- Publish annotated images to `/camera/annotated`

---



