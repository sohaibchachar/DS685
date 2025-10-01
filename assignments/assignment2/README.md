# Assignment 2: TurtleBot3 Object Detection

This assignment implements real-time object detection for TurtleBot3 using a PyTorch COCO pre-trained model.

## Overview

The `turtlebot_object_detection` package provides:
- Real-time object detection using Faster R-CNN with ResNet-50 backbone
- COCO dataset pre-trained model (80 object classes)
- GPU acceleration support (automatically detects CUDA)
- Annotated image publishing with bounding boxes and confidence scores
- Console logging of detected objects

## Package Structure

```
assignments/assignment2/
├── turtlebot_object_detection/
│   ├── launch/
│   │   └── object_detection.launch.py
│   ├── turtlebot_object_detection/
│   │   └── object_detection_node.py
│   ├── package.xml
│   ├── setup.py
│   └── setup.cfg
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
cd /workspaces/eng-ai-agents/assignments/assignment2
pip install torch torchvision opencv-python numpy Pillow
```

### 2. Build the Package

```bash
cd /workspaces/eng-ai-agents/assignments/assignment2
colcon build --packages-select turtlebot_object_detection
source install/setup.bash
```

## Usage

### Step 1: Start TurtleBot3 Simulation

Launch the TurtleBot3 world with camera:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Step 2: Run Object Detection

In a new terminal, source both workspaces and run the detection node:

```bash
# Source turtlebot workspace
source /workspaces/eng-ai-agents/turtlebot_ws/install/setup.bash

# Source assignment workspace
source /workspaces/eng-ai-agents/assignments/assignment2/install/setup.bash

# Run object detection
ros2 run turtlebot_object_detection object_detection_node
```

Or use the launch file:
```bash
ros2 launch turtlebot_object_detection object_detection.launch.py
```

### Step 3: View Results

To see the annotated camera feed:
```bash
ros2 run rqt_image_view rqt_image_view /camera/annotated
```

Or view the original camera feed:
```bash
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

## Topics

### Subscribed Topics
- `/camera/image_raw` (sensor_msgs/Image): Raw camera feed from TurtleBot3

### Published Topics
- `/camera/annotated` (sensor_msgs/Image): Camera feed with object detection annotations

## Configuration

You can modify the following parameters:

- `confidence_threshold`: Minimum confidence for detections (default: 0.5)
- `use_sim_time`: Use simulation time (default: true)

## Detected Objects

The node can detect 80 different object classes from the COCO dataset, including:

**People & Animals:**
- person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:**
- bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Furniture:**
- chair, couch, bed, dining table, toilet, bench

**Objects:**
- bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange
- laptop, mouse, remote, keyboard, cell phone, book, clock, vase
- sports ball, frisbee, kite, baseball bat, tennis racket

## Performance

- **CPU Mode**: ~2-5 FPS (depending on hardware)
- **GPU Mode**: ~10-30 FPS (with CUDA-enabled GPU)
- **Memory Usage**: ~2-4 GB RAM
- **Model Size**: ~170 MB (Faster R-CNN ResNet-50)

## Troubleshooting

### Common Issues

1. **CUDA not available**: The node will automatically fall back to CPU mode
2. **Low FPS**: Try reducing the confidence threshold
3. **Memory issues**: Close other applications or reduce image resolution
4. **Package not found**: Make sure to source both workspaces

### Debug Mode

To see more detailed logging:
```bash
ros2 run turtlebot_object_detection object_detection_node --ros-args --log-level debug
```

## Requirements

- ROS2 Humble
- Python 3.8+
- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- TurtleBot3 simulation packages

## License

Apache-2.0
