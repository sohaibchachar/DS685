#!/usr/bin/env python3
"""
Launch file for object detection with TurtleBot3
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate launch description for object detection
    """
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Confidence threshold for object detection'
    )
    
    # Object detection node
    object_detection_node = Node(
        package='turtlebot_object_detection',
        executable='object_detection_node',
        name='object_detection_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold')
        }]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        confidence_threshold_arg,
        object_detection_node,
    ])
