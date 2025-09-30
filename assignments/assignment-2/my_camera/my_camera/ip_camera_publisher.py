#!/usr/bin/env python3

import rclpy  # Import the ROS 2 Python client library
from rclpy.node import Node  # Import the Node class from ROS 2
import cv2  # Import OpenCV for computer vision tasks
from cv_bridge import CvBridge  # Import CvBridge to convert between ROS and OpenCV images
from sensor_msgs.msg import Image  # Import the Image message type from sensor_msgs
import argparse  # Import argparse for command-line arguments


class IPCameraPublisher(Node):
    """
    A ROS2 node that streams video from an IP camera and publishes it to /camera/image_raw topic.
    """
    
    def __init__(self, ip):
        super().__init__('ip_camera_publisher')  # Initialize the Node with the name 'ip_camera_publisher'
        
        # Create a publisher for the Image topic
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 1)
        
        # Create a timer to call timer_callback every 0.001 seconds (approximately 100 FPS)
        # Change timer from 0.001 to 0.033 (30 FPS)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        # Open the IP camera stream
        self.cap = cv2.VideoCapture(f'http://{ip}/video')
        
        # Check if the webcam is opened successfully
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open IP camera stream')
            raise RuntimeError("Could not open IP camera")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.bridge = CvBridge()  # Initialize the CvBridge to convert between ROS and OpenCV images
        
        self.get_logger().info(f"IP camera publisher node started with IP: {ip}")
        self.get_logger().info("Publishing IP camera feed to /camera/image_raw topic")

    def timer_callback(self):
        """
        Timer callback function that captures a frame and publishes it.
        """
        try:
            ret, frame = self.cap.read()  # Capture a frame from the webcam
            if ret:  # Check if the frame was captured successfully
                # Resize frame to standard size
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                
                # Convert the OpenCV image to a ROS Image message
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                
                # Publish the Image message
                self.publisher.publish(msg)
                
                # Log every 100 frames to avoid spam
                if hasattr(self, 'frame_count'):
                    self.frame_count += 1
                else:
                    self.frame_count = 1
                
                if self.frame_count % 100 == 0:
                    self.get_logger().info(f"Published {self.frame_count} frames from IP camera")
            else:
                self.get_logger().error('Failed to capture image from IP camera')
                
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {str(e)}")
    
    def destroy_node(self):
        """
        Clean up resources when node is destroyed.
        """
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("IP camera released")
        super().destroy_node()


def main(args=None):
    """
    Main function to run the IP camera publisher node.
    """
    parser = argparse.ArgumentParser(description='ROS2 IP Camera Streamer')
    parser.add_argument('--ip', type=str, required=True, help='IP address and port of the IP camera (e.g., 192.168.0.180:8080)')
    cli_args = parser.parse_args()

    rclpy.init(args=args)  # Initialize the ROS 2 Python client library
    node = IPCameraPublisher(cli_args.ip)  # Create an instance of the IPCameraPublisher with the IP address and port
    try:
        rclpy.spin(node)  # Spin the node to keep it alive and processing callbacks
    except KeyboardInterrupt:
        pass  # Allow the user to exit with Ctrl+C
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()  # Shut down the ROS 2 Python client library


if __name__ == '__main__':
    main()  # Run the main function if this script is executed
