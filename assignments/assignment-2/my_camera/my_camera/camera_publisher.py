#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import argparse


class CameraPublisher(Node):
    """
    A ROS2 node that captures video from the default camera and publishes it to /camera/image_raw topic.
    """
    
    def __init__(self, cam=0):
        super().__init__('camera_publisher')
        
        # Create publisher for camera images
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 1)
        
        # Create timer for publishing frames
        timer_period = 0.001  # High frequency timer
        # Change timer from 0.001 to 0.033 (30 FPS)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        # Initialize OpenCV camera capture
        self.cap = cv2.VideoCapture(cam)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            raise RuntimeError("Could not open camera")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Initialize CV bridge for converting OpenCV images to ROS messages
        self.bridge = CvBridge()
        
        self.get_logger().info("Camera publisher node started")
        self.get_logger().info("Publishing camera feed to /camera/image_raw topic")
    
    def timer_callback(self):
        """
        Timer callback function that captures a frame and publishes it.
        """
        try:
            # Capture frame from camera
            ret, frame = self.cap.read()
            
            if ret:
                # Convert OpenCV image to ROS Image message (keep BGR encoding)
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                
                # Publish the image
                self.publisher.publish(ros_image)
                
                # Log every 100 frames to avoid spam
                if hasattr(self, 'frame_count'):
                    self.frame_count += 1
                else:
                    self.frame_count = 1
                
                if self.frame_count % 100 == 0:
                    self.get_logger().info(f"Published {self.frame_count} frames")
            else:
                self.get_logger().error('Failed to capture image')
                
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {str(e)}")
    
    def destroy_node(self):
        """
        Clean up resources when node is destroyed.
        """
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("Camera released")
        super().destroy_node()


def main(args=None):
    """
    Main function to run the camera publisher node.
    """
    parser = argparse.ArgumentParser(description='ROS2 Camera Node')
    parser.add_argument('--cam', type=int, default=0, help='Index of the camera (default is 0)')
    cli_args = parser.parse_args()
    
    rclpy.init(args=args)
    
    try:
        camera_publisher = CameraPublisher(cli_args.cam)
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'camera_publisher' in locals():
            camera_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



