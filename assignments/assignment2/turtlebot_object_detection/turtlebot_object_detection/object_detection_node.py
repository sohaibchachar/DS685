#!/usr/bin/env python3
"""
ROS2 Object Detection Node for TurtleBot3
Subscribes to camera feed and performs object detection using PyTorch COCO model
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from PIL import Image as PILImage
import time


class ObjectDetectionNode(Node):
    """
    ROS2 Node for object detection using PyTorch COCO model
    """
    
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Initialize CV bridge for ROS2 image conversion
        self.bridge = CvBridge()
        
        # COCO class names
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'backpack', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
            'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' , "cell phone", 'book'
        ]
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        # Load pre-trained model
        self.get_logger().info('Loading Faster R-CNN model...')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info('Model loaded successfully!')
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.detection_count = 0
        
        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher for annotated images
        self.annotated_pub = self.create_publisher(Image, '/camera/annotated', 10)
        
        # Publisher for detection results
        self.detection_pub = self.create_publisher(Image, '/detection_results', 10)
        
        self.get_logger().info('Object Detection Node initialized')
        self.get_logger().info('Subscribed to /camera/image_raw')
        self.get_logger().info('Publishing annotated images to /camera/annotated')
        
    def image_callback(self, msg):
        """
        Callback function for processing incoming camera images
        """
        try:
            # Convert ROS2 image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Perform object detection
            detections = self.detect_objects(cv_image)
            
            # Annotate image with detections
            annotated_image = self.annotate_image(cv_image, detections)
            
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
            
            # Log detections
            if detections:
                self.log_detections(detections)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def detect_objects(self, image):
        """
        Perform object detection on the input image
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        
        # Preprocess image
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Process predictions
        detections = []
        pred = predictions[0]
        
        for i in range(len(pred['boxes'])):
            confidence = pred['scores'][i].item()
            if confidence > self.confidence_threshold:
                box = pred['boxes'][i].cpu().numpy()
                class_id = pred['labels'][i].item()
                class_name = self.coco_classes[class_id]
                
                detection = {
                    'bbox': box.astype(int),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }
                detections.append(detection)
        
        return detections
    
    def annotate_image(self, image, detections):
        """
        Draw bounding boxes and labels on the image
        """
        annotated = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{class_name}: {confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def log_detections(self, detections):
        """
        Log detected objects to console
        """
        self.detection_count += 1
        self.get_logger().info(f'Detection #{self.detection_count}:')
        
        for detection in detections:
            self.get_logger().info(
                f'  - {detection["class_name"]}: {detection["confidence"]:.3f} '
                f'at [{detection["bbox"][0]}, {detection["bbox"][1]}, '
                f'{detection["bbox"][2]}, {detection["bbox"][3]}]'
            )


def main(args=None):
    """
    Main function to run the object detection node
    """
    rclpy.init(args=args)
    
    try:
        node = ObjectDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
