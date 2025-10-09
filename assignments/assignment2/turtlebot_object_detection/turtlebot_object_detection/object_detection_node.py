#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch import nn
from torchvision import models
from PIL import Image as PILImage
import numpy as np
import tf2_ros
from .semantic_db import SemanticDB


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()
        # Full COCO class names (91 classes)
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.get_logger().info('Loading Faster R-CNN model...')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info('Model loaded successfully!')

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


        self.embedding_model = models.resnet50(pretrained=True)
        self.embedding_model = nn.Sequential(*list(self.embedding_model.children())[:-1])
        self.embedding_model.to(self.device)
        self.embedding_model.eval()

        self.embed_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.confidence_threshold = 0.5
        self.detection_count = 0
        
        self.subscription = self.create_subscription(Image,'/camera/image_raw',self.image_callback,10)
        
        self.annotated_pub = self.create_publisher(Image, '/camera/annotated', 10)
        
        self.detection_pub = self.create_publisher(Image, '/detection_results', 10)
        
        # Semantic DB - automatically enabled with default settings
        self.db_enabled = True
        try:
            self.get_logger().info('🔗 Attempting to connect to semantic database...')
            self.semantic_db = SemanticDB()
            self.get_logger().info('✅ Semantic DB connected successfully!')
            self.get_logger().info('💾 Database will automatically store detected objects with embeddings')
            self.get_logger().info('🤖 Robot pose tracking enabled for spatial storage')
        except Exception as e:
            self.db_enabled = False
            self.get_logger().warn(f'❌ Semantic DB disabled: {e}')
            self.get_logger().warn('⚠️ Object detections will still work but not be stored in database')
            self.get_logger().warn('💡 Make sure PostgreSQL is running: sudo service postgresql start')

        # TF2 for robot pose
        try:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        except Exception as e:
            self.tf_buffer = None
            self.get_logger().warn(f'TF2 unavailable: {e}')
        
        # Subscribe to AMCL pose for robot position  
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self.amcl_pose_sub = self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.amcl_pose_callback,10)
        self.current_robot_pose = None  # Store latest pose (x, y, theta)

        self.get_logger().info('Object Detection Node initialized')
        self.get_logger().info('Subscribed to /camera/image_raw')
        self.get_logger().info('Publishing annotated images to /camera/annotated')
        
    def amcl_pose_callback(self, msg):
        pose = msg.pose.pose
        x = pose.position.x
        y = pose.position.y
        
        q = pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = np.arctan2(siny_cosp, cosy_cosp)
        
        self.current_robot_pose = (float(x), float(y), float(theta))
        self.get_logger().debug(f'AMCL pose: ({x:.2f}, {y:.2f}, {theta:.2f})')
        
    def image_callback(self, msg):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            detections = self.detect_objects(cv_image)
            
            if detections:
                self.get_logger().info(f"🎯 DETECTED {len(detections)} OBJECTS:")
                for i, det in enumerate(detections):
                    self.get_logger().info(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.3f})")
                    self.get_logger().info(f"     Bbox: [{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]")
                    if 'embedding' in det:
                        self.get_logger().info(f"Has 2048-d embedding")
                    else:
                        self.get_logger().info(f"No embedding extracted")
            
            # Store detections in database if enabled
            if self.db_enabled and detections:
                pose = self._get_robot_pose(msg.header.stamp)
                if pose is not None:
                    robot_x, robot_y, robot_theta = pose
                    region_name = f"location_{int(robot_x)}_{int(robot_y)}"
                    self.get_logger().info(f"🤖 ROBOT POSE: x={robot_x:.2f}, y={robot_y:.2f}, θ={robot_theta:.2f}")
                    self.get_logger().info(f"📍 REGION: {region_name}")
                    
                    self.get_logger().info(f"💾 STORING TO DATABASE:")
                    for det in detections:
                        try:
                            embedding = det.get('embedding')
                            if embedding is None:
                                embedding = [0.0] * 2048
                                self.get_logger().warn(f"⚠️ Using zero vector for {det['class_name']}")
                            obj_id = self.semantic_db.insert_object(det['class_name'], det['class_id'], float(det['confidence']), embedding)
                            self.semantic_db.insert_observation(obj_id, robot_x, robot_y, robot_theta, det['bbox'], region_name)
                            self.get_logger().info(f"✅ Stored '{det['class_name']}' (ID: {obj_id}) at {region_name}")
                        except Exception as e:
                            self.get_logger().warn(f'❌ Failed to store {det["class_name"]}: {e}')
                else:
                    self.get_logger().warn(f"⚠️ Cannot store detections: Robot pose unavailable")
            elif self.db_enabled and not detections:
                self.get_logger().debug("No objects detected above threshold")

            annotated_image = self.annotate_image(cv_image, detections)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def _get_robot_pose(self, stamp):
        # First priority: Use AMCL pose (most reliable for semantic localization)
        if hasattr(self, 'current_robot_pose') and self.current_robot_pose is not None:
            self.get_logger().debug(f'Using AMCL pose: {self.current_robot_pose}')
            return self.current_robot_pose
        
        # Backup: Try TF2 lookup
        if getattr(self, 'tf_buffer', None) is not None:
            frame_combinations = [
                ('map', 'base_link'),
                ('odom', 'base_link'), 
                ('odom', 'base_footprint'),
                ('map', 'base_footprint')
            ]
            
            for parent_frame, child_frame in frame_combinations:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        parent_frame, child_frame, stamp, 
                        timeout=rclpy.duration.Duration(seconds=0.2)
                    )
                    x = transform.transform.translation.x
                    y = transform.transform.translation.y
                    q = transform.transform.rotation
                    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
                    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
                    theta = np.arctan2(siny_cosp, cosy_cosp)
                    self.get_logger().debug(f'Got TF2 pose from {parent_frame}->{child_frame}: ({x:.2f}, {y:.2f}, {theta:.2f})')
                    return float(x), float(y), float(theta)
                except Exception as e:
                    continue
        
        # Last resort: use origin point if all pose methods fail
        self.get_logger().warn(f'All pose methods failed. Using fallback pose.')
        return (0.0, 0.0, 0.0)
    
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
                
                # Safety check for class_id
                if class_id >= len(self.coco_classes):
                    self.get_logger().warn(f'Detected class_id {class_id} is out of range (max: {len(self.coco_classes)-1}). Skipping.')
                    continue
                    
                class_name = self.coco_classes[class_id]
                
                detection = {
                    'bbox': box.astype(int),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }

                # Extract embedding for this detection
                try:
                    embedding = self.extract_embedding(image, detection['bbox'])
                    if embedding is not None:
                        detection['embedding'] = embedding.tolist()
                except Exception as e:
                    # Keep detection even if embedding fails
                    self.get_logger().debug(f'Embedding extraction failed: {e}')
                detections.append(detection)
        
        return detections
    
    def extract_embedding(self, image, bbox):
        """
        Extract a 2048-d feature embedding for the object inside the bounding box.
        Args:
            image: OpenCV BGR image (H, W, 3)
            bbox: [x1, y1, x2, y2] int array
        Returns:
            np.ndarray of shape (2048,) or None if crop invalid
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Convert to RGB PIL image
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = PILImage.fromarray(rgb_crop)

        # Preprocess for embedding
        input_tensor = self.embed_transform(pil_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.embedding_model(input_tensor)  # (1, 2048, 1, 1)
            embedding = feat.squeeze().detach().cpu().numpy()

        # Ensure shape (2048,)
        embedding = embedding.reshape(-1)
        return embedding
    
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
