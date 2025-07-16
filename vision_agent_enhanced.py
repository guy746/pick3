#!/usr/bin/env python3
"""
Enhanced Vision Agent for Pick1 System
Supports both simulation data feed and OAK-D Pro camera with YOLO detection
Detects green objects, identifies lanes, calculates area and height
"""

import redis
import time
import json
import threading
import logging
import cv2
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from base_agent import BaseAgent

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class VisionBackend(ABC):
    """Abstract base class for vision backends"""
    
    @abstractmethod
    def initialize(self):
        """Initialize the vision backend"""
        pass
    
    @abstractmethod
    def get_frame_data(self):
        """Get current frame data (color, depth, timestamp)"""
        pass
    
    @abstractmethod
    def detect_objects(self, color_frame, depth_frame):
        """Detect objects in the frame and return detection data"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass

class SimulationBackend(VisionBackend):
    """Simulation backend using Redis data"""
    
    def __init__(self, redis_client, vision_line=50):
        self.redis = redis_client
        self.vision_line = vision_line
        self.last_positions = {}
        self.detected_objects = {}
        self.detection_lock = threading.Lock()
        
    def initialize(self):
        """Initialize simulation backend"""
        logging.info("Initializing simulation vision backend")
        return True
        
    def get_frame_data(self):
        """Simulate frame data from Redis"""
        # In simulation mode, we don't have actual frames
        return None, None, time.time()
        
    def detect_objects(self, color_frame=None, depth_frame=None):
        """Detect objects using Redis simulation data"""
        detections = []
        
        # Get all active objects from Redis
        active_objects = self.redis.zrange('objects:active', 0, -1)
        
        for obj_id in active_objects:
            # Get current position
            position_data = self.redis.hget(f'object:{obj_id}', 'position_x')
            if not position_data:
                continue
                
            current_position = float(position_data)
            last_position = self.last_positions.get(obj_id, 0)
            
            # Check if object crossed the vision line
            if last_position < self.vision_line <= current_position:
                with self.detection_lock:
                    if obj_id not in self.detected_objects:
                        # Get object details from Redis
                        obj_data = self.redis.hgetall(f'object:{obj_id}')
                        if obj_data:
                            detection = {
                                'id': obj_id,
                                'type': obj_data.get('type', 'unknown'),
                                'lane': int(obj_data.get('lane', 0)),
                                'position_x': current_position,
                                'area': float(obj_data.get('area', 1500)),
                                'height': float(obj_data.get('height', 32.5)),
                                'confidence': 1.0,
                                'color': obj_data.get('color', 'green'),
                                'bbox': [0, 0, 100, 100],  # Simulated bounding box
                                'timestamp': time.time()
                            }
                            detections.append(detection)
                            self.detected_objects[obj_id] = datetime.now()
            
            # Update last known position
            self.last_positions[obj_id] = current_position
            
        return detections
    
    def cleanup(self):
        """Cleanup simulation backend"""
        logging.info("Cleaning up simulation backend")

class OAKDProBackend(VisionBackend):
    """OAK-D Pro camera backend with YOLO detection"""
    
    def __init__(self, model_path="yolo_models/yolov8n.pt", confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.pipeline = None
        self.device = None
        self.yolo_model = None
        
        # Load configuration for zone and lane boundaries
        self.config = self._load_vision_config()
        self.detection_zone = self.config.get('oak_d_pro', {}).get('detection_zone', {})
        self.lane_boundaries = self.config.get('oak_d_pro', {}).get('lane_boundaries', {})
        
        # Detection zone parameters
        self.zone_x_min = self.detection_zone.get('x_min', 0)
        self.zone_x_max = self.detection_zone.get('x_max', 640)
        self.zone_y_min = self.detection_zone.get('y_min', 0) 
        self.zone_y_max = self.detection_zone.get('y_max', 480)
        
        # Camera calibration parameters (to be calibrated for specific setup)
        self.camera_matrix = None
        self.depth_scale = self.config.get('oak_d_pro', {}).get('depth_scale', 1.0)
        
        # Belt calibration parameters
        self.belt_calibration = self.config.get('oak_d_pro', {}).get('belt_calibration', {})
        self.base_height_mm = self.belt_calibration.get('base_height_mm', 1200.0)
        self.height_tolerance = self.belt_calibration.get('height_tolerance', 5.0)
        self.mm_per_pixel_x = self.belt_calibration.get('mm_per_pixel_x', 0.8)
        self.mm_per_pixel_y = self.belt_calibration.get('mm_per_pixel_y', 0.8)
        self.belt_origin = self.belt_calibration.get('belt_origin', {})
        
        # Calibration status
        self.is_calibrated = self.belt_calibration.get('calibrated', False)
        
        # Object tracking for detection line crossing
        self.tracked_objects = {}  # Track objects across frames
        self.detection_line_x = self.detection_zone.get('x_position', 400)
        
        # Belt speed estimation
        self.speed_config = self.belt_calibration.get('belt_speed_estimation', {})
        self.speed_estimation_enabled = self.speed_config.get('enabled', True)
        self.speed_estimation_interval = self.speed_config.get('estimation_interval', 30.0)
        self.last_speed_estimation = 0
        self.current_belt_speed_mms = self.speed_config.get('validation', {}).get('expected_speed_mms', 150.0)
        self.speed_history = deque(maxlen=self.speed_config.get('speed_calculation', {}).get('smoothing_window', 5))
        
        # Optical flow tracking for speed estimation
        self.prev_frame = None
        self.prev_features = None
        self.speed_roi = self.speed_config.get('optical_flow', {}).get('roi_area', {})
        
    def _load_vision_config(self):
        """Load vision configuration"""
        try:
            import yaml
            import os
            config_path = os.path.join(os.path.dirname(__file__), 'vision_config.yaml')
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load vision config: {e}")
            return {}
        
    def _calculate_lane_boundaries(self):
        """Calculate pixel boundaries for each lane (0-3)"""
        # These would be calibrated based on camera setup
        # Example boundaries in pixels (y-coordinates)
        return {
            0: (50, 150),   # Lane 0: pixels 50-150
            1: (150, 250),  # Lane 1: pixels 150-250  
            2: (250, 350),  # Lane 2: pixels 250-350
            3: (350, 450)   # Lane 3: pixels 350-450
        }
        
    def initialize(self):
        """Initialize OAK-D Pro camera and YOLO model"""
        try:
            # Initialize DepthAI pipeline
            import depthai as dai
            
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Create color camera node
            color_cam = self.pipeline.create(dai.node.ColorCamera)
            color_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            color_cam.setInterleaved(False)
            color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            color_cam.setFps(30)
            
            # Create depth camera nodes
            left_cam = self.pipeline.create(dai.node.MonoCamera)
            right_cam = self.pipeline.create(dai.node.MonoCamera)
            depth = self.pipeline.create(dai.node.StereoDepth)
            
            left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            
            # Configure depth
            depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
            depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            depth.setLeftRightCheck(True)
            depth.setSubpixel(False)
            
            # Link cameras to depth
            left_cam.out.link(depth.left)
            right_cam.out.link(depth.right)
            
            # Create outputs
            color_out = self.pipeline.create(dai.node.XLinkOut)
            depth_out = self.pipeline.create(dai.node.XLinkOut)
            color_out.setStreamName("color")
            depth_out.setStreamName("depth")
            
            color_cam.video.link(color_out.input)
            depth.depth.link(depth_out.input)
            
            # Connect to device
            self.device = dai.Device(self.pipeline)
            
            # Initialize YOLO model
            self._initialize_yolo()
            
            logging.info("OAK-D Pro camera initialized successfully")
            return True
            
        except ImportError:
            logging.error("DepthAI not installed. Install with: pip install depthai")
            return False
        except Exception as e:
            logging.error(f"Failed to initialize OAK-D Pro: {e}")
            return False
    
    def _initialize_yolo(self):
        """Initialize YOLO model for object detection (Roboflow)"""
        try:
            # Load configuration
            import yaml
            import os
            
            config_path = os.path.join(os.path.dirname(__file__), 'vision_config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            yolo_config = config.get('yolo', {})
            platform = yolo_config.get('platform', 'roboflow')
            
            if platform == 'roboflow':
                self._initialize_roboflow_model(yolo_config)
            else:
                self._initialize_ultralytics_model(yolo_config)
                
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
    
    def _initialize_roboflow_model(self, config):
        """Initialize Roboflow YOLO model"""
        try:
            from roboflow import Roboflow
            import os
            
            roboflow_config = config.get('roboflow', {})
            
            # Get configuration from environment variables
            api_key = os.getenv('ROBOFLOW_API_KEY')
            workspace_name = os.getenv('ROBOFLOW_WORKSPACE')
            project_name = os.getenv('ROBOFLOW_PROJECT')
            
            if not api_key:
                raise ValueError("Roboflow API key not found. Set ROBOFLOW_API_KEY environment variable")
            if not workspace_name:
                raise ValueError("Roboflow workspace not found. Set ROBOFLOW_WORKSPACE environment variable") 
            if not project_name:
                raise ValueError("Roboflow project not found. Set ROBOFLOW_PROJECT environment variable")
            
            # Initialize Roboflow
            rf = Roboflow(api_key=api_key)
            
            # Get model version from config
            version = roboflow_config.get('version', 1)
            
            project = rf.workspace(workspace_name).project(project_name)
            self.yolo_model = project.version(version).model
            
            logging.info(f"Roboflow model loaded: {workspace_name}/{project_name}/v{version}")
            
        except ImportError:
            logging.error("Roboflow not installed. Install with: pip install roboflow")
            # Fallback to local model if available
            model_path = config.get('model_path')
            if model_path and os.path.exists(model_path):
                self._initialize_ultralytics_model(config)
            else:
                self.yolo_model = None
        except Exception as e:
            logging.error(f"Failed to initialize Roboflow model: {e}")
            self.yolo_model = None
    
    def _initialize_ultralytics_model(self, config):
        """Initialize Ultralytics YOLO model as fallback"""
        try:
            from ultralytics import YOLO
            model_path = config.get('model_path', 'yolov8n.pt')
            self.yolo_model = YOLO(model_path)
            logging.info(f"Ultralytics YOLO model loaded: {model_path}")
        except ImportError:
            logging.error("Ultralytics YOLO not installed. Install with: pip install ultralytics")
            self.yolo_model = None
        except Exception as e:
            logging.error(f"Failed to load Ultralytics model: {e}")
            self.yolo_model = None
    
    def get_frame_data(self):
        """Get current color and depth frames"""
        if not self.device:
            return None, None, time.time()
            
        try:
            # Get color frame
            color_queue = self.device.getOutputQueue("color", maxSize=1, blocking=False)
            color_frame = None
            if color_queue.has():
                color_in = color_queue.get()
                color_frame = color_in.getCvFrame()
            
            # Get depth frame  
            depth_queue = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
            depth_frame = None
            if depth_queue.has():
                depth_in = depth_queue.get()
                depth_frame = depth_in.getFrame()
                
            return color_frame, depth_frame, time.time()
            
        except Exception as e:
            logging.error(f"Error getting frame data: {e}")
            return None, None, time.time()
    
    def detect_objects(self, color_frame, depth_frame):
        """Detect green objects using YOLO and depth data with zone monitoring"""
        detections = []
        
        if color_frame is None or self.yolo_model is None:
            return detections
            
        try:
            # Step 1: Estimate belt speed periodically (for timing predictions)
            current_time = time.time()
            if self.speed_estimation_enabled:
                self.estimate_belt_speed(color_frame, current_time)
            
            # Step 2: Crop frame to detection zone for efficiency
            detection_roi = self._extract_detection_zone(color_frame)
            depth_roi = self._extract_detection_zone(depth_frame) if depth_frame is not None else None
            
            # Step 3: Run YOLO on detection zone only
            if hasattr(self.yolo_model, 'predict'):
                # Roboflow model
                results = self.yolo_model.predict(detection_roi, confidence=int(self.confidence_threshold * 100))
                zone_detections = self._process_roboflow_results(results, detection_roi, depth_roi)
            else:
                # Ultralytics model
                results = self.yolo_model(detection_roi, conf=self.confidence_threshold)
                zone_detections = self._process_ultralytics_results(results, detection_roi, depth_roi)
            
            # Step 4: Track objects and detect line crossing
            detections = self._track_objects_and_detect_crossing(zone_detections)
            
            # Step 5: Add belt speed to detection data for timing calculations
            for detection in detections:
                detection['belt_speed_mms'] = self.current_belt_speed_mms
                detection['estimated_time_to_pickup'] = self._estimate_pickup_time(detection)
                            
        except Exception as e:
            logging.error(f"Error in YOLO detection: {e}")
            
        return detections
    
    def _extract_detection_zone(self, frame):
        """Extract detection zone from full frame"""
        if frame is None:
            return None
            
        try:
            # Crop frame to detection zone
            zone_frame = frame[
                self.zone_y_min:self.zone_y_max,
                self.zone_x_min:self.zone_x_max
            ]
            return zone_frame
        except Exception as e:
            logging.error(f"Error extracting detection zone: {e}")
            return frame
    
    def _track_objects_and_detect_crossing(self, zone_detections):
        """Track objects across frames and detect detection line crossing"""
        crossing_detections = []
        current_time = time.time()
        
        # Convert zone coordinates back to full frame coordinates
        full_frame_detections = []
        for detection in zone_detections:
            # Adjust bounding box coordinates
            bbox = detection['bbox']
            bbox[0] += self.zone_x_min  # x1
            bbox[1] += self.zone_y_min  # y1  
            bbox[2] += self.zone_x_min  # x2
            bbox[3] += self.zone_y_min  # y2
            detection['bbox'] = bbox
            
            # Calculate object center
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            detection['center'] = (center_x, center_y)
            
            full_frame_detections.append(detection)
        
        # Track objects and detect line crossing
        for detection in full_frame_detections:
            object_id = self._get_object_id(detection)
            center_x, center_y = detection['center']
            
            # Check if object crossed detection line
            if object_id in self.tracked_objects:
                # Existing object - check for line crossing
                last_x = self.tracked_objects[object_id]['last_x']
                
                # Object crossed from left to right across detection line
                if last_x < self.detection_line_x <= center_x:
                    # This is a crossing detection!
                    detection['crossing_detected'] = True
                    crossing_detections.append(detection)
                    logging.info(f"Object {object_id} crossed detection line at x={center_x}")
                
                # Update tracking
                self.tracked_objects[object_id].update({
                    'last_x': center_x,
                    'last_y': center_y,
                    'last_seen': current_time,
                    'detection': detection
                })
            else:
                # New object - start tracking
                self.tracked_objects[object_id] = {
                    'last_x': center_x,
                    'last_y': center_y,
                    'last_seen': current_time,
                    'detection': detection,
                    'first_seen': current_time
                }
        
        # Clean up old tracked objects (not seen for 2 seconds)
        self._cleanup_old_tracks(current_time)
        
        return crossing_detections
    
    def _get_object_id(self, detection):
        """Generate consistent object ID for tracking"""
        # Use center position and size to create semi-stable ID
        center_x, center_y = detection['center']
        bbox = detection['bbox']
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Create ID based on quantized position and size
        grid_x = int(center_x // 20)  # 20-pixel grid
        grid_y = int(center_y // 20)
        size_bucket = int(size // 100)  # Size buckets
        
        return f"obj_{grid_x}_{grid_y}_{size_bucket}"
    
    def _cleanup_old_tracks(self, current_time):
        """Remove objects not seen for 2 seconds"""
        timeout = 2.0  # seconds
        expired_ids = []
        
        for obj_id, track in self.tracked_objects.items():
            if current_time - track['last_seen'] > timeout:
                expired_ids.append(obj_id)
        
        for obj_id in expired_ids:
            del self.tracked_objects[obj_id]
    
    def _process_roboflow_results(self, results, color_frame, depth_frame):
        """Process Roboflow detection results"""
        detections = []
        
        try:
            # Roboflow returns predictions as a list of dictionaries
            predictions = results.get('predictions', [])
            
            for i, pred in enumerate(predictions):
                # Extract bounding box
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)
                
                # Convert center coordinates to corner coordinates
                x1 = x - width / 2
                y1 = y - height / 2
                x2 = x + width / 2
                y2 = y + height / 2
                
                confidence = pred.get('confidence', 0)
                class_name = pred.get('class', 'unknown')
                
                # Check if object is green (additional color filtering if needed)
                if self._is_green_object(color_frame, x1, y1, x2, y2):
                    # Calculate object properties
                    lane = self._determine_lane(y1, y2)
                    area = self._calculate_area(x1, y1, x2, y2, depth_frame)
                    height_mm = self._calculate_height(x1, y1, x2, y2, depth_frame)
                    position_x = self._calculate_position_x(x1, x2, depth_frame)
                    
                    detection = {
                        'id': f"oak_{int(time.time() * 1000)}_{len(detections)}",
                        'type': 'detected_object',
                        'lane': lane,
                        'position_x': position_x,
                        'area': area,
                        'height': height_mm,
                        'confidence': float(confidence),
                        'color': 'green',
                        'class_name': class_name,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'timestamp': time.time()
                    }
                    detections.append(detection)
                    
        except Exception as e:
            logging.error(f"Error processing Roboflow results: {e}")
            
        return detections
    
    def _process_ultralytics_results(self, results, color_frame, depth_frame):
        """Process Ultralytics detection results"""
        detections = []
        
        try:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Check if object is green (additional color filtering)
                        if self._is_green_object(color_frame, x1, y1, x2, y2):
                            # Calculate object properties
                            lane = self._determine_lane(y1, y2)
                            area = self._calculate_area(x1, y1, x2, y2, depth_frame)
                            height_mm = self._calculate_height(x1, y1, x2, y2, depth_frame)
                            position_x = self._calculate_position_x(x1, x2, depth_frame)
                            
                            detection = {
                                'id': f"oak_{int(time.time() * 1000)}_{len(detections)}",
                                'type': 'detected_object',
                                'lane': lane,
                                'position_x': position_x,
                                'area': area,
                                'height': height_mm,
                                'confidence': float(confidence),
                                'color': 'green',
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'timestamp': time.time(),
                                'class_id': class_id
                            }
                            detections.append(detection)
                            
        except Exception as e:
            logging.error(f"Error processing Ultralytics results: {e}")
            
        return detections
    
    def _is_green_object(self, frame, x1, y1, x2, y2):
        """Check if object in bounding box is predominantly green"""
        try:
            # Extract ROI
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            if roi.size == 0:
                return False
                
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            
            # Define green color range in HSV
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            
            # Create mask for green pixels
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Calculate percentage of green pixels
            green_ratio = np.sum(green_mask > 0) / green_mask.size
            
            # Object is considered green if >30% of pixels are green
            return green_ratio > 0.3
            
        except Exception as e:
            logging.error(f"Error in green detection: {e}")
            return False
    
    def _determine_lane(self, y1, y2):
        """Determine which lane the object is in based on y-coordinates"""
        center_y = (y1 + y2) / 2
        
        # Check against configured lane boundaries
        for lane_name, boundaries in self.lane_boundaries.items():
            if isinstance(boundaries, dict):
                y_min = boundaries.get('y_min', 0)
                y_max = boundaries.get('y_max', 480)
            else:
                # Legacy format support
                y_min, y_max = boundaries
                
            if y_min <= center_y <= y_max:
                # Extract lane number from lane_name (e.g., "lane_0" -> 0)
                if lane_name.startswith('lane_'):
                    return int(lane_name.split('_')[1])
                return 0
                
        return 0  # Default to lane 0 if not found
    
    def _calculate_area(self, x1, y1, x2, y2, depth_frame):
        """Calculate real-world area of object using depth data"""
        if depth_frame is None:
            # Fallback: estimate area from pixel size
            pixel_area = (x2 - x1) * (y2 - y1)
            return pixel_area * 0.1  # Rough conversion factor
            
        try:
            # Get depth at object center
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            depth_mm = depth_frame[center_y, center_x] * self.depth_scale
            
            if depth_mm == 0:
                return 1500  # Default area if no depth data
                
            # Calculate real-world dimensions
            # These conversion factors depend on camera calibration
            pixel_to_mm_x = depth_mm * 0.001  # Example conversion
            pixel_to_mm_y = depth_mm * 0.001
            
            width_mm = (x2 - x1) * pixel_to_mm_x
            height_mm = (y2 - y1) * pixel_to_mm_y
            
            return width_mm * height_mm
            
        except Exception as e:
            logging.error(f"Error calculating area: {e}")
            return 1500  # Default area
    
    def _calculate_height(self, x1, y1, x2, y2, depth_frame):
        """Calculate object height relative to belt surface using depth data"""
        if depth_frame is None:
            return 32.5  # Default height
            
        try:
            # Sample depth values across the object
            roi_depth = depth_frame[int(y1):int(y2), int(x1):int(x2)]
            
            if roi_depth.size == 0:
                return 32.5
                
            # Filter out invalid depth values (0)
            valid_depths = roi_depth[roi_depth > 0] * self.depth_scale
            
            if len(valid_depths) == 0:
                return 32.5
                
            # Calculate object surface depth (closest point to camera)
            object_surface_depth = np.min(valid_depths)
            
            # Calculate height relative to belt base height
            if self.is_calibrated and self.base_height_mm > 0:
                # Height = belt_base_depth - object_surface_depth
                relative_height = self.base_height_mm - object_surface_depth
                
                # Validate height is reasonable
                if relative_height < 1.0:  # Object below belt surface
                    logging.warning(f"Object appears below belt surface: {relative_height:.1f}mm")
                    return 5.0  # Minimum valid height
                elif relative_height > 200.0:  # Unreasonably tall
                    logging.warning(f"Object height seems too large: {relative_height:.1f}mm")
                    return 50.0  # Cap at reasonable height
                
                logging.debug(f"Object height calculation: belt_depth={self.base_height_mm:.1f}mm, "
                            f"object_depth={object_surface_depth:.1f}mm, height={relative_height:.1f}mm")
                
                return relative_height
            else:
                # Fallback: use depth variation method if not calibrated
                logging.warning("Belt not calibrated, using depth variation for height estimation")
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)
                height = max_depth - min_depth
                
                # Clamp to reasonable values
                return max(10.0, min(100.0, height))
            
        except Exception as e:
            logging.error(f"Error calculating height: {e}")
            return 32.5
    
    def _calculate_position_x(self, x1, x2, depth_frame):
        """Calculate X position on conveyor belt"""
        # This would be calibrated based on camera setup
        # For now, return a position based on image center
        center_x = (x1 + x2) / 2
        
        # Example: convert pixel position to mm on belt
        # This needs proper calibration
        belt_position = center_x * 0.5  # Example conversion factor
        
        return belt_position
    
    def calibrate_belt_height(self, depth_frame=None):
        """Calibrate the base height of the conveyor belt"""
        if depth_frame is None:
            # Get current depth frame
            _, depth_frame, _ = self.get_frame_data()
            if depth_frame is None:
                logging.error("Cannot calibrate: No depth frame available")
                return False
        
        try:
            # Get calibration configuration
            calib_config = self.config.get('calibration', {}).get('belt_height_calibration', {})
            sample_area = calib_config.get('sample_area', {})
            
            # Default sample area (center area of belt)
            x_min = sample_area.get('x_min', 300)
            x_max = sample_area.get('x_max', 500)
            y_min = sample_area.get('y_min', 200)
            y_max = sample_area.get('y_max', 400)
            
            # Extract sample region
            sample_region = depth_frame[y_min:y_max, x_min:x_max]
            
            # Filter out invalid depth values
            valid_depths = sample_region[sample_region > 0] * self.depth_scale
            
            if len(valid_depths) < 10:
                logging.error("Insufficient valid depth measurements for calibration")
                return False
            
            # Remove outliers
            outlier_threshold = calib_config.get('outlier_threshold', 2.0)
            mean_depth = np.mean(valid_depths)
            std_depth = np.std(valid_depths)
            
            # Keep only depths within threshold standard deviations
            mask = np.abs(valid_depths - mean_depth) <= (outlier_threshold * std_depth)
            filtered_depths = valid_depths[mask]
            
            if len(filtered_depths) < 5:
                logging.error("Too few measurements after outlier removal")
                return False
            
            # Calculate base height using specified method
            height_method = calib_config.get('height_method', 'median')
            if height_method == 'median':
                self.base_height_mm = float(np.median(filtered_depths))
            elif height_method == 'mean':
                self.base_height_mm = float(np.mean(filtered_depths))
            else:  # mode or fallback to median
                self.base_height_mm = float(np.median(filtered_depths))
            
            # Calculate measurement accuracy
            measurement_std = np.std(filtered_depths)
            
            logging.info(f"Belt height calibrated: {self.base_height_mm:.1f}mm (±{measurement_std:.1f}mm)")
            
            # Update calibration status
            self._update_calibration_status('height', self.base_height_mm, measurement_std)
            
            return True
            
        except Exception as e:
            logging.error(f"Belt height calibration failed: {e}")
            return False
    
    def calibrate_belt_coordinates(self, reference_objects=None):
        """Calibrate pixel-to-world coordinate transformation using reference objects"""
        if not reference_objects:
            # Use configuration reference objects
            calib_config = self.config.get('calibration', {}).get('coordinate_calibration', {})
            reference_objects = calib_config.get('reference_objects', [])
        
        if not reference_objects:
            logging.error("No reference objects provided for coordinate calibration")
            return False
        
        try:
            # Get current frame for object detection
            color_frame, depth_frame, _ = self.get_frame_data()
            if color_frame is None:
                logging.error("Cannot calibrate: No camera frame available")
                return False
            
            pixel_points = []
            world_points = []
            
            for ref_obj in reference_objects:
                name = ref_obj.get('name', 'unknown')
                known_pos = ref_obj.get('known_position_mm', [0, 0])
                known_size = ref_obj.get('known_size_mm', [50, 50, 25])
                expected_pixel = ref_obj.get('pixel_position', [400, 250])
                
                # Detect the reference object in the image
                detected_pixel = self._detect_reference_object(color_frame, name, expected_pixel)
                
                if detected_pixel:
                    pixel_points.append(detected_pixel)
                    world_points.append(known_pos)
                    logging.info(f"Found reference object '{name}' at pixel {detected_pixel}, "
                               f"expected world position {known_pos}mm")
                else:
                    logging.warning(f"Could not detect reference object '{name}'")
            
            if len(pixel_points) < 2:
                logging.error("Need at least 2 reference objects for coordinate calibration")
                return False
            
            # Calculate transformation parameters
            pixel_points = np.array(pixel_points, dtype=np.float32)
            world_points = np.array(world_points, dtype=np.float32)
            
            # Simple affine transformation calculation
            # For now, use linear scaling from first two points
            if len(pixel_points) >= 2:
                # Calculate scaling factors
                pixel_diff = pixel_points[1] - pixel_points[0]
                world_diff = world_points[1] - world_points[0]
                
                if abs(pixel_diff[0]) > 1 and abs(pixel_diff[1]) > 1:
                    self.mm_per_pixel_x = abs(world_diff[0] / pixel_diff[0])
                    self.mm_per_pixel_y = abs(world_diff[1] / pixel_diff[1])
                    
                    # Update belt origin
                    self.belt_origin = {
                        'pixel_x': float(pixel_points[0][0]),
                        'pixel_y': float(pixel_points[0][1]),
                        'world_x_mm': float(world_points[0][0]),
                        'world_y_mm': float(world_points[0][1])
                    }
                    
                    logging.info(f"Coordinate calibration successful:")
                    logging.info(f"  X scale: {self.mm_per_pixel_x:.3f} mm/pixel")
                    logging.info(f"  Y scale: {self.mm_per_pixel_y:.3f} mm/pixel")
                    logging.info(f"  Origin: pixel({self.belt_origin['pixel_x']:.0f}, {self.belt_origin['pixel_y']:.0f}) = "
                               f"world({self.belt_origin['world_x_mm']:.0f}, {self.belt_origin['world_y_mm']:.0f})mm")
                    
                    # Validate calibration accuracy
                    self._validate_coordinate_calibration(pixel_points, world_points)
                    
                    return True
            
            logging.error("Could not calculate coordinate transformation")
            return False
            
        except Exception as e:
            logging.error(f"Coordinate calibration failed: {e}")
            return False
    
    def _detect_reference_object(self, frame, object_name, expected_pixel):
        """Detect a reference calibration object in the frame"""
        # This is a simplified detection - in practice you'd use more sophisticated methods
        # For now, return the expected pixel position (manual calibration)
        # In a real implementation, you might:
        # 1. Use template matching
        # 2. Use feature detection (SIFT, ORB)
        # 3. Use color-based detection for colored calibration blocks
        # 4. Use QR codes or ArUco markers
        
        logging.info(f"Using expected position for reference object '{object_name}': {expected_pixel}")
        return expected_pixel
    
    def _validate_coordinate_calibration(self, pixel_points, world_points):
        """Validate the accuracy of coordinate calibration"""
        try:
            errors = []
            for i, (pixel, world) in enumerate(zip(pixel_points, world_points)):
                # Convert pixel back to world coordinates using calibration
                calculated_world = self._pixel_to_world_coordinates(pixel[0], pixel[1])
                error = np.sqrt((calculated_world[0] - world[0])**2 + (calculated_world[1] - world[1])**2)
                errors.append(error)
                logging.info(f"Point {i}: Expected {world}mm, Calculated {calculated_world}, Error: {error:.1f}mm")
            
            rms_error = np.sqrt(np.mean(np.array(errors)**2))
            logging.info(f"Coordinate calibration RMS error: {rms_error:.1f}mm")
            
            # Update calibration status
            self._update_calibration_status('coordinates', rms_error, rms_error)
            
            return rms_error
            
        except Exception as e:
            logging.error(f"Calibration validation failed: {e}")
            return float('inf')
    
    def _pixel_to_world_coordinates(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates using calibration"""
        # Offset from origin
        dx_pixels = pixel_x - self.belt_origin.get('pixel_x', 400)
        dy_pixels = pixel_y - self.belt_origin.get('pixel_y', 250)
        
        # Convert to world coordinates
        world_x = self.belt_origin.get('world_x_mm', 0) + (dx_pixels * self.mm_per_pixel_x)
        world_y = self.belt_origin.get('world_y_mm', 0) + (dy_pixels * self.mm_per_pixel_y)
        
        return [world_x, world_y]
    
    def _update_calibration_status(self, calibration_type, value, accuracy):
        """Update calibration status and save to config"""
        try:
            from datetime import datetime
            
            # Update in-memory calibration status
            if calibration_type == 'height':
                self.base_height_mm = value
                logging.info(f"Base height updated: {value:.1f}mm")
            elif calibration_type == 'coordinates':
                logging.info(f"Coordinate calibration accuracy: {accuracy:.1f}mm RMS")
            
            # Mark as calibrated if both height and coordinates are calibrated
            # For now, we'll consider it calibrated after any successful calibration
            self.is_calibrated = True
            
            # You could save updated calibration to config file here if needed
            # This would require updating the YAML file with new values
            
        except Exception as e:
            logging.error(f"Failed to update calibration status: {e}")
    
    def estimate_belt_speed(self, color_frame, timestamp):
        """Estimate belt speed using optical flow analysis"""
        if not self.speed_estimation_enabled or color_frame is None:
            return None
            
        try:
            # Check if it's time for speed estimation
            if timestamp - self.last_speed_estimation < self.speed_estimation_interval:
                return self.current_belt_speed_mms
            
            # Convert to grayscale for optical flow
            gray = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
            
            # Extract ROI for speed measurement
            roi_x_min = self.speed_roi.get('x_min', 200)
            roi_x_max = self.speed_roi.get('x_max', 600) 
            roi_y_min = self.speed_roi.get('y_min', 100)
            roi_y_max = self.speed_roi.get('y_max', 400)
            
            roi_gray = gray[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
            
            speed_estimate = None
            
            if self.prev_frame is not None:
                # Calculate time difference
                time_diff = timestamp - self.last_speed_estimation
                
                # Get optical flow configuration
                flow_config = self.speed_config.get('optical_flow', {})
                method = flow_config.get('flow_method', 'lucas_kanade')
                
                if method == 'lucas_kanade' and self.prev_features is not None:
                    speed_estimate = self._estimate_speed_lucas_kanade(roi_gray, time_diff)
                elif method == 'farneback':
                    speed_estimate = self._estimate_speed_farneback(roi_gray, time_diff)
                
                if speed_estimate is not None:
                    # Validate and smooth the speed estimate
                    speed_estimate = self._validate_and_smooth_speed(speed_estimate)
                    
                    if speed_estimate is not None:
                        self.current_belt_speed_mms = speed_estimate
                        self._publish_speed_update(speed_estimate, timestamp)
                        logging.info(f"Belt speed estimated: {speed_estimate:.1f} mm/s")
                        
            # Update previous frame and features for next estimation
            self.prev_frame = roi_gray.copy()
            self._detect_features_for_tracking(roi_gray)
            self.last_speed_estimation = timestamp
            
            return self.current_belt_speed_mms
            
        except Exception as e:
            logging.error(f"Belt speed estimation failed: {e}")
            return self.current_belt_speed_mms
    
    def _estimate_speed_lucas_kanade(self, current_roi, time_diff):
        """Estimate speed using Lucas-Kanade optical flow"""
        try:
            if self.prev_features is None or len(self.prev_features) == 0:
                return None
                
            # Lucas-Kanade parameters
            lk_params = dict(winSize=(15, 15),
                            maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            # Calculate optical flow
            next_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_roi, self.prev_features, None, **lk_params)
            
            # Select good points
            good_new = next_features[status == 1]
            good_old = self.prev_features[status == 1]
            
            if len(good_new) < 5:  # Need at least 5 good points
                return None
                
            # Calculate displacement vectors
            displacements = good_new - good_old
            
            # Filter out small movements (noise)
            speed_calc_config = self.speed_config.get('speed_calculation', {})
            pixel_threshold = speed_calc_config.get('pixel_distance_threshold', 5)
            
            significant_displacements = []
            for disp in displacements:
                if np.linalg.norm(disp) > pixel_threshold:
                    # Project displacement onto belt direction (assume X direction)
                    belt_displacement = disp[0]  # X component
                    significant_displacements.append(belt_displacement)
            
            if len(significant_displacements) < 3:
                return None
                
            # Calculate median displacement in pixels
            median_displacement_pixels = np.median(significant_displacements)
            
            # Convert to mm/s using calibration
            pixels_per_second = median_displacement_pixels / time_diff
            speed_mms = pixels_per_second * self.mm_per_pixel_x
            
            return abs(speed_mms)  # Return absolute speed
            
        except Exception as e:
            logging.error(f"Lucas-Kanade speed estimation failed: {e}")
            return None
    
    def _estimate_speed_farneback(self, current_roi, time_diff):
        """Estimate speed using Farneback dense optical flow"""
        try:
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowPyrLK(self.prev_frame, current_roi, 
                                          pyr_scale=0.5, levels=3, winsize=15,
                                          iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            
            # Calculate magnitude and angle of flow vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Filter by magnitude threshold
            speed_calc_config = self.speed_config.get('speed_calculation', {})
            pixel_threshold = speed_calc_config.get('pixel_distance_threshold', 5)
            
            # Get significant flow vectors
            significant_flow = magnitude[magnitude > pixel_threshold]
            
            if len(significant_flow) < 10:
                return None
                
            # Calculate median flow magnitude in pixels per frame
            median_flow_pixels = np.median(significant_flow)
            
            # Convert to mm/s
            pixels_per_second = median_flow_pixels / time_diff
            speed_mms = pixels_per_second * self.mm_per_pixel_x
            
            return speed_mms
            
        except Exception as e:
            logging.error(f"Farneback speed estimation failed: {e}")
            return None
    
    def _detect_features_for_tracking(self, roi_gray):
        """Detect features for optical flow tracking"""
        try:
            flow_config = self.speed_config.get('optical_flow', {})
            
            # Feature detection parameters
            feature_params = dict(
                maxCorners=flow_config.get('max_features', 100),
                qualityLevel=flow_config.get('quality_level', 0.01),
                minDistance=flow_config.get('min_distance', 10),
                blockSize=7
            )
            
            # Detect corner features
            features = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
            self.prev_features = features
            
        except Exception as e:
            logging.error(f"Feature detection failed: {e}")
            self.prev_features = None
    
    def _validate_and_smooth_speed(self, raw_speed):
        """Validate speed measurement and apply smoothing"""
        try:
            speed_calc_config = self.speed_config.get('speed_calculation', {})
            validation_config = self.speed_config.get('validation', {})
            
            # Check speed bounds
            min_speed = speed_calc_config.get('min_speed_mms', 10)
            max_speed = speed_calc_config.get('max_speed_mms', 500)
            
            if raw_speed < min_speed or raw_speed > max_speed:
                logging.warning(f"Speed {raw_speed:.1f} mm/s outside valid range [{min_speed}-{max_speed}]")
                return None
            
            # Add to history
            self.speed_history.append(raw_speed)
            
            # Apply smoothing
            if len(self.speed_history) >= 2:
                smoothed_speed = np.median(self.speed_history)
                
                # Check for sudden speed changes
                if len(self.speed_history) > 1:
                    speed_change_percent = abs(smoothed_speed - self.current_belt_speed_mms) / self.current_belt_speed_mms * 100
                    alert_threshold = validation_config.get('speed_change_alert_percent', 10)
                    
                    if speed_change_percent > alert_threshold:
                        logging.warning(f"Sudden belt speed change detected: {self.current_belt_speed_mms:.1f} -> {smoothed_speed:.1f} mm/s ({speed_change_percent:.1f}%)")
                        self._publish_speed_alert(speed_change_percent, smoothed_speed)
                
                return smoothed_speed
            else:
                return raw_speed
                
        except Exception as e:
            logging.error(f"Speed validation failed: {e}")
            return raw_speed
    
    def _publish_speed_update(self, speed_mms, timestamp):
        """Publish belt speed update event"""
        try:
            # This would be called through the vision agent's event system
            # For now, just log the update
            validation_config = self.speed_config.get('validation', {})
            expected_speed = validation_config.get('expected_speed_mms', 150)
            tolerance_percent = validation_config.get('speed_tolerance_percent', 20)
            
            speed_deviation_percent = abs(speed_mms - expected_speed) / expected_speed * 100
            
            status = "normal"
            if speed_deviation_percent > tolerance_percent:
                status = "warning"
                logging.warning(f"Belt speed outside tolerance: {speed_mms:.1f} mm/s "
                              f"(expected {expected_speed:.1f} ±{tolerance_percent}%)")
            
            # Speed event data
            speed_event = {
                'event': 'belt_speed_update',
                'timestamp': timestamp,
                'data': {
                    'speed_mms': speed_mms,
                    'expected_speed_mms': expected_speed,
                    'deviation_percent': speed_deviation_percent,
                    'status': status,
                    'measurement_method': self.speed_config.get('measurement_method', 'optical_flow')
                }
            }
            
            # Store for later publishing through vision agent
            self._speed_event = speed_event
            
        except Exception as e:
            logging.error(f"Failed to publish speed update: {e}")
    
    def _publish_speed_alert(self, change_percent, new_speed):
        """Publish speed change alert"""
        try:
            alert_event = {
                'event': 'belt_speed_alert',
                'timestamp': time.time(),
                'data': {
                    'alert_type': 'sudden_change',
                    'old_speed_mms': self.current_belt_speed_mms,
                    'new_speed_mms': new_speed,
                    'change_percent': change_percent,
                    'severity': 'warning' if change_percent > 20 else 'info'
                }
            }
            
            # Store for later publishing through vision agent
            self._speed_alert = alert_event
            
        except Exception as e:
            logging.error(f"Failed to publish speed alert: {e}")
    
    def _estimate_pickup_time(self, detection):
        """Estimate time until object reaches pickup position"""
        try:
            # Get object center position
            bbox = detection.get('bbox', [0, 0, 0, 0])
            center_x = (bbox[0] + bbox[2]) / 2
            
            # Distance from detection line to pickup position (estimated)
            # This would be calibrated based on actual system layout
            pickup_position_x = 600  # Example: pickup happens at x=600 pixels
            distance_pixels = pickup_position_x - center_x
            
            if distance_pixels <= 0:
                return 0.0  # Object already past pickup point
            
            # Convert distance to mm
            distance_mm = distance_pixels * self.mm_per_pixel_x
            
            # Calculate time based on belt speed
            if self.current_belt_speed_mms > 0:
                time_to_pickup = distance_mm / self.current_belt_speed_mms  # seconds
                return max(0.0, time_to_pickup)
            else:
                return 10.0  # Default estimate if speed unknown
                
        except Exception as e:
            logging.error(f"Error estimating pickup time: {e}")
            return 10.0
    
    def get_current_belt_speed(self):
        """Get current belt speed estimate"""
        return self.current_belt_speed_mms
    
    def cleanup(self):
        """Cleanup OAK-D Pro resources"""
        if self.device:
            self.device.close()
            logging.info("OAK-D Pro device closed")

class EnhancedVisionAgent(BaseAgent):
    """Enhanced vision agent supporting both simulation and OAK-D Pro camera"""
    
    def __init__(self, config_file="vision_config.yaml"):
        super().__init__(
            name="vision_enhanced",
            subscribe_channels=["events:system"],
            publish_channel="events:vision"
        )
        
        # Load configuration
        self.config = self._load_config(config_file)
        vision_config = self.config.get('vision', {})
        
        self.mode = vision_config.get('mode', 'simulation')
        self.vision_backend = None
        self.detection_memory_seconds = vision_config.get('detection_memory_seconds', 2)
        self.check_interval = vision_config.get('check_interval', 0.05)  # 20Hz
        self.vision_line = vision_config.get('vision_line', 50)
        
        # Detection tracking
        self.recent_detections = {}
        self.detection_lock = threading.Lock()
        
        # Performance metrics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        # Initialize backend
        self._initialize_backend()
        
    def _load_config(self, config_file):
        """Load vision configuration from YAML file"""
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), config_file)
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded vision config from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
        
    def _initialize_backend(self):
        """Initialize the appropriate vision backend"""
        if self.mode == "simulation":
            self.vision_backend = SimulationBackend(self.redis, self.vision_line)
        elif self.mode == "oak_d_pro":
            yolo_config = self.config.get('yolo', {})
            model_path = yolo_config.get('model_path', 'yolo_models/yolov8n.pt')
            confidence = yolo_config.get('confidence_threshold', 0.5)
            self.vision_backend = OAKDProBackend(model_path, confidence)
        else:
            raise ValueError(f"Unknown vision mode: {self.mode}")
            
        success = self.vision_backend.initialize()
        if not success:
            raise RuntimeError(f"Failed to initialize {self.mode} backend")
            
        self.logger.info(f"Vision agent initialized in {self.mode} mode")
    
    def switch_mode(self, new_mode, **kwargs):
        """Switch between simulation and camera modes"""
        if new_mode == self.mode:
            self.logger.info(f"Already in {new_mode} mode")
            return
            
        self.logger.info(f"Switching from {self.mode} to {new_mode} mode")
        
        # Cleanup current backend
        if self.vision_backend:
            self.vision_backend.cleanup()
            
        # Initialize new backend
        self.mode = new_mode
        self._initialize_backend(**kwargs)
        
        # Announce mode switch
        self._announce_mode_switch(new_mode)
    
    def _announce_mode_switch(self, new_mode):
        """Announce mode switch to other agents"""
        event = {
            'event': 'vision_mode_changed',
            'timestamp': time.time(),
            'data': {
                'old_mode': getattr(self, '_previous_mode', 'unknown'),
                'new_mode': new_mode,
                'agent': self.name
            }
        }
        self.publish('mode_change', event)
        self.logger.info(f"Vision mode switched to: {new_mode}")
    
    def handle_message(self, channel, message):
        """Handle incoming messages"""
        event_type = message.get('event')
        data = message.get('data', {})
        
        if event_type == 'switch_vision_mode':
            new_mode = data.get('mode')
            if new_mode:
                self.switch_mode(new_mode, **data)
        elif event_type == 'calibrate_camera':
            self._calibrate_camera()
        elif event_type == 'calibrate_belt_height':
            self._calibrate_belt_height()
        elif event_type == 'calibrate_belt_coordinates':
            self._calibrate_belt_coordinates(data.get('reference_objects'))
        elif event_type == 'get_vision_status':
            self._publish_status()
        elif event_type == 'get_belt_speed':
            self._publish_belt_speed()
        elif event_type == 'set_expected_belt_speed':
            self._set_expected_belt_speed(data.get('speed_mms', 150.0))
    
    def _calibrate_camera(self):
        """Perform camera calibration (placeholder)"""
        if self.mode == "oak_d_pro" and isinstance(self.vision_backend, OAKDProBackend):
            self.logger.info("Starting camera calibration...")
            # Implement calibration procedure here
            # This would involve detecting calibration patterns
            # and calculating camera matrix and lane boundaries
            pass
    
    def _calibrate_belt_height(self):
        """Calibrate belt base height"""
        if self.mode == "oak_d_pro" and isinstance(self.vision_backend, OAKDProBackend):
            self.logger.info("Starting belt height calibration...")
            success = self.vision_backend.calibrate_belt_height()
            
            status_event = {
                'event': 'calibration_result',
                'timestamp': time.time(),
                'data': {
                    'calibration_type': 'belt_height',
                    'success': success,
                    'base_height_mm': self.vision_backend.base_height_mm if success else None
                }
            }
            self.publish('calibration', status_event)
            
            if success:
                self.logger.info(f"Belt height calibration successful: {self.vision_backend.base_height_mm:.1f}mm")
            else:
                self.logger.error("Belt height calibration failed")
        else:
            self.logger.warning("Belt height calibration only available in OAK-D Pro mode")
    
    def _calibrate_belt_coordinates(self, reference_objects=None):
        """Calibrate belt coordinate system"""
        if self.mode == "oak_d_pro" and isinstance(self.vision_backend, OAKDProBackend):
            self.logger.info("Starting belt coordinate calibration...")
            success = self.vision_backend.calibrate_belt_coordinates(reference_objects)
            
            status_event = {
                'event': 'calibration_result', 
                'timestamp': time.time(),
                'data': {
                    'calibration_type': 'belt_coordinates',
                    'success': success,
                    'mm_per_pixel_x': self.vision_backend.mm_per_pixel_x if success else None,
                    'mm_per_pixel_y': self.vision_backend.mm_per_pixel_y if success else None,
                    'belt_origin': self.vision_backend.belt_origin if success else None
                }
            }
            self.publish('calibration', status_event)
            
            if success:
                self.logger.info("Belt coordinate calibration successful")
            else:
                self.logger.error("Belt coordinate calibration failed")
        else:
            self.logger.warning("Belt coordinate calibration only available in OAK-D Pro mode")
    
    def _publish_belt_speed(self):
        """Publish current belt speed"""
        if self.mode == "oak_d_pro" and isinstance(self.vision_backend, OAKDProBackend):
            speed = self.vision_backend.get_current_belt_speed()
            
            speed_event = {
                'event': 'belt_speed_status',
                'timestamp': time.time(),
                'data': {
                    'current_speed_mms': speed,
                    'estimation_enabled': self.vision_backend.speed_estimation_enabled,
                    'last_estimation': self.vision_backend.last_speed_estimation
                }
            }
            self.publish('belt_speed', speed_event)
        else:
            self.logger.warning("Belt speed monitoring only available in OAK-D Pro mode")
    
    def _set_expected_belt_speed(self, speed_mms):
        """Set expected belt speed for validation"""
        if self.mode == "oak_d_pro" and isinstance(self.vision_backend, OAKDProBackend):
            self.vision_backend.speed_config.setdefault('validation', {})['expected_speed_mms'] = speed_mms
            self.logger.info(f"Expected belt speed set to {speed_mms:.1f} mm/s")
            
            # Publish confirmation
            event = {
                'event': 'belt_speed_config_updated',
                'timestamp': time.time(),
                'data': {
                    'expected_speed_mms': speed_mms
                }
            }
            self.publish('belt_speed', event)
        else:
            self.logger.warning("Belt speed configuration only available in OAK-D Pro mode")
    
    def _publish_status(self):
        """Publish current vision system status"""
        uptime = time.time() - self.start_time
        fps = self.frame_count / uptime if uptime > 0 else 0
        
        status = {
            'event': 'vision_status',
            'timestamp': time.time(),
            'data': {
                'mode': self.mode,
                'uptime': uptime,
                'frame_count': self.frame_count,
                'detection_count': self.detection_count,
                'fps': fps,
                'backend_type': type(self.vision_backend).__name__
            }
        }
        self.publish('status', status)
    
    def _cleanup_old_detections(self):
        """Remove old detections from memory"""
        current_time = datetime.now()
        with self.detection_lock:
            expired_objects = []
            for obj_id, detection_time in self.recent_detections.items():
                if current_time - detection_time > timedelta(seconds=self.detection_memory_seconds):
                    expired_objects.append(obj_id)
            
            for obj_id in expired_objects:
                del self.recent_detections[obj_id]
    
    def _publish_detection(self, detection):
        """Publish detection event"""
        event = {
            'event': 'object_detected',
            'timestamp': time.time(),
            'data': detection
        }
        
        self.publish('detection', event)
        
        # Also publish to legacy channel for compatibility
        self.redis.publish('events:vision', json.dumps(event))
        
        self.logger.info(f"Detected: {detection['type']} object {detection['id']} "
                        f"in lane {detection['lane']} (confidence: {detection['confidence']:.2f})")
    
    def _process_detections(self, detections):
        """Process and publish new detections"""
        for detection in detections:
            obj_id = detection['id']
            
            with self.detection_lock:
                if obj_id not in self.recent_detections:
                    # New detection
                    self.recent_detections[obj_id] = datetime.now()
                    self._publish_detection(detection)
                    self.detection_count += 1
    
    def run_vision_loop(self):
        """Main vision processing loop"""
        self.logger.info(f"Starting vision loop in {self.mode} mode")
        
        while True:
            try:
                start_time = time.time()
                
                # Get frame data
                color_frame, depth_frame, timestamp = self.vision_backend.get_frame_data()
                
                # Detect objects
                detections = self.vision_backend.detect_objects(color_frame, depth_frame)
                
                # Process detections
                if detections:
                    self._process_detections(detections)
                
                # Update metrics
                self.frame_count += 1
                
                # Cleanup old detections
                if self.frame_count % 100 == 0:  # Every 100 frames
                    self._cleanup_old_detections()
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in vision loop: {e}")
                time.sleep(1.0)  # Wait before retrying
    
    def run(self):
        """Run the vision agent"""
        # Start vision processing in separate thread
        vision_thread = threading.Thread(target=self.run_vision_loop, daemon=True)
        vision_thread.start()
        
        # Run base agent message loop
        super().run()
    
    def cleanup(self):
        """Cleanup vision agent resources"""
        if self.vision_backend:
            self.vision_backend.cleanup()
        self.logger.info("Vision agent cleaned up")

def main():
    """Main entry point for enhanced vision agent"""
    import sys
    
    # Parse command line arguments
    mode = "simulation"
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    print(f"Enhanced Vision Agent starting in {mode} mode...")
    
    try:
        agent = EnhancedVisionAgent(mode=mode)
        agent.run()
    except KeyboardInterrupt:
        print("\nVision Agent shutting down...")
    except Exception as e:
        print(f"Vision Agent error: {e}")
        raise
    finally:
        if 'agent' in locals():
            agent.cleanup()

if __name__ == "__main__":
    main()