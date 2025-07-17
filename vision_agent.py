#!/usr/bin/env python3
"""
Vision Agent for Pick1 System
Monitors objects crossing the vision detection line and publishes detection events
"""

import redis
import time
import json
import threading
import logging
import signal
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from collections import deque
from datetime import datetime, timedelta

# Redis connection - using localhost with host networking
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r_pubsub = r.pubsub()

# Vision configuration
VISION_LINE = 50  # mm - detection line position
DETECTION_MEMORY_SECONDS = 2  # Remember detected objects for 2 seconds
CHECK_INTERVAL = 0.05  # 20Hz checking rate for accurate detection
REDIS_TIMEOUT = 2.0  # Timeout for Redis operations
MAX_CONSECUTIVE_FAILURES = 5  # Max Redis failures before restart
HEALTH_CHECK_INTERVAL = 10.0  # Health check every 10 seconds

# State management
detected_objects = {}  # Track recently detected objects {obj_id: timestamp}
detection_lock = threading.Lock()
redis_failure_count = 0
last_health_check = time.time()
agent_state = 'starting'  # starting, running, error, recovering

def cleanup_old_detections():
    """Remove old detections from memory after DETECTION_MEMORY_SECONDS"""
    current_time = datetime.now()
    with detection_lock:
        expired_objects = []
        for obj_id, detection_time in detected_objects.items():
            if current_time - detection_time > timedelta(seconds=DETECTION_MEMORY_SECONDS):
                expired_objects.append(obj_id)
        
        for obj_id in expired_objects:
            del detected_objects[obj_id]
            print(f"Removed {obj_id} from detection memory")

def get_object_details(obj_id):
    """Get detailed information about an object"""
    obj_data = r.hgetall(f'object:{obj_id}')
    if not obj_data:
        return None
    
    return {
        'id': obj_id,
        'type': obj_data.get('type', 'unknown'),
        'lane': int(obj_data.get('lane', 0)),
        'position_x': float(obj_data.get('position_x', 0)),
        'area': float(obj_data.get('area', 1500)),
        'height': float(obj_data.get('height', 32.5)),
        'confidence': 1.0  # Perfect detection as requested
    }

def publish_detection(obj_details):
    """Publish detection event to Redis"""
    event = {
        'event': 'object_detected',
        'timestamp': time.time(),
        'data': obj_details
    }
    
    # If it's a green object, add yellow ring for pickup marking
    if obj_details['type'] == 'green' and obj_details['id']:
        r.hset(f"object:{obj_details['id']}", 'has_ring', 'true')
        r.hset(f"object:{obj_details['id']}", 'ring_color', 'yellow')
        print(f"Vision: Added yellow ring to pickable object {obj_details['id']}")
    
    # Publish to events:vision channel
    r.publish('events:vision', json.dumps(event))
    
    # Detailed trace logging
    logging.debug(f"Published detection event: {event}")
    print(f"Vision detected: {obj_details['type']} object {obj_details['id']} at lane {obj_details['lane']}")

def monitor_vision_line():
    """Main monitoring loop for vision detection"""
    last_positions = {}  # Track last known position of each object
    
    while True:
        start_time = time.time()
        
        # Get all active objects
        active_objects = r.zrange('objects:active', 0, -1)
        
        # Check each object for vision line crossing
        for obj_id in active_objects:
            # Get current position
            position_data = r.hget(f'object:{obj_id}', 'position_x')
            if not position_data:
                continue
                
            current_position = float(position_data)
            last_position = last_positions.get(obj_id, 0)
            
            # Check if object crossed the vision line (from before to after)
            if last_position < VISION_LINE <= current_position:
                # Check if we've already detected this object recently
                with detection_lock:
                    if obj_id not in detected_objects:
                        # New detection!
                        obj_details = get_object_details(obj_id)
                        if obj_details:
                            # Only detect GREEN objects
                            if obj_details['type'] == 'green':
                                # Mark as detected
                                detected_objects[obj_id] = datetime.now()
                                # Publish detection event
                                logging.info(f"Detected green object: {obj_id} at {current_position}mm")
                                publish_detection(obj_details)
                                
                                # Debug: Log object type
                                print(f"[VisionAgent] Detected GREEN object {obj_details['id']} at {current_position}mm")
                            else:
                                print(f"[VisionAgent] Ignoring {obj_details['type']} object {obj_id} - only green objects detected")
                        else:
                            print(f"[VisionAgent] Failed to get details for object {obj_id}")
                    else:
                        print(f"[VisionAgent] Skipping already detected object {obj_id}")
            
            # Update last known position
            last_positions[obj_id] = current_position
        
        # Clean up objects that have left the belt
        current_active = set(active_objects)
        removed_objects = set(last_positions.keys()) - current_active
        for obj_id in removed_objects:
            del last_positions[obj_id]
            with detection_lock:
                if obj_id in detected_objects:
                    del detected_objects[obj_id]
        
        # Periodic cleanup of old detections
        cleanup_old_detections()
        
        # Sleep to maintain check rate
        elapsed = time.time() - start_time
        sleep_time = max(0, CHECK_INTERVAL - elapsed)
        time.sleep(sleep_time)

def main():
    """Main entry point for vision agent"""
    print("Vision Agent starting...")
    print(f"Monitoring vision line at {VISION_LINE}mm")
    print(f"Detection memory: {DETECTION_MEMORY_SECONDS} seconds")
    print(f"Check rate: {1/CHECK_INTERVAL:.1f}Hz")
    
    try:
        # Start monitoring
        monitor_vision_line()
    except KeyboardInterrupt:
        print("\nVision Agent shutting down...")
    except Exception as e:
        print(f"Vision Agent error: {e}")
        raise

if __name__ == "__main__":
    main()
