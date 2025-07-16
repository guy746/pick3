#!/usr/bin/env python3
"""
Post-Pick Monitor Agent for Pick1 System
Monitors green objects crossing the 475mm post-pickup line
"""

import redis
import time
import json
import threading

# Redis connection - using localhost with host networking
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Monitor configuration
MONITOR_LINE = 475  # mm - post-pick monitoring position
CHECK_INTERVAL = 0.05  # 20Hz checking rate
BELT_SPEED = 50  # mm/sec

# Tracking
last_positions = {}  # Track last known positions to detect crossings
monitor_lock = threading.Lock()

def publish_monitor_event(obj_id, obj_data, position, velocity):
    """Publish monitor line crossed event for green objects"""
    event = {
        'event': 'monitor_line_crossed',
        'timestamp': time.time(),
        'data': {
            'object_id': obj_id,
            'type': 'green',
            'lane': int(obj_data.get('lane', 0)),
            'position': position,
            'velocity': velocity,
            'height': float(obj_data.get('height', 32.5)),
            'area': float(obj_data.get('area', 1500))
        }
    }
    
    r.publish('events:monitor', json.dumps(event))
    print(f"[PostPickMonitor] Green object {obj_id} detected at {position:.1f}mm")

def calculate_velocity(obj_id, current_pos, last_pos, time_delta):
    """Calculate object velocity"""
    if time_delta > 0:
        return (current_pos - last_pos) / time_delta
    return BELT_SPEED  # Default to belt speed

def monitor_line():
    """Monitor the 475mm line for green objects"""
    last_check_time = time.time()
    
    while True:
        start_time = time.time()
        current_time = start_time
        time_delta = current_time - last_check_time
        
        # Get objects near monitor line
        # Check slightly before and after the line to catch crossings
        objects_near_line = r.zrangebyscore('objects:active', 
                                          MONITOR_LINE - 10,
                                          MONITOR_LINE + 5,
                                          withscores=True)
        
        with monitor_lock:
            for obj_id, position in objects_near_line:
                # Get object details
                obj_data = r.hgetall(f'object:{obj_id}')
                
                # Only monitor green objects
                if obj_data.get('type') != 'green':
                    continue
                
                # Check if object just crossed the monitor line
                last_pos = last_positions.get(obj_id, 0)
                
                if last_pos < MONITOR_LINE <= position:
                    # Object just crossed the monitor line!
                    velocity = calculate_velocity(obj_id, position, last_pos, time_delta)
                    publish_monitor_event(obj_id, obj_data, position, velocity)
                
                # Update last known position
                last_positions[obj_id] = position
        
        # Clean up positions for objects no longer active
        active_ids = set(obj_id for obj_id, _ in r.zrange('objects:active', 0, -1, withscores=True))
        with monitor_lock:
            last_positions_copy = last_positions.copy()
            for obj_id in last_positions_copy:
                if obj_id not in active_ids:
                    del last_positions[obj_id]
        
        # Update timing
        last_check_time = current_time
        
        # Sleep to maintain check rate
        elapsed = time.time() - start_time
        sleep_time = max(0, CHECK_INTERVAL - elapsed)
        time.sleep(sleep_time)

def status_reporter():
    """Periodically report monitor status"""
    while True:
        time.sleep(30)
        
        with monitor_lock:
            tracking_count = len(last_positions)
        
        print(f"[PostPickMonitor] Status: Tracking {tracking_count} objects")

def main():
    """Main entry point for post-pick monitor agent"""
    print("Post-Pick Monitor Agent starting...")
    print(f"Monitoring line at {MONITOR_LINE}mm for green objects")
    print(f"Check rate: {1/CHECK_INTERVAL:.1f}Hz")
    print("=" * 50)
    
    # Start threads
    threads = [
        threading.Thread(target=monitor_line, daemon=True),
        threading.Thread(target=status_reporter, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nPost-Pick Monitor Agent shutting down...")

if __name__ == "__main__":
    main()