#!/usr/bin/env python3
"""
Test data generator for Pick1 visualization - Event-driven version with Scoring Agent
"""

import redis
import time
import random
import json
import threading
from datetime import datetime

# Connect to Redis - using host.docker.internal for container-to-host communication
# Note: localhost won't work from inside Docker containers
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r_pubsub = r.pubsub()

# Removed pickup globals - now handled by separate CNC agent

# Scoring agent has been moved to scoring_agent.py
# This file now only handles test data generation and simulation

def create_test_object(obj_id, position_x, lane, obj_type='green'):
    """Create a test object in Redis"""
    obj_data = {
        'position_x': str(position_x),
        'lane': str(lane),
        'type': obj_type,
        'status': 'moving',
        'area': '1500',
        'height': '32.5',
        'created_at': str(time.time()),
        'updated_at': str(time.time()),
        'has_ring': 'false',
        'ring_color': 'yellow'
    }
    
    # Set object data
    r.hset(f'object:{obj_id}', mapping=obj_data)
    
    # Add to active objects
    r.zadd('objects:active', {obj_id: position_x})
    
    print(f"Created {obj_type} object {obj_id} at position {position_x}")

def setup_conveyor_config():
    """Set up conveyor configuration"""
    config = {
        'belt_speed': '50',
        'length': '500',
        'lanes': '4',
        'lane_width': '100',
        'vision_zone': '50',
        'trigger_zone': '250',
        'pickup_zone_start': '300',
        'pickup_zone_end': '375',
        'post_pick_zone': '475'
    }
    r.hset('conveyor:config', mapping=config)
    print("Set conveyor configuration")

def setup_cnc():
    """Set up CNC initial state"""
    cnc_data = {
        'position_x': '337.5',
        'position_y': '200',
        'position_z': '100',
        'status': 'idle',
        'has_object': 'false'
    }
    r.hset('cnc:0', mapping=cnc_data)
    print("Set CNC initial state")


def animate_objects():
    """Animate objects moving along belt"""
    belt_speed = 50.0  # mm/sec (slowed down from 133.33)
    update_interval = 0.05  # 20Hz for smoother movement (was 10Hz)
    
    while True:
        start_time = time.time()
        
        # Get all active objects
        active_objects = r.zrange('objects:active', 0, -1, withscores=True)
        
        # Use pipeline for better performance
        pipe = r.pipeline()
        
        for obj_id, position in active_objects:
            # Check if object is being picked up by CNC
            if r.hget(f'object:{obj_id}', 'status') == 'picking':
                continue
            
            new_position = position + (belt_speed * update_interval)
            
            if new_position > 500:
                # Remove object that's past the belt
                pipe.zrem('objects:active', obj_id)
                pipe.delete(f'object:{obj_id}')
                print(f"Removed {obj_id} - past belt end")
            else:
                # Update position
                pipe.hset(f'object:{obj_id}', 'position_x', str(new_position))
                pipe.zadd('objects:active', {obj_id: new_position})
        
        # Execute all updates at once
        pipe.execute()
        
        # Sleep for consistent frame rate
        elapsed = time.time() - start_time
        sleep_time = max(0, update_interval - elapsed)
        time.sleep(sleep_time)

def spawn_objects():
    """Spawn new objects periodically"""
    obj_counter = 1
    colors = ['green', 'blue', 'red', 'yellow', 'orange']
    weights = [0.4, 0.15, 0.15, 0.15, 0.15]  # 40% green
    
    while True:
        # Increased spawn rate for more objects
        time.sleep(random.uniform(1.0, 2.0))
        
        # Create new object
        obj_id = f'obj_{obj_counter:04d}'
        lane = random.randint(0, 3)
        
        # Select color based on weights
        color = random.choices(colors, weights=weights)[0]
        print(f"Spawning {color} object {obj_id}")
        
        create_test_object(obj_id, 0, lane, color)
        
        obj_counter += 1


# Event listening removed - now handled by scoring_agent.py

def monitor_performance():
    """Monitor and report performance stats"""
    while True:
        time.sleep(10)
        
        active_count = r.zcard('objects:active')
        cnc_status = r.hget('cnc:0', 'status')
        tracked_count = len(scoring_agent.tracked_objects)
        
        print(f"\n[Monitor] Active objects: {active_count}, CNC: {cnc_status}, Tracked: {tracked_count}")

def main():
    """Run test data generator with event-driven architecture"""
    print("Starting Pick1 test environment (Event-driven with Scoring Agent)...")
    print("="*50)
    
    # Clear existing data
    r.flushdb()
    
    # Setup static data
    setup_conveyor_config()
    setup_cnc()
    
    # Create initial objects
    create_test_object('obj_0001', 50, 0, 'green')
    create_test_object('obj_0002', 150, 1, 'blue')
    create_test_object('obj_0003', 250, 2, 'green')
    create_test_object('obj_0004', 350, 3, 'red')
    create_test_object('obj_0005', 400, 0, 'orange')
    
    print("\nStarting threads...")
    print("Press Ctrl+C to stop\n")
    
    # Start animation and control threads
    threads = [
        threading.Thread(target=animate_objects, daemon=True),
        threading.Thread(target=spawn_objects, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping test data generator...")

if __name__ == '__main__':
    main()
