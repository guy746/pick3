#!/usr/bin/env python3
"""
Test data generator for Pick1 visualization - Fixed version
"""

import redis
import time
import random
import json
import threading

# Connect to Redis
# Using host.docker.internal for Docker container networking
r = redis.Redis(host='host.docker.internal', port=6379, decode_responses=True)

# Global variables for thread communication
pickup_in_progress = False
pickup_lock = threading.Lock()

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
        'belt_speed': '133.33',
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
    belt_speed = 133.33  # mm/sec
    update_interval = 0.1  # 10Hz for smoother movement
    
    while True:
        start_time = time.time()
        
        # Get all active objects
        active_objects = r.zrange('objects:active', 0, -1, withscores=True)
        
        # Use pipeline for better performance
        pipe = r.pipeline()
        
        for obj_id, position in active_objects:
            # Skip if object is being picked up
            with pickup_lock:
                if pickup_in_progress and r.hget(f'object:{obj_id}', 'status') == 'picking':
                    continue
            
            new_position = position + (belt_speed * update_interval)
            
            # Check if green object crosses trigger line (250mm)
            if (r.hget(f'object:{obj_id}', 'type') == 'green' and 
                position < 250 and new_position >= 250):
                # Flash trigger line yellow for green object detection
                r.setex('trigger:flash', 1, 'true')
                print(f"Green object {obj_id} detected at trigger line!")
            
            # Check if green object crosses post-pick monitor line (475mm)
            if (r.hget(f'object:{obj_id}', 'type') == 'green' and 
                position < 475 and new_position >= 475):
                # Flash post-pick monitor line yellow for green object detection
                r.setex('monitor:flash', 1, 'true')
                print(f"Green object {obj_id} detected at post-pick monitor!")
            
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
        # Longer spawn interval to reduce race conditions
        time.sleep(random.uniform(3.0, 5.0))
        
        # Check if pickup is in progress - delay spawning green objects during pickup
        with pickup_lock:
            is_pickup_active = pickup_in_progress
        
        # Create new object
        obj_id = f'obj_{obj_counter:04d}'
        lane = random.randint(0, 3)
        
        # Reduce green object spawning during active pickup to prevent conflicts
        if is_pickup_active:
            # Spawn fewer green objects during pickup operations
            adjusted_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal distribution
            color = random.choices(colors, weights=adjusted_weights)[0]
            print(f"Spawning {color} object {obj_id} (pickup in progress)")
        else:
            color = random.choices(colors, weights=weights)[0]
            print(f"Spawning {color} object {obj_id}")
        
        create_test_object(obj_id, 0, lane, color)
        
        obj_counter += 1

def simulate_pickup():
    """Simulate CNC pickup operations"""
    global pickup_in_progress
    
    while True:
        time.sleep(0.5)  # Check twice per second
        
        # Skip if already picking
        with pickup_lock:
            if pickup_in_progress:
                continue
        
        # Find green objects in pickup zone
        objects_in_zone = r.zrangebyscore('objects:active', 300, 375, withscores=True)
        
        green_objects = []
        for obj_id, pos in objects_in_zone:
            if r.hget(f'object:{obj_id}', 'type') == 'green':
                green_objects.append((obj_id, pos))
        
        if green_objects:
            # Pick the first green object
            obj_id, position = green_objects[0]
            
            with pickup_lock:
                pickup_in_progress = True
            
            print(f"\nStarting pickup of {obj_id} at position {position:.1f}")
            
            # Mark object as being picked
            r.hset(f'object:{obj_id}', 'status', 'picking')
            
            # Move CNC to object position
            r.hset('cnc:0', 'position_x', str(position))
            r.hset('cnc:0', 'status', 'moving')
            time.sleep(0.4)  # Doubled from 0.2
            
            # Start picking
            r.hset('cnc:0', 'status', 'picking')
            r.hset('cnc:0', 'has_object', 'true')
            time.sleep(0.6)  # Doubled from 0.3
            
            # Remove object from belt
            r.zrem('objects:active', obj_id)
            r.delete(f'object:{obj_id}')
            print(f"Picked up {obj_id}")
            
            # Move to drop position
            r.hset('cnc:0', 'position_x', '337.5')
            r.hset('cnc:0', 'status', 'moving')
            time.sleep(0.4)  # Doubled from 0.2
            
            # Drop object
            r.hset('cnc:0', 'status', 'dropping')
            time.sleep(0.4)  # Doubled from 0.2
            r.hset('cnc:0', 'has_object', 'false')
            
            # Flash bin
            r.setex('bin:0:flash', 1, 'true')
            
            # Return to home position
            r.hset('cnc:0', 'status', 'returning_home')
            r.hset('cnc:0', 'position_x', '337.5')
            time.sleep(0.5)  # Time to return to home position
            
            # Set to idle at home position
            r.hset('cnc:0', 'status', 'idle')
            print(f"Completed pickup cycle - CNC returned to home\n")
            
            with pickup_lock:
                pickup_in_progress = False
            
            # Brief cooldown
            time.sleep(1.0)  # Doubled from 0.5

def monitor_performance():
    """Monitor and report performance stats"""
    while True:
        time.sleep(10)
        
        active_count = r.zcard('objects:active')
        cnc_status = r.hget('cnc:0', 'status')
        
        print(f"\n[Monitor] Active objects: {active_count}, CNC: {cnc_status}")

def main():
    """Run test data generator"""
    print("Starting Pick1 test environment (Fixed)...")
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
    
    print("\nStarting animation threads...")
    print("Press Ctrl+C to stop\n")
    
    # Start animation threads
    threads = [
        threading.Thread(target=animate_objects, daemon=True),
        threading.Thread(target=spawn_objects, daemon=True),
        threading.Thread(target=simulate_pickup, daemon=True),
        threading.Thread(target=monitor_performance, daemon=True)
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
