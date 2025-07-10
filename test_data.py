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
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

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
        'trigger_zone': '300',
        'pickup_zone_start': '375',
        'pickup_zone_end': '425',
        'post_pick_zone': '475'
    }
    r.hset('conveyor:config', mapping=config)
    print("Set conveyor configuration")

def setup_cnc():
    """Set up CNC initial state"""
    cnc_data = {
        'position_x': '400',
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
            
            if new_position > 500:
                # Remove object that's past the belt
                pipe.zrem('objects:active', obj_id)
                pipe.delete(f'object:{obj_id}')
                print(f"Removed {obj_id} - past belt end")
            else:
                # Update position
                pipe.hset(f'object:{obj_id}', 'position_x', str(new_position))
                pipe.zadd('objects:active', {obj_id: new_position})
                
                # Update ring status based on position
                obj_type = r.hget(f'object:{obj_id}', 'type')
                if obj_type == 'green':
                    if 50 < new_position < 300:
                        # Vision zone to trigger - yellow ring
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'yellow')
                    elif 300 <= new_position < 375:
                        # At trigger line - turn red
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'red')
                    elif new_position >= 375:
                        # In pickup zone - keep red
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'red')
        
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
        # Shorter spawn interval for 500mm belt
        time.sleep(random.uniform(1.5, 2.5))
        
        # Create new object
        obj_id = f'obj_{obj_counter:04d}'
        lane = random.randint(0, 3)
        color = random.choices(colors, weights=weights)[0]
        
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
        objects_in_zone = r.zrangebyscore('objects:active', 375, 425, withscores=True)
        
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
            time.sleep(0.2)
            
            # Start picking
            r.hset('cnc:0', 'status', 'picking')
            r.hset('cnc:0', 'has_object', 'true')
            time.sleep(0.3)
            
            # Remove object from belt
            r.zrem('objects:active', obj_id)
            r.delete(f'object:{obj_id}')
            print(f"Picked up {obj_id}")
            
            # Move to drop position
            r.hset('cnc:0', 'position_x', '400')
            r.hset('cnc:0', 'status', 'moving')
            time.sleep(0.2)
            
            # Drop object
            r.hset('cnc:0', 'status', 'dropping')
            time.sleep(0.2)
            r.hset('cnc:0', 'has_object', 'false')
            
            # Flash bin
            r.setex('bin:0:flash', 1, 'true')
            
            # Return to idle
            r.hset('cnc:0', 'status', 'idle')
            print(f"Completed pickup cycle\n")
            
            with pickup_lock:
                pickup_in_progress = False
            
            # Brief cooldown
            time.sleep(0.5)

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
