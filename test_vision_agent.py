#!/usr/bin/env python3
"""
Test script for vision agent
Creates test objects and monitors vision detection events
"""

import redis
import json
import time
import threading

# Using host.docker.internal for Docker container networking
r = redis.Redis(host='host.docker.internal', port=6379, decode_responses=True)
r_pubsub = r.pubsub()

def event_listener():
    """Listen for vision detection events"""
    r_pubsub.subscribe(['events:vision'])
    
    print("Listening for vision events...")
    for message in r_pubsub.listen():
        if message['type'] == 'message':
            try:
                event = json.loads(message['data'])
                print(f"\n[VISION EVENT] {event['event']}:")
                print(f"  Object: {event['data']['id']}")
                print(f"  Type: {event['data']['type']}")
                print(f"  Lane: {event['data']['lane']}")
                print(f"  Position: {event['data']['position_x']:.1f}mm")
                print(f"  Area: {event['data']['area']}")
                print(f"  Height: {event['data']['height']}")
                print(f"  Confidence: {event['data']['confidence']}")
            except Exception as e:
                print(f"Error parsing event: {e}")

def create_test_object(obj_id, start_x, lane, obj_type):
    """Create a test object"""
    obj_data = {
        'position_x': str(start_x),
        'lane': str(lane),
        'type': obj_type,
        'status': 'moving',
        'area': '1500',
        'height': '32.5',
        'has_ring': 'false',
        'ring_color': 'yellow'
    }
    r.hset(f'object:{obj_id}', mapping=obj_data)
    r.zadd('objects:active', {obj_id: start_x})
    print(f"Created {obj_type} object {obj_id} at position {start_x}")

def move_object(obj_id, speed=50):
    """Move object along belt"""
    for i in range(20):  # Move for 2 seconds
        position = float(r.hget(f'object:{obj_id}', 'position_x'))
        new_position = position + (speed * 0.1)  # 10Hz update
        
        r.hset(f'object:{obj_id}', 'position_x', str(new_position))
        r.zadd('objects:active', {obj_id: new_position})
        
        print(f"  {obj_id} at {new_position:.1f}mm", end='\r')
        time.sleep(0.1)
    
    # Remove object
    r.zrem('objects:active', obj_id)
    r.delete(f'object:{obj_id}')
    print(f"\n  {obj_id} removed from belt")

def main():
    """Test vision agent with sample objects"""
    print("Vision Agent Test")
    print("-" * 40)
    
    # Start event listener in background
    listener_thread = threading.Thread(target=event_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    # Give listener time to start
    time.sleep(0.5)
    
    # Test 1: Green object crossing vision line
    print("\nTest 1: Green object crossing vision line")
    create_test_object('test_obj_001', 0, 1, 'green')
    move_object('test_obj_001')
    
    time.sleep(1)
    
    # Test 2: Non-green object
    print("\nTest 2: Blue object crossing vision line")
    create_test_object('test_obj_002', 0, 2, 'blue')
    move_object('test_obj_002')
    
    time.sleep(1)
    
    # Test 3: Multiple objects
    print("\nTest 3: Multiple objects simultaneously")
    create_test_object('test_obj_003', 20, 0, 'green')
    create_test_object('test_obj_004', 30, 3, 'red')
    
    # Move them in parallel
    t1 = threading.Thread(target=move_object, args=('test_obj_003',))
    t2 = threading.Thread(target=move_object, args=('test_obj_004',))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()