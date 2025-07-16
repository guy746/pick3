#!/usr/bin/env python3
"""
Slow down conveyor belt by 25% to improve pickup success
"""

def update_app_py():
    """Update app.py with slower belt speed"""
    print("Updating app.py...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Replace belt speed from 133.33 to 100
    content = content.replace("'belt_speed': 133.33,", "'belt_speed': 100,")
    content = content.replace("133.33", "100")  # Catch any other references
    
    # Update comment about travel time
    content = content.replace("500mm in 3.75 seconds", "500mm in 5 seconds")
    
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("✓ app.py updated - belt speed now 100 mm/sec")

def create_updated_simple_test():
    """Create test_data.py with slower belt speed"""
    print("Creating updated simple_pickup_test.py...")
    
    content = '''#!/usr/bin/env python3
"""
Simple pickup test with 25% slower belt speed
Belt now moves at 100 mm/sec (was 133.33 mm/sec)
"""

import redis
import time
import random
import threading

# Using host.docker.internal for Docker container networking
r = redis.Redis(host='host.docker.internal', port=6379, decode_responses=True)

# Belt configuration
BELT_SPEED = 100  # mm/sec (25% slower than original 133.33)
UPDATE_INTERVAL = 0.1  # 10Hz updates
MOVEMENT_PER_UPDATE = BELT_SPEED * UPDATE_INTERVAL  # 10mm per update

def setup():
    """Setup conveyor and CNC"""
    # Conveyor config with slower speed
    r.hset('conveyor:config', mapping={
        'belt_speed': str(BELT_SPEED),
        'length': '500',
        'vision_zone': '50',
        'trigger_zone': '300',
        'pickup_zone_start': '375',
        'pickup_zone_end': '425',
        'post_pick_zone': '475'
    })
    
    # CNC initial state
    r.hset('cnc:0', mapping={
        'position_x': '400',
        'position_y': '200',
        'position_z': '100',
        'status': 'idle',
        'has_object': 'false'
    })
    
    print(f"Setup complete - Belt speed: {BELT_SPEED} mm/sec")
    print(f"Objects will take {500/BELT_SPEED:.1f} seconds to travel the belt")

def create_object(obj_id, pos, lane, color):
    """Create an object"""
    r.hset(f'object:{obj_id}', mapping={
        'position_x': str(pos),
        'lane': str(lane),
        'type': color,
        'status': 'moving',
        'has_ring': 'false',
        'ring_color': 'yellow'
    })
    r.zadd('objects:active', {obj_id: pos})
    print(f"Created {color} object {obj_id} at {pos}mm in lane {lane}")

def move_objects():
    """Move all objects along belt at slower speed"""
    while True:
        objects = r.zrange('objects:active', 0, -1, withscores=True)
        pipe = r.pipeline()
        
        for obj_id, pos in objects:
            new_pos = pos + MOVEMENT_PER_UPDATE
            
            if new_pos > 500:
                # Remove if past belt
                pipe.zrem('objects:active', obj_id)
                pipe.delete(f'object:{obj_id}')
                print(f"  Removed {obj_id} - past belt end")
            else:
                # Update position
                pipe.hset(f'object:{obj_id}', 'position_x', str(new_pos))
                pipe.zadd('objects:active', {obj_id: new_pos})
                
                # Update ring for green objects
                obj_type = r.hget(f'object:{obj_id}', 'type')
                if obj_type == 'green':
                    if 50 < new_pos < 300:
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'yellow')
                    elif new_pos >= 300:
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'red')
        
        pipe.execute()
        time.sleep(UPDATE_INTERVAL)

def cnc_pickup_cycle():
    """CNC pickup logic with proper timing for slower belt"""
    pickup_count = 0
    
    while True:
        time.sleep(0.3)  # Check more frequently
        
        # Check if CNC is idle
        cnc_status = r.hget('cnc:0', 'status')
        if cnc_status != 'idle':
            continue
        
        # Find green objects in pickup zone
        in_zone = r.zrangebyscore('objects:active', 375, 425, withscores=True)
        
        for obj_id, pos in in_zone:
            obj_type = r.hget(f'object:{obj_id}', 'type')
            
            if obj_type == 'green':
                print(f"\\n{'='*50}")
                print(f"PICKUP SEQUENCE START: {obj_id}")
                print(f"Object at position: {pos:.1f}mm")
                print(f"Belt speed: {BELT_SPEED} mm/sec")
                print(f"{'='*50}")
                
                # Full pickup sequence
                steps = [
                    ('moving_to_position', f"Moving to {pos:.1f}mm", 0.2),
                    ('lowering', "Lowering gripper", 0.2),
                    ('picking', "Vacuum ON - Grabbing object", 0.3),
                    ('raising', "Raising with object", 0.2),
                    ('moving_to_bin', "Moving to bin", 0.3),
                    ('dropping', "Vacuum OFF - Dropping", 0.2),
                    ('returning', "Returning to home", 0.2)
                ]
                
                # Execute pickup sequence
                for status, message, duration in steps:
                    print(f"  → {message}")
                    r.hset('cnc:0', 'status', status)
                    
                    if status == 'moving_to_position':
                        r.hset('cnc:0', 'position_x', str(pos))
                    elif status == 'picking':
                        r.hset('cnc:0', 'has_object', 'true')
                        # Remove from belt
                        r.zrem('objects:active', obj_id)
                        r.delete(f'object:{obj_id}')
                    elif status == 'moving_to_bin':
                        r.hset('cnc:0', 'position_x', '400')
                    elif status == 'dropping':
                        r.hset('cnc:0', 'has_object', 'false')
                        r.setex('bin:0:flash', 1, 'true')
                    elif status == 'returning':
                        r.hset('cnc:0', 'position_x', '400')
                    
                    time.sleep(duration)
                
                # Return to idle
                r.hset('cnc:0', 'status', 'idle')
                pickup_count += 1
                
                print(f"\\nPICKUP COMPLETE! Total successful pickups: {pickup_count}")
                print(f"{'='*50}\\n")
                
                # Wait before next pickup
                time.sleep(0.5)
                break

def spawn_objects():
    """Spawn objects with timing adjusted for slower belt"""
    counter = 1
    colors = ['green', 'blue', 'red', 'yellow', 'orange']
    
    while True:
        # Slightly longer spawn interval for slower belt
        time.sleep(random.uniform(2.5, 3.5))
        
        obj_id = f'obj_{counter:04d}'
        lane = random.randint(0, 3)
        
        # 40% chance of green
        if random.random() < 0.4:
            color = 'green'
        else:
            color = random.choice(['blue', 'red', 'yellow', 'orange'])
        
        create_object(obj_id, 0, lane, color)
        counter += 1

def status_monitor():
    """Show status periodically"""
    while True:
        time.sleep(5)
        active = r.zcard('objects:active')
        cnc_status = r.hget('cnc:0', 'status')
        
        # Calculate approximate object positions
        objects = r.zrange('objects:active', 0, -1, withscores=True)
        green_count = sum(1 for obj_id, _ in objects if r.hget(f'object:{obj_id}', 'type') == 'green')
        
        print(f"[Monitor] Active: {active} (Green: {green_count}), CNC: {cnc_status}, Belt: {BELT_SPEED}mm/s")

def main():
    print("="*60)
    print("Simple Pickup Test - 25% SLOWER BELT SPEED")
    print("="*60)
    print(f"Belt speed reduced from 133.33 to {BELT_SPEED} mm/sec")
    print(f"Objects now take {500/BELT_SPEED:.1f} seconds to travel the belt")
    print("This gives CNC more time to pick up objects!")
    print("="*60 + "\\n")
    
    # Clear Redis
    r.flushdb()
    
    # Setup
    setup()
    
    # Create some initial objects
    create_object('obj_0001', 100, 0, 'green')
    create_object('obj_0002', 200, 1, 'blue')
    create_object('obj_0003', 300, 2, 'green')
    create_object('obj_0004', 350, 3, 'red')
    
    # Start threads
    threads = [
        threading.Thread(target=move_objects, daemon=True),
        threading.Thread(target=cnc_pickup_cycle, daemon=True),
        threading.Thread(target=spawn_objects, daemon=True),
        threading.Thread(target=status_monitor, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    print("Running... Press Ctrl+C to stop\\n")
    print("Watch for PICKUP SEQUENCE messages!\\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\nStopping...")

if __name__ == '__main__':
    main()
'''
    
    with open('simple_pickup_test.py', 'w') as f:
        f.write(content)
    
    print("✓ Created updated simple_pickup_test.py with slower belt")

def main():
    print("Slowing Conveyor Belt by 25%")
    print("="*50)
    print("Original speed: 133.33 mm/sec (3.75 seconds)")
    print("New speed: 100 mm/sec (5 seconds)")
    print("="*50 + "\n")
    
    try:
        update_app_py()
        create_updated_simple_test()
        
        print("\n✅ Updates complete!")
        print("\nBenefits of slower belt:")
        print("- Objects take 5 seconds to travel (was 3.75)")
        print("- More time for CNC to complete pickups")
        print("- Less chance of missing objects")
        print("- Better success rate overall")
        
        print("\nTo test:")
        print("1. Stop current scripts (Ctrl+C)")
        print("2. Restart app.py")
        print("3. Run: python simple_pickup_test.py")
        print("\nYou should see much better pickup success!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()