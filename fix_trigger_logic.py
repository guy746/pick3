#!/usr/bin/env python3
"""
Fix the trigger logic in enhanced_cnc_simulation.py
"""

def create_fixed_simulation():
    """Create fixed simulation with proper trigger logic"""
    print("Creating fixed enhanced_cnc_simulation.py...")
    
    content = '''#!/usr/bin/env python3
"""
Enhanced test data generator with fixed trigger logic
Only tracks assigned green objects, no false timeouts
"""

import redis
import time
import random
import json
import threading
from enum import Enum

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# CNC States
class CNCState(Enum):
    IDLE = "idle"
    MOVING_TO_LANE = "moving_to_lane"
    WAITING_FOR_TRIGGER = "waiting_for_trigger"
    LOWERING = "lowering"
    PICKING = "picking"
    RAISING = "raising"
    MOVING_TO_BIN = "moving_to_bin"
    DROPPING = "dropping"
    RETURNING_HOME = "returning_home"

# Global variables
pickup_in_progress = False
pickup_lock = threading.Lock()
assigned_objects = {}  # Track assignments like real system
missed_pickups = 0
successful_pickups = 0

# CNC positions
HOME_POSITION = 400
BIN_POSITION = 400  # Same as home but different Z
LANE_POSITIONS = [50, 150, 250, 350]  # Center of each lane

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
    
    r.hset(f'object:{obj_id}', mapping=obj_data)
    r.zadd('objects:active', {obj_id: position_x})
    
    print(f"Created {obj_type} object {obj_id} in lane {lane}")

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
        'position_x': str(HOME_POSITION),
        'position_y': '200',
        'position_z': '100',
        'status': CNCState.IDLE.value,
        'has_object': 'false',
        'target_lane': '-1',
        'target_object': ''
    }
    r.hset('cnc:0', mapping=cnc_data)
    print("Set CNC initial state")

def animate_cnc_position(current_pos, target_pos, speed=200):
    """Animate CNC movement from current to target position"""
    distance = abs(target_pos - current_pos)
    duration = distance / speed  # seconds
    steps = int(duration * 10)  # 10Hz updates
    
    if steps == 0:
        r.hset('cnc:0', 'position_x', str(target_pos))
        return
    
    for i in range(steps + 1):
        pos = current_pos + (target_pos - current_pos) * (i / steps)
        r.hset('cnc:0', 'position_x', str(pos))
        time.sleep(0.1)

def simulate_vision_detection():
    """Simulate vision agent detecting objects"""
    while True:
        time.sleep(0.1)
        
        # Check objects in vision zone
        objects_in_vision = r.zrangebyscore('objects:active', 50, 100, withscores=True)
        
        for obj_id, pos in objects_in_vision:
            obj_type = r.hget(f'object:{obj_id}', 'type')
            has_ring = r.hget(f'object:{obj_id}', 'has_ring')
            
            # ONLY detect green objects
            if obj_type == 'green' and has_ring == 'false':
                # Simulate detection with 98% success rate
                if random.random() < 0.98:
                    print(f"[Vision] Detected green object: {obj_id}")

def simulate_scoring_assignment():
    """Simulate scoring agent assigning objects"""
    global assigned_objects
    
    while True:
        time.sleep(0.5)
        
        # Check for green objects approaching trigger zone
        approaching_objects = r.zrangebyscore('objects:active', 200, 290, withscores=True)
        
        for obj_id, pos in approaching_objects:
            obj_type = r.hget(f'object:{obj_id}', 'type')
            
            # ONLY assign green objects that aren't already assigned
            if obj_type == 'green' and obj_id not in assigned_objects:
                # Check if CNC is available
                cnc_status = r.hget('cnc:0', 'status')
                if cnc_status == CNCState.IDLE.value:
                    lane = int(r.hget(f'object:{obj_id}', 'lane'))
                    assigned_objects[obj_id] = {
                        'lane': lane,
                        'assigned_time': time.time(),
                        'assigned_position': pos
                    }
                    print(f"[Scoring] Assigned GREEN object {obj_id} to CNC (lane {lane})")
                    break  # Only assign one at a time

def simulate_pickup():
    """Simulate full CNC pickup sequence"""
    global pickup_in_progress, missed_pickups, successful_pickups
    
    while True:
        time.sleep(0.2)
        
        # Check for assigned objects
        if not assigned_objects:
            continue
            
        # Get CNC status
        cnc_status = r.hget('cnc:0', 'status')
        if cnc_status != CNCState.IDLE.value:
            continue
        
        # Process oldest assignment
        obj_id = next(iter(assigned_objects))
        assignment = assigned_objects[obj_id]
        lane = assignment['lane']
        assigned_pos = assignment['assigned_position']
        
        # Check if object still exists
        if not r.exists(f'object:{obj_id}'):
            del assigned_objects[obj_id]
            continue
        
        with pickup_lock:
            pickup_in_progress = True
        
        print(f"\\n[CNC] Starting pickup sequence for GREEN object {obj_id} in lane {lane}")
        print(f"[CNC] Object was at position {assigned_pos:.1f}mm when assigned")
        
        # 1. Move to lane position
        current_x = float(r.hget('cnc:0', 'position_x'))
        lane_x = LANE_POSITIONS[lane] + 350  # Offset to pickup zone
        
        r.hset('cnc:0', 'status', CNCState.MOVING_TO_LANE.value)
        r.hset('cnc:0', 'target_lane', str(lane))
        r.hset('cnc:0', 'target_object', obj_id)
        
        animate_cnc_position(current_x, lane_x)
        
        # 2. Wait for trigger - Calculate proper timeout
        r.hset('cnc:0', 'status', CNCState.WAITING_FOR_TRIGGER.value)
        
        # Calculate how long object needs to reach pickup zone
        current_obj_pos = r.zscore('objects:active', obj_id)
        if current_obj_pos:
            distance_to_pickup = 375 - current_obj_pos
            time_to_arrival = distance_to_pickup / 133.33  # belt speed
            timeout = max(5.0, time_to_arrival + 2.0)  # Add 2 second buffer
            
            print(f"[CNC] Waiting for {obj_id} - currently at {current_obj_pos:.1f}mm")
            print(f"[CNC] Expected arrival in {time_to_arrival:.1f} seconds (timeout: {timeout:.1f}s)")
        else:
            timeout = 5.0
        
        # Wait for specific object only
        wait_start = time.time()
        object_found = False
        
        while time.time() - wait_start < timeout:
            # ONLY check for the assigned object
            obj_pos = r.zscore('objects:active', obj_id)
            
            if obj_pos is None:
                # Object disappeared
                print(f"[CNC] Object {obj_id} disappeared!")
                break
                
            if 375 <= obj_pos <= 425:
                object_found = True
                print(f"[Trigger] TARGET object {obj_id} in pickup zone at {obj_pos:.1f}mm!")
                break
            
            # Show waiting status
            if int(time.time() - wait_start) % 1 == 0:
                remaining = timeout - (time.time() - wait_start)
                if obj_pos:
                    print(f"[CNC] Tracking {obj_id} at {obj_pos:.1f}mm (timeout in {remaining:.1f}s)")
            
            time.sleep(0.1)
        
        if not object_found:
            print(f"[CNC] MISSED - {obj_id} didn't reach pickup zone in time")
            missed_pickups += 1
            
            # Return home
            r.hset('cnc:0', 'status', CNCState.RETURNING_HOME.value)
            animate_cnc_position(lane_x, HOME_POSITION)
            r.hset('cnc:0', 'status', CNCState.IDLE.value)
            r.hset('cnc:0', 'target_object', '')
            del assigned_objects[obj_id]
            
            with pickup_lock:
                pickup_in_progress = False
            continue
        
        # 3. Lower to pick
        r.hset('cnc:0', 'status', CNCState.LOWERING.value)
        time.sleep(0.2)
        
        # 4. Pick object (vacuum on)
        r.hset('cnc:0', 'status', CNCState.PICKING.value)
        r.hset('cnc:0', 'has_object', 'true')
        
        # Remove object from belt
        r.zrem('objects:active', obj_id)
        r.delete(f'object:{obj_id}')
        time.sleep(0.1)
        
        # 5. Raise with object
        r.hset('cnc:0', 'status', CNCState.RAISING.value)
        time.sleep(0.2)
        
        # 6. Move to bin
        r.hset('cnc:0', 'status', CNCState.MOVING_TO_BIN.value)
        current_x = float(r.hget('cnc:0', 'position_x'))
        animate_cnc_position(current_x, BIN_POSITION)
        
        # 7. Drop object
        r.hset('cnc:0', 'status', CNCState.DROPPING.value)
        time.sleep(0.2)
        r.hset('cnc:0', 'has_object', 'false')
        r.setex('bin:0:flash', 1, 'true')
        
        successful_pickups += 1
        print(f"[CNC] SUCCESS - Picked up {obj_id}")
        
        # 8. Return to home
        r.hset('cnc:0', 'status', CNCState.RETURNING_HOME.value)
        animate_cnc_position(BIN_POSITION, HOME_POSITION)
        
        # 9. Back to idle
        r.hset('cnc:0', 'status', CNCState.IDLE.value)
        r.hset('cnc:0', 'target_lane', '-1')
        r.hset('cnc:0', 'target_object', '')
        
        del assigned_objects[obj_id]
        
        with pickup_lock:
            pickup_in_progress = False
        
        print(f"[CNC] Ready for next pickup\\n")

def monitor_missed_objects():
    """Monitor objects that pass through without pickup"""
    global missed_pickups
    
    while True:
        time.sleep(1)
        
        # Check GREEN objects past pickup zone
        past_pickup = r.zrangebyscore('objects:active', 426, 475, withscores=True)
        
        for obj_id, pos in past_pickup:
            obj_type = r.hget(f'object:{obj_id}', 'type')
            # Only count missed GREEN objects
            if obj_type == 'green' and obj_id in assigned_objects:
                print(f"[Monitor] GREEN object {obj_id} passed through without pickup!")
                missed_pickups += 1
                del assigned_objects[obj_id]

def animate_objects():
    """Animate objects moving along belt"""
    belt_speed = 133.33
    update_interval = 0.1
    
    while True:
        start_time = time.time()
        
        active_objects = r.zrange('objects:active', 0, -1, withscores=True)
        pipe = r.pipeline()
        
        for obj_id, position in active_objects:
            new_position = position + (belt_speed * update_interval)
            
            if new_position > 500:
                pipe.zrem('objects:active', obj_id)
                pipe.delete(f'object:{obj_id}')
                if obj_id in assigned_objects:
                    del assigned_objects[obj_id]
            else:
                pipe.hset(f'object:{obj_id}', 'position_x', str(new_position))
                pipe.zadd('objects:active', {obj_id: new_position})
                
                obj_type = r.hget(f'object:{obj_id}', 'type')
                if obj_type == 'green':
                    if 50 < new_position < 300:
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'yellow')
                    elif 300 <= new_position:
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'red')
        
        pipe.execute()
        
        elapsed = time.time() - start_time
        sleep_time = max(0, update_interval - elapsed)
        time.sleep(sleep_time)

def spawn_objects():
    """Spawn new objects periodically"""
    obj_counter = 1
    colors = ['green', 'blue', 'red', 'yellow', 'orange']
    weights = [0.35, 0.16, 0.16, 0.16, 0.17]  # 35% green
    
    while True:
        time.sleep(random.uniform(1.5, 2.5))
        
        obj_id = f'obj_{obj_counter:04d}'
        lane = random.randint(0, 3)
        color = random.choices(colors, weights=weights)[0]
        
        create_test_object(obj_id, 0, lane, color)
        obj_counter += 1

def report_statistics():
    """Report pickup statistics"""
    while True:
        time.sleep(10)
        
        total = successful_pickups + missed_pickups
        if total > 0:
            success_rate = (successful_pickups / total) * 100
            print(f"\\n[Stats] GREEN Object Pickups: {successful_pickups}, Misses: {missed_pickups}, Success Rate: {success_rate:.1f}%")
            print(f"[Stats] Currently tracking: {len(assigned_objects)} assigned green objects\\n")

def main():
    """Run enhanced test data generator"""
    print("Starting Pick1 Enhanced Test Environment (Fixed)")
    print("="*50)
    print("Features:")
    print("- ONLY tracks assigned GREEN objects")
    print("- Proper timeout calculation based on object position")
    print("- No false misses from timeout")
    print("- Realistic misses only when CNC is busy")
    print("="*50 + "\\n")
    
    r.flushdb()
    
    setup_conveyor_config()
    setup_cnc()
    
    # Create initial objects - mix of colors
    create_test_object('obj_0001', 50, 0, 'green')
    create_test_object('obj_0002', 100, 1, 'blue')
    create_test_object('obj_0003', 150, 2, 'red')
    create_test_object('obj_0004', 200, 3, 'green')
    create_test_object('obj_0005', 250, 0, 'yellow')
    
    print("\\nStarting simulation threads...")
    print("Press Ctrl+C to stop\\n")
    
    threads = [
        threading.Thread(target=animate_objects, daemon=True),
        threading.Thread(target=spawn_objects, daemon=True),
        threading.Thread(target=simulate_vision_detection, daemon=True),
        threading.Thread(target=simulate_scoring_assignment, daemon=True),
        threading.Thread(target=simulate_pickup, daemon=True),
        threading.Thread(target=monitor_missed_objects, daemon=True),
        threading.Thread(target=report_statistics, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n\\nFinal Statistics:")
        print(f"GREEN objects successfully picked: {successful_pickups}")
        print(f"GREEN objects missed: {missed_pickups}")
        if successful_pickups + missed_pickups > 0:
            print(f"Success rate: {(successful_pickups/(successful_pickups+missed_pickups))*100:.1f}%")

if __name__ == '__main__':
    main()
'''
    
    with open('enhanced_cnc_simulation.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed enhanced_cnc_simulation.py created")

def main():
    print("Fixing CNC trigger logic...\n")
    
    create_fixed_simulation()
    
    print("\n✅ Fix applied!")
    print("\nWhat was fixed:")
    print("1. CNC now ONLY tracks the specific assigned GREEN object")
    print("2. Proper timeout calculation based on object position")
    print("3. No more false timeouts - waits appropriately for object arrival")
    print("4. Vision and scoring ONLY process green objects")
    print("5. Clear console messages showing what object is being tracked")
    print("\nMisses now only happen when:")
    print("- CNC is busy with another pickup")
    print("- Multiple green objects arrive too close together")
    print("\nRun the fixed simulation to see the difference!")

if __name__ == '__main__':
    main()