#!/usr/bin/env python3
"""
CNC Agent for Pick1 System
Receives assignments, coordinates with trigger camera, executes G-code routines
"""

import redis
import time
import json
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from gcode_routines import get_routine, get_routine_time, format_gcode_line

# Redis connection - using localhost with host networking
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# CNC configuration
CNC_ID = 'cnc:0'
HOME_POSITION = {'x': 337.5, 'y': 200, 'z': 100}
PICKUP_ZONE_CENTER = 337.5

# CNC state
current_position = HOME_POSITION.copy()
current_assignment = None
assignment_lock = threading.Lock()

def update_cnc_state(status, position=None):
    """Update CNC state in Redis"""
    if position:
        current_position.update(position)
    
    cnc_data = {
        'position_x': str(current_position['x']),
        'position_y': str(current_position['y']),
        'position_z': str(current_position['z']),
        'status': status,
        'has_object': 'true' if status in ['picking', 'moving_to_bin'] else 'false'
    }
    r.hset(CNC_ID, mapping=cnc_data)

def publish_ready():
    """Publish ready for assignment event"""
    event = {
        'event': 'ready_for_assignment',
        'timestamp': time.time(),
        'data': {
            'cnc_id': CNC_ID,
            'position': current_position['x']
        }
    }
    r.publish('events:cnc', json.dumps(event))
    print(f"[CNC] Published ready_for_assignment")

def request_trigger_watch(obj_id, lane):
    """Request trigger camera to watch for object"""
    # Calculate expected arrival time based on current position
    obj_pos = float(r.hget(f'object:{obj_id}', 'position_x') or 0)
    distance_to_trigger = 250 - obj_pos
    belt_speed = 50
    expected_time = distance_to_trigger / belt_speed if distance_to_trigger > 0 else 0
    
    # Add buffer for timeout
    timeout = expected_time + 1.0
    
    event = {
        'event': 'watch_for_object',
        'timestamp': time.time(),
        'data': {
            'object_id': obj_id,
            'lane': lane,
            'timeout': timeout,
            'requested_by': CNC_ID
        }
    }
    
    r.publish('events:trigger', json.dumps(event))
    print(f"[CNC] Requested trigger watch for {obj_id} in lane {lane} (timeout: {timeout:.1f}s)")

def execute_gcode_routine(routine_name, lane=None, pickup_x=None):
    """Execute a G-code routine with timing"""
    routine = get_routine(routine_name, lane=lane, pickup_x=pickup_x)
    
    print(f"[CNC] Executing {routine_name} routine...")
    
    for gcode, params, desc in routine:
        gcode_line = format_gcode_line(gcode, params)
        print(f"  {gcode_line} ; {desc}")
        
        # Simulate execution time based on command
        if gcode == 'G00':  # Rapid move
            time.sleep(0.1)
        elif gcode == 'G01':  # Feed move
            time.sleep(0.2)
        elif gcode == 'G04':  # Dwell
            dwell_time = float(params.replace('P', ''))
            time.sleep(dwell_time)
        elif gcode in ['M03', 'M05']:  # Gripper
            time.sleep(0.1)
    
    # Total routine time
    total_time = get_routine_time(routine_name)
    print(f"[CNC] {routine_name} complete ({total_time}s)")

def handle_pickup_assignment(assignment_data):
    """Handle pickup assignment from scoring agent"""
    global current_assignment
    
    obj_id = assignment_data.get('object_id')
    lane = assignment_data.get('lane')
    position = assignment_data.get('position')
    
    if not obj_id:
        return
    
    with assignment_lock:
        current_assignment = {
            'object_id': obj_id,
            'lane': lane,
            'position': position,
            'status': 'preparing',
            'start_time': time.time(),
            'trigger_timeout': time.time() + 5.0  # 5-second timeout for trigger message
        }
    
    # Set assigned lane for display
    r.set('cnc:assigned_lane', str(lane))
    
    logging.info(f"Received assignment for {obj_id} in lane {lane} at {position}mm")
    print(f"\n[CNC] Received assignment for {obj_id} in lane {lane}")
    
    # Update status
    update_cnc_state('preparing')
    
    # Pre-position to lane
    execute_gcode_routine('prepare', lane=lane)
    current_position['y'] = lane * 100  # Update Y position
    update_cnc_state('waiting_for_trigger', current_position)
    
    # Request trigger camera to watch
    request_trigger_watch(obj_id, lane)
    
    print(f"[CNC] Pre-positioned to lane {lane}, waiting for trigger...")

def calculate_pickup_timing(trigger_data):
    """Calculate when to start pickup based on trigger data"""
    current_pos = trigger_data.get('current_position', 250)
    velocity = trigger_data.get('velocity', 133.33)
    
    # Target pickup at center of zone
    target_position = PICKUP_ZONE_CENTER
    distance = target_position - current_pos
    
    # Time for object to reach target
    travel_time = distance / velocity if velocity > 0 else 0
    
    # Subtract pickup routine descent time (about 0.3s)
    pickup_prep_time = 0.3
    wait_time = max(0, travel_time - pickup_prep_time)
    
    return wait_time, target_position

def handle_trigger_notification(trigger_data):
    """Handle trigger camera notification"""
    global current_assignment
    
    obj_id = trigger_data.get('object_id')
    
    with assignment_lock:
        if not current_assignment or current_assignment['object_id'] != obj_id:
            logging.warning(f"Ignoring trigger for {obj_id} (not my assignment)")
            print(f"[CNC] Ignoring trigger for {obj_id} (not my assignment)")
            return
        
        current_assignment['status'] = 'executing'
        # Clear trigger timeout since we received the trigger message
        current_assignment.pop('trigger_timeout', None)
    
    # Calculate timing
    wait_time, pickup_x = calculate_pickup_timing(trigger_data)
    
    logging.info(f"Trigger received for {obj_id}! Waiting {wait_time:.3f}s before pickup...")
    print(f"[CNC] Trigger received! Waiting {wait_time:.3f}s before pickup...")
    
    # Wait for optimal moment
    if wait_time > 0:
        time.sleep(wait_time)
    
    # Execute pickup
    update_cnc_state('picking')
    # Set object status to 'picking' to stop belt animation
    r.hset(f'object:{obj_id}', 'status', 'picking')
    execute_gcode_routine('pickup')
    
    # Publish pickup event - let app.py handle object removal
    pickup_event = {
        'event': 'object_picked',
        'timestamp': time.time(),
        'data': {
            'cnc_id': CNC_ID,
            'object_id': obj_id
        }
    }
    r.publish('events:cnc', json.dumps(pickup_event))
    print(f"[CNC] Picked up {obj_id}")
    
    # Deliver to bin
    update_cnc_state('moving_to_bin')
    execute_gcode_routine('deliver')
    
    # Flash bin to indicate drop
    r.setex('bin:0:flash', 1, 'true')
    
    # Return home
    update_cnc_state('returning_home')
    execute_gcode_routine('home')
    current_position.update(HOME_POSITION)
    
    # Clear assignment
    with assignment_lock:
        current_assignment = None
    
    # Set idle and publish ready
    update_cnc_state('idle', current_position)
    print(f"[CNC] Cycle complete, returning to idle\n")
    
    # Brief cooldown
    time.sleep(1.0)
    
    # Publish ready for next assignment
    publish_ready()

def event_listener():
    """Listen for CNC and trigger events"""
    pubsub = redis.Redis(host='localhost', port=6379, decode_responses=True).pubsub()
    pubsub.subscribe(['events:cnc', 'events:trigger'])
    
    print("[CNC] Event listener started...")
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                event = json.loads(message['data'])
                logging.debug(f"Received event: {message['channel']} - {event}")
                
                if message['channel'] == 'events:cnc' and event.get('event') == 'pickup_assignment':
                    # Handle new assignment
                    handle_pickup_assignment(event.get('data', {}))
                    
                elif message['channel'] == 'events:trigger' and event.get('event') == 'object_approaching':
                    # Handle trigger notification
                    handle_trigger_notification(event.get('data', {}))
                    
            except Exception as e:
                logging.error(f"Error handling event: {e}")
                print(f"[CNC] Error handling event: {e}")

def status_reporter():
    """Report CNC status periodically"""
    while True:
        time.sleep(10)
        
        with assignment_lock:
            if current_assignment:
                obj_id = current_assignment['object_id']
                status = current_assignment['status']
                print(f"[CNC] Status: Working on {obj_id} ({status})")
            else:
                print(f"[CNC] Status: Idle at ({current_position['x']}, {current_position['y']}, {current_position['z']})")

def main():
    """Main entry point for CNC agent"""
    global current_assignment
    print("CNC Agent starting...")
    print(f"CNC ID: {CNC_ID}")
    print(f"Home position: {HOME_POSITION}")
    print("=" * 50)
    
    # Initialize CNC state
    update_cnc_state('idle', HOME_POSITION)
    
    # Start threads
    threads = [
        threading.Thread(target=event_listener, daemon=True),
        threading.Thread(target=status_reporter, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    # Publish initial ready state after startup
    time.sleep(2.0)
    publish_ready()
    
    # Track last ready announcement
    last_ready_time = time.time()
    ready_interval = 1.0  # Announce ready every 1 second when idle
    
    try:
        while True:
            current_time = time.time()
            
            # Check for assignment timeout and trigger timeout
            with assignment_lock:
                if current_assignment:
                    # Check for overall assignment timeout (10 seconds)
                    if (current_assignment.get('status') == 'waiting_for_trigger' and
                        current_time - current_assignment.get('start_time', 0) > 10.0):
                        
                        print(f"[CNC] Assignment timeout for {current_assignment.get('object_id')} - resetting to idle")
                        current_assignment = None
                        update_cnc_state('idle', HOME_POSITION)
                        
                        # Clear lane assignment in Redis
                        r.delete('cnc:assigned_lane')
                        r.delete('scoring:confirmed_target')
                        
                        # Publish ready for next assignment
                        time.sleep(1.0)
                        publish_ready()
                        last_ready_time = current_time
                    
                    # Check for trigger message timeout (5 seconds)
                    elif (current_assignment.get('status') == 'waiting_for_trigger' and
                          current_time > current_assignment.get('trigger_timeout', 0)):
                        
                        print(f"[CNC] Trigger timeout for {current_assignment.get('object_id')} - resetting to ready position")
                        
                        # Return to home position
                        execute_gcode_routine('home')
                        current_position.update(HOME_POSITION)
                        
                        # Clear assignment and reset to idle
                        current_assignment = None
                        update_cnc_state('idle', HOME_POSITION)
                        
                        # Clear lane assignment in Redis
                        r.delete('cnc:assigned_lane')
                        r.delete('scoring:confirmed_target')
                        
                        # Publish ready for next assignment
                        time.sleep(1.0)
                        publish_ready()
                        last_ready_time = current_time
                
                # Periodically announce ready when idle and no assignment
                elif (current_assignment is None and 
                      current_time - last_ready_time > ready_interval):
                    publish_ready()
                    last_ready_time = current_time
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCNC Agent shutting down...")

if __name__ == "__main__":
    main()
