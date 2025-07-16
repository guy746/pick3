#!/usr/bin/env python3
"""
Trigger Camera Agent for Pick1 System
Monitors specific lanes at the 250mm trigger line for watched objects
"""

import redis
import time
import json
import threading
import logging
from datetime import datetime, timedelta

# Redis connection - using localhost with host networking
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r_pubsub = r.pubsub()

# Trigger configuration
TRIGGER_LINE = 250  # mm - trigger camera position
CHECK_INTERVAL = 0.02  # 50Hz checking rate for high precision
BELT_SPEED = 50  # mm/sec (default)

# Watch list management
watch_list = {}  # {obj_id: {'lane': lane, 'timeout': timestamp, 'requested_by': cnc_id}}
watch_lock = threading.Lock()

def cleanup_expired_watches():
    """Remove watches that have timed out"""
    current_time = datetime.now()
    with watch_lock:
        expired = []
        for obj_id, watch_data in watch_list.items():
            if current_time > watch_data['timeout']:
                expired.append(obj_id)
        
        for obj_id in expired:
            del watch_list[obj_id]
            print(f"[TriggerCamera] Watch expired for {obj_id}")

def calculate_object_velocity(obj_id, current_pos):
    """Calculate object velocity based on position changes"""
    # For simulation, use belt speed
    # In real system, would track position history
    return BELT_SPEED

def publish_trigger_event(obj_id, position, lane, velocity):
    """Publish object approaching event"""
    # Calculate ETA to pickup zone center (337.5mm)
    pickup_zone_center = 337.5
    distance_to_pickup = pickup_zone_center - position
    eta = distance_to_pickup / velocity if velocity > 0 else 0
    
    event = {
        'event': 'object_approaching',
        'timestamp': time.time(),
        'data': {
            'object_id': obj_id,
            'current_position': position,
            'lane': lane,
            'velocity': velocity,
            'eta_to_pickup': eta,
            'trigger_position': TRIGGER_LINE
        }
    }
    
    r.publish('events:trigger', json.dumps(event))
    
    # Detailed trace logging
    print(f"[TriggerCamera] Published trigger event: {event}")
    print(f"[TriggerCamera] Triggered: {obj_id} at {position:.1f}mm, ETA: {eta:.3f}s")

def monitor_trigger_line():
    """Monitor specific lanes for watched objects - SIMULATION MODE: 100% detection"""
    
    while True:
        start_time = time.time()
        
        # Cleanup expired watches
        cleanup_expired_watches()
        
        # Get current watch list
        with watch_lock:
            if not watch_list:
                # No watches active, sleep and continue
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Copy watch list for processing
            current_watches = dict(watch_list)
        
        # Track objects to remove after iteration
        objects_to_remove = []
        
        # SIMULATION MODE: Always detect watched objects that exist
        for obj_id, watch_data in current_watches.items():
            lane = watch_data['lane']
            
            # Check if object exists and get its position
            obj_exists = r.exists(f'object:{obj_id}')
            if obj_exists:
                obj_position = r.zscore('objects:active', obj_id)
                obj_lane = r.hget(f'object:{obj_id}', 'lane')
                
                if obj_position is not None and obj_lane and int(obj_lane) == lane:
                    # SIMULATION: Always trigger when object is at or past trigger line
                    if obj_position >= TRIGGER_LINE:
                        print(f"[TriggerCamera] SIMULATION: Force detecting {obj_id} at {obj_position:.1f}mm in lane {lane}")
                        
                        velocity = calculate_object_velocity(obj_id, obj_position)
                        publish_trigger_event(obj_id, obj_position, lane, velocity)
                        
                        # Mark for removal from watch list after triggering
                        objects_to_remove.append(obj_id)
                    else:
                        print(f"[TriggerCamera] SIMULATION: Watching {obj_id} at {obj_position:.1f}mm (waiting for {TRIGGER_LINE}mm)")
        
        # Remove triggered objects from watch list
        if objects_to_remove:
            with watch_lock:
                for obj_id in objects_to_remove:
                    if obj_id in watch_list:
                        del watch_list[obj_id]
                        print(f"[TriggerCamera] Removed {obj_id} from watch list (triggered)")
        
        # Sleep to maintain check rate
        elapsed = time.time() - start_time
        sleep_time = max(0, CHECK_INTERVAL - elapsed)
        time.sleep(sleep_time)

def handle_watch_request(event_data):
    """Handle request to watch for specific object"""
    data = event_data.get('data', {})
    obj_id = data.get('object_id')
    lane = data.get('lane')
    timeout_seconds = data.get('timeout', 3.0)  # Default 3 second timeout
    requested_by = data.get('requested_by', 'cnc:0')
    
    if obj_id and lane is not None:
        timeout_time = datetime.now() + timedelta(seconds=timeout_seconds)
        
        with watch_lock:
            watch_list[obj_id] = {
                'lane': lane,
                'timeout': timeout_time,
                'requested_by': requested_by
            }
        
        print(f"[TriggerCamera] Watching for {obj_id} in lane {lane} (timeout: {timeout_seconds}s)")

def event_listener():
    """Listen for watch requests"""
    # Subscribe to trigger events
    r_pubsub.subscribe(['events:trigger'])
    
    print("[TriggerCamera] Event listener started...")
    
    for message in r_pubsub.listen():
        if message['type'] == 'message':
            try:
                event = json.loads(message['data'])
                print(f"[TriggerCamera] Received event: {event}")
                event_type = event.get('event')
                
                if event_type == 'watch_for_object':
                    handle_watch_request(event)
                    
            except Exception as e:
                print(f"[TriggerCamera] Error handling event: {e}")
                print(f"[TriggerCamera] Error handling event: {e}")

def status_reporter():
    """Periodically report trigger camera status"""
    while True:
        time.sleep(10)
        
        with watch_lock:
            watch_count = len(watch_list)
            if watch_count > 0:
                watches = list(watch_list.keys())
                print(f"[TriggerCamera] Active watches: {watch_count} - {watches}")

def main():
    """Main entry point for trigger camera agent"""
    # Logging configured to use print statements for debugging
    
    print("Trigger Camera Agent starting...")
    print(f"Monitoring trigger line at {TRIGGER_LINE}mm")
    print(f"Check rate: {1/CHECK_INTERVAL:.1f}Hz")
    print("=" * 50)
    
    # Start threads
    threads = [
        threading.Thread(target=monitor_trigger_line, daemon=True),
        threading.Thread(target=event_listener, daemon=True),
        threading.Thread(target=status_reporter, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTrigger Camera Agent shutting down...")

if __name__ == "__main__":
    main()
