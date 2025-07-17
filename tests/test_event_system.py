#!/usr/bin/env python3
"""
Test the event-driven system by monitoring all events
"""

import redis
import json
import threading
import time

def monitor_events():
    """Monitor all events in the system"""
    # Using host.docker.internal for Docker container networking
    r = redis.Redis(host='host.docker.internal', port=6379, decode_responses=True)
    pubsub = r.pubsub()
    
    # Subscribe to all event channels
    pubsub.subscribe(['events:vision', 'events:cnc', 'events:trigger', 'events:motion', 'events:system', 'events:monitor'])
    
    print("Monitoring all events...")
    print("="*60)
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                event = json.loads(message['data'])
                event_type = event.get('event')
                channel = message['channel']
                
                print(f"\n[{time.strftime('%H:%M:%S')}] Channel: {channel}")
                print(f"Event: {event_type}")
                
                if channel == 'events:vision' and event_type == 'object_detected':
                    data = event.get('data', {})
                    print(f"  - Object: {data.get('id')}")
                    print(f"  - Type: {data.get('type')}")
                    print(f"  - Position: {data.get('position_x')}mm")
                    
                elif channel == 'events:cnc' and event_type == 'ready_for_assignment':
                    data = event.get('data', {})
                    print(f"  - CNC: {data.get('cnc_id')} is ready")
                    
                elif channel == 'events:cnc' and event_type == 'pickup_assignment':
                    data = event.get('data', {})
                    print(f"  - Assigned: {data.get('object_id')}")
                    print(f"  - Position: {data.get('position')}mm")
                    print(f"  - Score: {data.get('score'):.3f}")
                    
                elif channel == 'events:trigger' and event_type == 'watch_for_object':
                    data = event.get('data', {})
                    print(f"  - Watch requested: {data.get('object_id')}")
                    print(f"  - Lane: {data.get('lane')}")
                    print(f"  - Timeout: {data.get('timeout'):.1f}s")
                    
                elif channel == 'events:trigger' and event_type == 'object_approaching':
                    data = event.get('data', {})
                    print(f"  - Object triggered: {data.get('object_id')}")
                    print(f"  - Position: {data.get('current_position')}mm")
                    print(f"  - Velocity: {data.get('velocity')}mm/s")
                    print(f"  - ETA to pickup: {data.get('eta_to_pickup'):.3f}s")
                    
                elif channel == 'events:monitor' and event_type == 'monitor_line_crossed':
                    data = event.get('data', {})
                    print(f"  - Monitor detected: {data.get('object_id')} (green)")
                    print(f"  - Position: {data.get('position')}mm")
                    print(f"  - Lane: {data.get('lane')}")
                    print(f"  - Velocity: {data.get('velocity')}mm/s")
                    print(f"  - Height: {data.get('height')}")
                    print(f"  - Area: {data.get('area')}")
                    
            except Exception as e:
                print(f"Error parsing event: {e}")

def show_system_state():
    """Periodically show system state"""
    # Using host.docker.internal for Docker container networking
    r = redis.Redis(host='host.docker.internal', port=6379, decode_responses=True)
    
    while True:
        time.sleep(5)
        
        # Get system state
        active_objects = r.zcard('objects:active')
        cnc_status = r.hget('cnc:0', 'status')
        cnc_position = r.hget('cnc:0', 'position_x')
        
        # Count green objects
        green_count = 0
        all_objects = r.zrange('objects:active', 0, -1)
        for obj_id in all_objects:
            if r.hget(f'object:{obj_id}', 'type') == 'green':
                green_count += 1
        
        print(f"\n{'='*60}")
        print(f"SYSTEM STATE: Objects: {active_objects} (Green: {green_count}) | CNC: {cnc_status} @ {cnc_position}mm")
        print(f"{'='*60}")

def main():
    """Run event monitor"""
    print("Event System Monitor")
    print("="*60)
    print("This tool monitors all events in the Pick1 system")
    print("Press Ctrl+C to stop")
    print("")
    
    # Start monitoring threads
    event_thread = threading.Thread(target=monitor_events, daemon=True)
    state_thread = threading.Thread(target=show_system_state, daemon=True)
    
    event_thread.start()
    state_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")

if __name__ == "__main__":
    main()