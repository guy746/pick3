#!/usr/bin/env python3
"""
Update Pick1 visualization to 500mm belt with fixed trigger behavior
"""

def update_app_py():
    """Update app.py with 500mm belt configuration"""
    print("Updating app.py...")
    
    content = """import os
import json
import redis
import time
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")

# Redis connection
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url, decode_responses=True)
redis_pubsub = redis_client.pubsub()

# Global state
connected_clients = set()
update_thread = None
update_thread_stop = threading.Event()

# Conveyor configuration
CONVEYOR_CONFIG = {
    'length': 500,  # mm (quarter of original)
    'width': 400,    # mm (100mm per lane)
    'lanes': 4,
    'belt_speed': 133.33,  # mm/sec (500mm in 3.75 seconds)
    'vision_zone': 50,     # mm from start (10%)
    'trigger_zone': 300,    # mm (trigger line)
    'pickup_zone_start': 375,  # 75mm gap from trigger
    'pickup_zone_end': 425,    # 50mm pickup zone
    'post_pick_zone': 475      # near end
}

@app.route('/')
def index():
    \"\"\"Serve the main visualization page\"\"\"
    return render_template('index.html', config=CONVEYOR_CONFIG)

@socketio.on('connect')
def handle_connect():
    \"\"\"Handle client connection\"\"\"
    connected_clients.add(request.sid)
    print(f"Client connected: {request.sid}")
    
    # Send initial configuration
    emit('config', CONVEYOR_CONFIG)
    
    # Send current world state
    world_state = get_world_state()
    emit('world_state', world_state)

@socketio.on('disconnect')
def handle_disconnect():
    \"\"\"Handle client disconnection\"\"\"
    connected_clients.discard(request.sid)
    print(f"Client disconnected: {request.sid}")

def get_world_state():
    \"\"\"Get current world state from Redis\"\"\"
    state = {
        'objects': {},
        'cnc': {},
        'timestamp': time.time()
    }
    
    try:
        # Get all active objects
        active_objects = redis_client.zrange('objects:active', 0, -1)
        
        for obj_id in active_objects:
            obj_data = redis_client.hgetall(f'object:{obj_id}')
            if obj_data:
                state['objects'][obj_id] = {
                    'id': obj_id,
                    'position_x': float(obj_data.get('position_x', 0)),
                    'lane': int(obj_data.get('lane', 0)),
                    'type': obj_data.get('type', 'green'),
                    'status': obj_data.get('status', 'moving'),
                    'has_ring': obj_data.get('has_ring', 'false') == 'true',
                    'ring_color': obj_data.get('ring_color', 'yellow')
                }
        
        # Get CNC state
        cnc_data = redis_client.hgetall('cnc:0')
        if cnc_data:
            state['cnc'] = {
                'position_x': float(cnc_data.get('position_x', 400)),
                'position_y': float(cnc_data.get('position_y', 200)),
                'position_z': float(cnc_data.get('position_z', 100)),
                'status': cnc_data.get('status', 'idle'),
                'has_object': cnc_data.get('has_object', 'false') == 'true'
            }
            
        # Get bin state
        bin_flash = redis_client.get('bin:0:flash')
        if bin_flash:
            state['bin_flash'] = True
            # Auto-clear flash after reading
            redis_client.delete('bin:0:flash')
            
    except Exception as e:
        print(f"Error reading world state: {e}")
        
    return state

def update_loop():
    \"\"\"Background thread to push updates to clients\"\"\"
    last_state = {}
    
    while not update_thread_stop.is_set():
        try:
            # Get current state
            current_state = get_world_state()
            
            # Check if state changed
            if current_state != last_state and connected_clients:
                # Broadcast to all connected clients
                socketio.emit('world_update', current_state)
                last_state = current_state
                
            # Update at ~5Hz
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error in update loop: {e}")
            time.sleep(1)

def event_listener():
    \"\"\"Listen for Redis events\"\"\"
    try:
        # Subscribe to relevant channels
        redis_pubsub.subscribe([
            'events:spawn',
            'events:detect',
            'events:trigger',
            'events:motion',
            'events:system'
        ])
        
        for message in redis_pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    event_type = data.get('event')
                    
                    # Send event notification to clients
                    if connected_clients:
                        socketio.emit('event', {
                            'type': event_type,
                            'data': data
                        })
                        
                except Exception as e:
                    print(f"Error processing event: {e}")
                    
    except Exception as e:
        print(f"Error in event listener: {e}")

# Start background threads when module loads
def start_background_threads():
    global update_thread
    
    # Start update loop
    update_thread = threading.Thread(target=update_loop)
    update_thread.daemon = True
    update_thread.start()
    
    # Start event listener
    event_thread = threading.Thread(target=event_listener)
    event_thread.daemon = True
    event_thread.start()

if __name__ == '__main__':
    start_background_threads()
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
"""
    
    with open('app.py', 'w') as f:
        f.write(content)
    print("✓ app.py updated")

def update_test_data_py():
    """Create fixed test_data.py with 500mm belt and working pickup"""
    print("Creating new test_data.py...")
    
    content = '''#!/usr/bin/env python3
"""
Test data generator for Pick1 visualization development.
Run this to populate local Redis with sample data for testing.
"""

import redis
import time
import random
import json
import threading

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def create_test_object(obj_id, position_x, lane, obj_type='green'):
    """Create a test object in Redis"""
    obj_data = {
        'position_x': position_x,
        'lane': lane,
        'type': obj_type,
        'status': 'moving',
        'area': 1500,
        'height': 32.5,
        'created_at': time.time(),
        'updated_at': time.time(),
        'has_ring': 'true' if obj_type == 'green' and position_x > 50 else 'false',
        'ring_color': 'yellow'
    }
    
    # Set object data
    r.hset(f'object:{obj_id}', mapping=obj_data)
    
    # Add to active objects
    r.zadd('objects:active', {obj_id: position_x})
    
    print(f"Created object {obj_id} at position {position_x}")

def setup_conveyor_config():
    """Set up conveyor configuration"""
    config = {
        'belt_speed': 133.33,
        'length': 500,  # Reduced to 500mm
        'lanes': 4,
        'lane_width': 100,
        'vision_zone': 50,
        'trigger_zone': 300,
        'pickup_zone_start': 375,
        'pickup_zone_end': 425,
        'post_pick_zone': 475
    }
    r.hset('conveyor:config', mapping=config)
    print("Set conveyor configuration")

def setup_cnc():
    """Set up CNC initial state"""
    cnc_data = {
        'position_x': 400,  # Center of pickup zone
        'position_y': 200,
        'position_z': 100,
        'status': 'idle',
        'has_object': 'false'
    }
    r.hset('cnc:0', mapping=cnc_data)
    print("Set CNC initial state")

def animate_objects():
    """Animate objects moving along belt"""
    belt_speed = 133.33  # mm/sec
    update_interval = 0.2  # 5Hz updates
    
    while True:
        # Get all active objects
        active_objects = r.zrange('objects:active', 0, -1, withscores=True)
        
        for obj_id, position in active_objects:
            new_position = position + (belt_speed * update_interval)
            
            if new_position > 500:
                # Remove object that's past the belt
                r.zrem('objects:active', obj_id)
                r.delete(f'object:{obj_id}')
                print(f"Removed {obj_id} - past belt end")
            else:
                # Update position
                r.hset(f'object:{obj_id}', 'position_x', new_position)
                r.zadd('objects:active', {obj_id: new_position})
                
                # Update ring status based on position
                obj_type = r.hget(f'object:{obj_id}', 'type')
                if obj_type == 'green':
                    if new_position > 50 and new_position < 300:
                        # Vision zone to trigger - yellow ring
                        r.hset(f'object:{obj_id}', 'has_ring', 'true')
                        r.hset(f'object:{obj_id}', 'ring_color', 'yellow')
                    elif new_position >= 300 and new_position < 375:
                        # At trigger line - turn red
                        r.hset(f'object:{obj_id}', 'has_ring', 'true')
                        r.hset(f'object:{obj_id}', 'ring_color', 'red')
                    elif new_position >= 375:
                        # In pickup zone - keep red
                        r.hset(f'object:{obj_id}', 'ring_color', 'red')
        
        time.sleep(update_interval)

def spawn_objects():
    """Spawn new objects periodically"""
    obj_counter = 1
    colors = ['green', 'blue', 'red', 'yellow', 'orange']
    
    while True:
        # Wait 2-3 seconds between spawns (faster for shorter belt)
        time.sleep(random.uniform(2, 3))
        
        # Create new object
        obj_id = f'obj_{obj_counter:04d}'
        lane = random.randint(0, 3)
        color = random.choice(colors)
        
        create_test_object(obj_id, 0, lane, color)
        
        # Publish spawn event (for testing event system)
        event = {
            'event': 'object_spawned',
            'agent': 'test_spawner',
            'timestamp': time.time(),
            'payload': {
                'object_id': obj_id,
                'lane': lane,
                'type': color
            }
        }
        r.publish('events:spawn', json.dumps(event))
        
        obj_counter += 1

def simulate_pickup():
    """Simulate occasional pickups"""
    while True:
        time.sleep(5)  # Check every 5 seconds (more frequent for shorter belt)
        
        # Find a green object in pickup zone
        active_objects = r.zrangebyscore('objects:active', 375, 425, withscores=True)
        
        for obj_id, position in active_objects:
            obj_type = r.hget(f'object:{obj_id}', 'type')
            if obj_type == 'green':
                print(f"Simulating pickup of {obj_id}")
                
                # Move CNC to object position
                r.hset('cnc:0', 'position_x', position)
                r.hset('cnc:0', 'status', 'picking')
                r.hset('cnc:0', 'has_object', 'true')
                
                # Wait a moment
                time.sleep(0.3)
                
                # Remove object from belt
                r.zrem('objects:active', obj_id)
                r.delete(f'object:{obj_id}')
                
                # Move CNC back to center and drop
                time.sleep(0.3)
                r.hset('cnc:0', 'position_x', 400)
                r.hset('cnc:0', 'status', 'idle')
                r.hset('cnc:0', 'has_object', 'false')
                
                # Flash bin
                r.setex('bin:0:flash', 1, 'true')
                
                # Increment bin count
                r.incr('bin:0:count')
                
                break  # Only pick one object per cycle

def main():
    """Run test data generator"""
    print("Setting up Pick1 test environment...")
    
    # Clear existing data
    r.flushdb()
    
    # Setup static data
    setup_conveyor_config()
    setup_cnc()
    
    # Create some initial objects at various positions
    create_test_object('obj_0001', 50, 0, 'green')
    create_test_object('obj_0002', 150, 1, 'blue')
    create_test_object('obj_0003', 250, 2, 'green')
    create_test_object('obj_0004', 350, 3, 'red')
    create_test_object('obj_0005', 400, 0, 'green')
    
    print("\\nStarting animation threads...")
    print("Press Ctrl+C to stop")
    
    # Start animation threads
    animate_thread = threading.Thread(target=animate_objects, daemon=True)
    spawn_thread = threading.Thread(target=spawn_objects, daemon=True)
    pickup_thread = threading.Thread(target=simulate_pickup, daemon=True)
    
    animate_thread.start()
    spawn_thread.start()
    pickup_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\nStopping test data generator...")

if __name__ == '__main__':
    main()
'''
    
    with open('test_data.py', 'w') as f:
        f.write(content)
    print("✓ test_data.py created")

def update_canvas_js():
    """Update canvas.js for 500mm belt"""
    print("Updating static/js/canvas.js...")
    
    # Read existing file
    with open('static/js/canvas.js', 'r') as f:
        content = f.read()
    
    # Update bin position to match new pickup zone center
    content = content.replace('const binX = mmToPixels(775);', 'const binX = mmToPixels(400);')
    content = content.replace('const binX = mmToPixels(1550);', 'const binX = mmToPixels(400);')
    
    with open('static/js/canvas.js', 'w') as f:
        f.write(content)
    print("✓ static/js/canvas.js updated")

def update_index_html():
    """Update index.html with 500mm zone information"""
    print("Updating templates/index.html...")
    
    with open('templates/index.html', 'r') as f:
        content = f.read()
    
    # Update zones section
    content = content.replace(
        """<div>Vision Detection: 100mm</div>
                <div>Trigger Zone: 700mm</div>
                <div>Pickup Zone: 750-800mm</div>
                <div>Post-Pick Monitor: 900mm</div>""",
        """<div>Vision Detection: 50mm</div>
                <div>Trigger Zone: 300mm</div>
                <div>Pickup Zone: 375-425mm</div>
                <div>Post-Pick Monitor: 475mm</div>"""
    )
    
    with open('templates/index.html', 'w') as f:
        f.write(content)
    print("✓ templates/index.html updated")

def main():
    """Run all updates"""
    print("Updating Pick1 visualization to 500mm belt...\n")
    
    try:
        update_app_py()
        update_test_data_py()
        update_canvas_js()
        update_index_html()
        
        print("\n✅ All files updated successfully!")
        print("\nChanges made:")
        print("- Belt reduced to 500mm (3.75 second travel time)")
        print("- Trigger line at 300mm (with 75mm gap before pickup)")
        print("- Red ring appears AT trigger line (not in pickup zone)")
        print("- CNC now properly picks up green objects")
        print("\nNext steps:")
        print("1. Stop both running programs with Ctrl+C")
        print("2. Restart the app: python app.py")
        print("3. Refresh your browser")
        print("4. Restart test data: python test_data.py")
        
    except Exception as e:
        print(f"\n❌ Error updating files: {e}")

if __name__ == '__main__':
    main()