import os
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
    'belt_speed': 50,   # mm/sec (500mm in 10 seconds)
    'vision_zone': 50,     # mm from start (10%)
    'trigger_zone': 250,    # mm (trigger line)
    'pickup_zone_start': 300,  # 50mm gap from trigger
    'pickup_zone_end': 375,    # 75mm pickup zone (50% larger)
    'post_pick_zone': 475      # near end
}

@app.route('/')
def index():
    """Serve the main visualization page"""
    return render_template('index.html', config=CONVEYOR_CONFIG)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    connected_clients.add(request.sid)
    print(f"Client connected: {request.sid}")
    
    # Send initial configuration
    emit('config', CONVEYOR_CONFIG)
    
    # Send current world state
    world_state = get_world_state()
    emit('world_state', world_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    connected_clients.discard(request.sid)
    print(f"Client disconnected: {request.sid}")

def get_world_state():
    """Get current world state from Redis"""
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
            
        # Get missed pickup alert
        missed_alert = redis_client.get('missed_pickup:alert')
        if missed_alert:
            state['missed_pickup_alert'] = True
            
        # Get trigger flash state
        trigger_flash = redis_client.get('trigger:flash')
        if trigger_flash:
            state['trigger_flash'] = True
            
        # Get monitor flash state
        monitor_flash = redis_client.get('monitor:flash')
        if monitor_flash:
            state['monitor_flash'] = True
            
    except Exception as e:
        print(f"Error reading world state: {e}")
        
    return state

def update_loop():
    """Background thread to push updates to clients"""
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
    """Listen for Redis events"""
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


@socketio.on('clear_missed_alert')
def handle_clear_missed_alert():
    """Clear the missed pickup alert"""
    redis_client.delete('missed_pickup:alert')

if __name__ == '__main__':
    start_background_threads()
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), allow_unsafe_werkzeug=True)
