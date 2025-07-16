import os
import json
import redis
import time
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from datetime import datetime

# Note: This application is managed by Supervisor service manager
# Services: pick1_webapp, pick1_cnc, pick1_vision, pick1_trigger, pick1_scoring, pick1_monitor, pick1_test_data, pick1_redis
# Use: sudo supervisorctl status|start|stop|restart pick1:service_name
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")

# Redis connection
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url, decode_responses=True)
redis_pubsub = redis_client.pubsub()

# Global state
connected_clients = set()
status_messages = []  # Store recent status messages
update_thread = None
update_thread_stop = threading.Event()

def add_status_message(agent, message):
    """Add a status message to the global list"""
    global status_messages
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_msg = {
        'timestamp': timestamp,
        'agent': agent,
        'message': message
    }
    status_messages.append(status_msg)
    print(f"DEBUG: Added status message: {agent}: {message} (total: {len(status_messages)})")
    
    # Keep only last 50 messages
    if len(status_messages) > 50:
        status_messages = status_messages[-50:]
    
    # Emit to all connected clients
    if connected_clients:
        socketio.emit('status_message', status_msg)
        print(f"DEBUG: Emitted to {len(connected_clients)} clients")

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

# API FUNCTIONS - ON HOLD
# Commenting out API endpoints to focus on WebSocket functionality
# TODO: Restore API endpoints when needed for debugging/monitoring

# @app.route('/api/status')
# def api_status():
#     """API endpoint to check current system status - ON HOLD"""
#     world_state = get_world_state()
#     response = {
#         'objects_count': len(world_state.get('objects', {})),
#         'cnc_status': world_state.get('cnc', {}).get('status', 'unknown'),
#         'status_messages': status_messages[-5:],  # Last 5 messages
#         'subscribers': {
#             'cnc': redis_client.pubsub_numsub('events:cnc')[0][1],
#             'vision': redis_client.pubsub_numsub('events:vision')[0][1],
#             'scoring': redis_client.pubsub_numsub('events:scoring')[0][1],
#             'trigger': redis_client.pubsub_numsub('events:trigger')[0][1]
#         }
#     }
#     
#     # Include confirmed target if available
#     if 'confirmed_target' in world_state:
#         response['confirmed_target'] = world_state['confirmed_target']
#     
#     return response

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
        
        # Get assigned lane number for CNC display
        assigned_lane = redis_client.get('cnc:assigned_lane')
        print(f"DEBUG: assigned_lane from Redis: {assigned_lane}")
        if assigned_lane:
            if 'cnc' not in state:
                state['cnc'] = {}
            state['cnc']['assigned_lane'] = int(assigned_lane)
            print(f"DEBUG: Set assigned_lane in state: {state['cnc']['assigned_lane']}")
        
        # Get confirmed target for top screen display
        confirmed_target = redis_client.get('scoring:confirmed_target')
        if confirmed_target:
            try:
                object_id, lane = confirmed_target.split(':')
                state['confirmed_target'] = {
                    'object_id': object_id,
                    'lane': int(lane)
                }
            except ValueError:
                pass
        
        # Add recent status messages to state
        state['status_messages'] = status_messages[-20:]  # Last 20 messages
        
        # Get bin state
        bin_flash = redis_client.get('bin:0:flash')
        if bin_flash:
            state['bin_flash'] = True
            # READ-ONLY MODE: Comment out Redis writes for debugging
            # redis_client.delete('bin:0:flash')  # UNCOMMENT for normal operation
            
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
    last_cnc_status = None
    
    while not update_thread_stop.is_set():
        try:
            # Get current state
            current_state = get_world_state()
            
            # Check for CNC status changes and emit status messages
            if current_state.get('cnc'):
                current_cnc_status = current_state['cnc'].get('status', 'idle')
                if current_cnc_status != last_cnc_status and current_cnc_status != 'idle':
                    # Map CNC statuses to readable messages
                    status_map = {
                        'moving_to_pickup': 'Moving to pickup position',
                        'picking': 'Picking object',
                        'moving_to_bin': 'Moving to bin',
                        'dropping': 'Dropping object',
                        'homing': 'Returning home'
                    }
                    message = status_map.get(current_cnc_status, current_cnc_status.replace('_', ' ').title())
                    add_status_message('cnc', message)
                last_cnc_status = current_cnc_status
            
            # Check if state changed
            if current_state != last_state and connected_clients:
                # Broadcast to all connected clients
                socketio.emit('world_update', current_state)
                print(f"DEBUG: Broadcasted world_update to {len(connected_clients)} clients")
                last_state = current_state
            elif connected_clients:
                # Force broadcast every 10 cycles even if no change (for debugging)
                if hasattr(update_loop, 'cycle_count'):
                    update_loop.cycle_count += 1
                else:
                    update_loop.cycle_count = 1
                
                if update_loop.cycle_count % 50 == 0:  # Every 10 seconds at 5Hz
                    socketio.emit('world_update', current_state)
                    print(f"DEBUG: Force broadcasted world_update to {len(connected_clients)} clients")
                
            # Update at ~10Hz for smoother animation
            time.sleep(0.1)
            
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
            'events:system',
            'events:vision',
            'events:monitor',
            'events:cnc',
            'events:scoring'
        ])
        
        for message in redis_pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    event_type = data.get('event')
                    
                    # Handle vision detection events
                    if message['channel'] == 'events:vision' and event_type == 'object_detected':
                        detection_data = data.get('data', {})
                        obj_id = detection_data.get('id')
                        obj_type = detection_data.get('type')
                        lane = detection_data.get('lane', 0)
                        position = detection_data.get('position_x', 0)
                        
                        # Add status message for detection
                        add_status_message('vision', f"Detected {obj_type} object {obj_id} in lane {lane} at {position}mm")
                        
                        # If it's a green object, add yellow ring
                        if obj_type == 'green' and obj_id:
                            # READ-ONLY MODE: Comment out Redis writes for debugging
                            # redis_client.hset(f'object:{obj_id}', 'has_ring', 'true')  # UNCOMMENT for normal operation
                            # redis_client.hset(f'object:{obj_id}', 'ring_color', 'yellow')  # UNCOMMENT for normal operation
                            add_status_message('vision', f"Added yellow ring to pickable object {obj_id}")
                            print(f"Vision: Added yellow ring to {obj_id}")
                    
                    # Handle CNC pickup assignment events
                    elif message['channel'] == 'events:cnc' and event_type == 'pickup_assignment':
                        assignment_data = data.get('data', {})
                        lane = assignment_data.get('lane')
                        obj_id = assignment_data.get('object_id')
                        cnc_id = assignment_data.get('cnc_id', 'cnc:0')
                        
                        if lane is not None:
                            # Store lane number for CNC display
                            # READ-ONLY MODE: Comment out Redis writes for debugging
                            # redis_client.set('cnc:assigned_lane', str(lane))  # UNCOMMENT for normal operation
                            add_status_message('cnc', f"Assigned to pickup {obj_id} from lane {lane}")
                            print(f"CNC: Assigned lane {lane} displayed on CNC icon")
                    
                    # Handle CNC ready events
                    elif message['channel'] == 'events:cnc' and event_type == 'ready_for_assignment':
                        cnc_data = data.get('data', {})
                        cnc_id = cnc_data.get('cnc_id', 'cnc:0')
                        add_status_message('cnc', f"Ready for new assignment")
                    
                    # Handle CNC pickup events - remove object from belt
                    elif message['channel'] == 'events:cnc' and event_type == 'object_picked':
                        pickup_data = data.get('data', {})
                        obj_id = pickup_data.get('object_id')
                        cnc_id = pickup_data.get('cnc_id', 'cnc:0')
                        
                        if obj_id:
                            # READ-ONLY MODE: Comment out Redis writes for debugging
                            # Remove object from belt (objects:active and object hash)
                            # redis_client.zrem('objects:active', obj_id)  # UNCOMMENT for normal operation
                            # redis_client.delete(f'object:{obj_id}')  # UNCOMMENT for normal operation
                            
                            # Clear confirmed target if this was the assigned object
                            confirmed_target = redis_client.get('scoring:confirmed_target')
                            if confirmed_target and confirmed_target.startswith(f'{obj_id}:'):
                                # redis_client.delete('scoring:confirmed_target')  # UNCOMMENT for normal operation
                                print(f"App: Cleared confirmed target for picked object {obj_id}")
                            
                            add_status_message('cnc', f"Picked up object {obj_id}")
                            print(f"App: Removed picked object {obj_id} from belt")
                    
                    # Handle trigger camera events - flash trigger line when object approaches
                    elif message['channel'] == 'events:trigger' and event_type == 'object_approaching':
                        trigger_data = data.get('data', {})
                        obj_id = trigger_data.get('object_id')
                        lane = trigger_data.get('lane')
                        position = trigger_data.get('current_position')
                        
                        # Flash trigger line yellow when object detected at trigger line
                        # READ-ONLY MODE: Comment out Redis writes for debugging
                        # redis_client.setex('trigger:flash', 1, 'true')  # UNCOMMENT for normal operation
                        add_status_message('trigger', f"Object {obj_id} approaching in lane {lane} at {position}mm")
                    
                    # Handle scoring agent target confirmation events
                    elif message['channel'] == 'events:scoring' and event_type == 'target_confirmed':
                        confirmation_data = data.get('data', {})
                        object_id = confirmation_data.get('object_id')
                        lane = confirmation_data.get('lane')
                        
                        if object_id and lane is not None:
                            # Store confirmation for top screen display
                            # READ-ONLY MODE: Comment out Redis writes for debugging
                            # redis_client.set('scoring:confirmed_target', f"{object_id}:{lane}")  # UNCOMMENT for normal operation
                            add_status_message('scoring', f"Confirmed target {object_id} in lane {lane}")
                            print(f"Scoring: Confirmed target {object_id} in lane {lane}")
                    
                    # Handle trigger agent events
                    elif message['channel'] == 'events:trigger':
                        if event_type == 'object_approaching':
                            trigger_data = data.get('data', {})
                            obj_id = trigger_data.get('object_id')
                            lane = trigger_data.get('lane', 0)
                            add_status_message('trigger', f"Object {obj_id} approaching in lane {lane}")
                        elif event_type in ['pick_operation_start', 'pick_operation_complete']:
                            # Remove lane number display when pick operation occurs
                            # READ-ONLY MODE: Comment out Redis writes for debugging
                            # redis_client.delete('cnc:assigned_lane')  # UNCOMMENT for normal operation
                            # redis_client.delete('scoring:confirmed_target')  # UNCOMMENT for normal operation
                            add_status_message('trigger', f"Pick operation {event_type.replace('_', ' ')}")
                            print(f"CNC: Removed lane display - pick operation {event_type}")
                        elif event_type == 'watch_for_object':
                            watch_data = data.get('data', {})
                            obj_id = watch_data.get('object_id')
                            lane = watch_data.get('lane', 0)
                            add_status_message('trigger', f"Watching for {obj_id} in lane {lane}")
                    
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
    # READ-ONLY MODE: Comment out Redis writes for debugging
    # redis_client.delete('missed_pickup:alert')  # UNCOMMENT for normal operation
    pass

# Start background threads when app is imported (for Gunicorn)
start_background_threads()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True, allow_unsafe_werkzeug=True)
