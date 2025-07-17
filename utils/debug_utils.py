import json
import time
import redis

# Redis connection for debug channel - using host.docker.internal for container-to-host communication
# Note: localhost won't work from inside Docker containers
r_debug = redis.Redis(host='host.docker.internal', port=6379, decode_responses=True)

def debug_event(agent_name, event_type, message, data=None):
    """Publish a debug event to Redis"""
    event = {
        'event': event_type,
        'agent': agent_name,
        'timestamp': time.time(),
        'message': message,
        'data': data or {}
    }
    r_debug.publish('events:debug', json.dumps(event))
    
def trace_action(agent_name, action, details=None):
    """Trace agent actions with detailed context"""
    debug_event(agent_name, 'action', action, details)

def trace_receive(agent_name, event_name, event_data):
    """Trace received events with full payload"""
    debug_event(agent_name, 'receive', f"Received {event_name}", event_data)
    
def trace_send(agent_name, event_name, event_data):
    """Trace sent events with full payload"""
    debug_event(agent_name, 'send', f"Sent {event_name}", event_data)
