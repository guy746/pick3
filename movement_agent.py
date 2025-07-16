#!/usr/bin/env python3
"""
Movement Agent - Integrated YAML-Based CNC Control
Handles own route management with GRBL interface control
"""

import time
import threading
import json
import yaml
import re
import os
from base_agent import BaseAgent
from grbl_interface_agent import GrblInterfaceAgent

class MovementAgent(BaseAgent):
    """
    Movement Agent with integrated route management
    Controls CNC operations through GRBL interface using YAML-defined routes
    """
    
    def __init__(self, routes_file="gcode_routes.yaml"):
        super().__init__(
            name="movement",
            subscribe_channels=["events:cnc", "events:trigger", "events:grbl"],
            publish_channel="events:movement"
        )
        
        # Load routes configuration
        self.routes_file = routes_file
        self.config = self._load_routes()
        
        # Initialize GRBL interface
        self.grbl = GrblInterfaceAgent()
        self.grbl_serial = self.grbl.get_serial_interface()
        
        # Start GRBL interface in separate thread
        self.grbl_thread = threading.Thread(target=self.grbl.run, daemon=True)
        self.grbl_thread.start()
        
        # Wait for GRBL to initialize
        time.sleep(1.0)
        self._initialize_grbl()
        
        # Movement state
        self.current_assignment = None
        self.assignment_lock = threading.Lock()
        self.position = {'x': 0, 'y': 0, 'z': 0}
        self.state = 'idle'
        
        # Command execution
        self.command_timeout = 10.0
        self.status_interval = 0.1
        
        # Start status monitor
        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()
    
    def _load_routes(self):
        """Load routes configuration from YAML file"""
        config_path = os.path.join(os.path.dirname(__file__), self.routes_file)
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.logger.info(f"Loaded routes from {config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Routes file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")
    
    def _substitute_parameters(self, text, **params):
        """Substitute parameters in text"""
        if not text:
            return text
            
        # Create substitution context
        context = {
            **self.config,  # Include all config
            **params        # Override with provided params
        }
        
        # Replace {key.subkey} patterns
        def replace_nested(match):
            key_path = match.group(1)
            try:
                # Navigate nested keys
                value = context
                for key in key_path.split('.'):
                    value = value[key]
                return str(value)
            except (KeyError, TypeError):
                return match.group(0)
        
        pattern = r'\{([^}]+)\}'
        return re.sub(pattern, replace_nested, text)
    
    def _format_gcode(self, gcode, params):
        """Format G-code command"""
        if params:
            return f"{gcode} {params}"
        return gcode
    
    def _get_route(self, route_name, **params):
        """Get route with parameter substitution"""
        # Check sequences first
        if route_name in self.config.get('sequences', {}):
            return self._process_sequence(route_name, **params)
            
        # Check regular routes
        if route_name not in self.config.get('routes', {}):
            raise ValueError(f"Route '{route_name}' not found")
            
        route = self.config['routes'][route_name]
        return self._process_route(route, **params)
    
    def _process_route(self, route, **params):
        """Process a single route"""
        commands = []
        
        for cmd in route['commands']:
            gcode = cmd['gcode']
            param_str = self._substitute_parameters(cmd.get('params', ''), **params)
            description = self._substitute_parameters(cmd.get('description', ''), **params)
            commands.append((gcode, param_str, description))
        
        return {
            'commands': commands,
            'announce_start': self._substitute_parameters(route.get('announce_start', ''), **params),
            'announce_complete': self._substitute_parameters(route.get('announce_complete', ''), **params)
        }
    
    def _process_sequence(self, sequence_name, **params):
        """Process a sequence of routes"""
        sequence = self.config['sequences'][sequence_name]
        
        result = {
            'commands': [],
            'announce_start': self._substitute_parameters(sequence.get('announce_start', ''), **params),
            'announce_complete': self._substitute_parameters(sequence.get('announce_complete', ''), **params)
        }
        
        # Process individual commands if any
        if 'commands' in sequence:
            route_data = self._process_route({'commands': sequence['commands']}, **params)
            result['commands'].extend(route_data['commands'])
        
        # Process sub-routes
        if 'routes' in sequence:
            for route_ref in sequence['routes']:
                route_name = route_ref['route']
                route_params = route_ref.get('params', {})
                
                # Substitute parameters in route params
                processed_params = {}
                for key, value in route_params.items():
                    if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                        param_name = value[1:-1]
                        processed_params[key] = params.get(param_name, value)
                    else:
                        processed_params[key] = value
                
                # Merge with provided params
                merged_params = {**params, **processed_params}
                sub_route = self._get_route(route_name, **merged_params)
                result['commands'].extend(sub_route['commands'])
                
        return result
    
    def _initialize_grbl(self):
        """Initialize GRBL connection and settings"""
        self.logger.info("Initializing GRBL interface...")
        self._announce("GRBL interface initializing...")
        
        # Clear any startup messages
        time.sleep(0.5)
        while self.grbl_serial.in_waiting():
            response = self.grbl_serial.readline().decode().strip()
            self.logger.debug(f"GRBL startup: {response}")
        
        # Send soft reset
        self._send_command("\x18", wait_for_ok=False)
        time.sleep(2)
        
        # Clear reset response
        while self.grbl_serial.in_waiting():
            self.grbl_serial.readline()
            
        # Check GRBL version
        self._send_command("$I")
        
        # Home the machine
        self._announce("Homing machine...")
        self._send_command("$H")
        
        # Set to absolute mode
        self._send_command("G90")
        
        # Set initial feed rate
        self._send_command("G1 F5000")
        
        # Move to ready position
        self._execute_route("ready")
        
        self._announce("GRBL interface ready - system initialized")
        self.logger.info("GRBL interface initialized")
        
    def handle_message(self, channel, message):
        """Handle incoming messages"""
        event_type = message.get('event')
        data = message.get('data', {})
        
        if channel == 'events:cnc':
            if event_type == 'pickup_assignment':
                self._handle_pickup_assignment(data)
                
        elif channel == 'events:trigger':
            if event_type == 'object_approaching':
                self._handle_trigger_notification(data)
                
        elif channel == 'events:grbl':
            if event_type == 'grbl_response':
                response = data.get('response', '')
                if response and not response.startswith('<'):
                    self.logger.debug(f"GRBL response: {response}")
    
    def _announce(self, message):
        """Publish announcement to Redis"""
        announcement = {
            'timestamp': time.time(),
            'agent': self.name,
            'message': message,
            'state': self.state
        }
        
        # Publish to movement events
        self.publish('announcement', announcement)
        
        # Also publish to general announcements channel
        self.redis.publish('events:announcements', json.dumps(announcement))
        
        # Log locally
        self.logger.info(f"ANNOUNCEMENT: {message}")
        print(f"[MOVEMENT] {message}")
    
    def _send_command(self, command, wait_for_ok=True):
        """Send command to GRBL and wait for response"""
        self.logger.debug(f"Sending: {command}")
        self.grbl_serial.write(command + '\n')
        
        if wait_for_ok:
            start_time = time.time()
            while time.time() - start_time < self.command_timeout:
                if self.grbl_serial.in_waiting():
                    response = self.grbl_serial.readline().decode().strip()
                    self.logger.debug(f"Received: {response}")
                    
                    if response == 'ok':
                        return True
                    elif response.startswith('error:'):
                        self.logger.error(f"GRBL error: {response}")
                        self._announce(f"GRBL error: {response}")
                        return False
                    elif response.startswith('[') or response.startswith('<'):
                        # Status or parameter response
                        continue
                        
                time.sleep(0.01)
                
            self.logger.error("Command timeout")
            self._announce("Command timeout - possible GRBL communication issue")
            return False
        
        return True
    
    def _execute_route(self, route_name, **params):
        """Execute a YAML-defined route"""
        try:
            # Get route data
            route_data = self._get_route(route_name, **params)
            
            # Announce start
            if route_data['announce_start']:
                self._announce(route_data['announce_start'])
            
            # Execute commands
            commands = route_data['commands']
            total_commands = len(commands)
            
            self.logger.info(f"Executing route '{route_name}' ({total_commands} commands)")
            
            for i, (gcode, gcode_params, description) in enumerate(commands, 1):
                # Format command
                command = self._format_gcode(gcode, gcode_params)
                
                self.logger.debug(f"[{i}/{total_commands}] {command} ; {description}")
                
                # Send to GRBL
                if not self._send_command(command):
                    self.logger.error(f"Failed to execute: {command}")
                    self._announce(f"Route execution failed at step {i}: {command}")
                    return False
                    
                # Handle dwell commands
                if gcode == 'G04' and 'P' in gcode_params:
                    dwell_match = re.search(r'P(\d+)', gcode_params)
                    if dwell_match:
                        dwell_ms = int(dwell_match.group(1))
                        time.sleep(dwell_ms / 1000.0)
            
            # Announce completion
            if route_data['announce_complete']:
                self._announce(route_data['announce_complete'])
                
            self.logger.info(f"Route '{route_name}' completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Route execution error: {e}")
            self._announce(f"Route execution failed: {route_name} - {str(e)}")
            return False
    
    def _get_status(self):
        """Get current status from GRBL"""
        self.grbl_serial.write('?')
        
        start_time = time.time()
        while time.time() - start_time < 0.5:
            if self.grbl_serial.in_waiting():
                response = self.grbl_serial.readline().decode().strip()
                if response.startswith('<') and response.endswith('>'):
                    return self._parse_status(response)
            time.sleep(0.01)
            
        return None
    
    def _parse_status(self, status_str):
        """Parse GRBL status report"""
        status = {}
        
        # Remove brackets
        status_str = status_str[1:-1]
        parts = status_str.split('|')
        
        # Parse state
        status['state'] = parts[0]
        
        # Parse other fields
        for part in parts[1:]:
            if ':' in part:
                key, value = part.split(':', 1)
                if key in ['WPos', 'MPos']:
                    coords = value.split(',')
                    if len(coords) >= 3:
                        status[key] = {
                            'x': float(coords[0]),
                            'y': float(coords[1]),
                            'z': float(coords[2])
                        }
                elif key == 'FS':
                    feeds = value.split(',')
                    if len(feeds) >= 2:
                        status['feed'] = float(feeds[0])
                        status['speed'] = float(feeds[1])
                        
        return status
    
    def _status_monitor(self):
        """Monitor GRBL status periodically"""
        while True:
            try:
                status = self._get_status()
                if status:
                    # Update position
                    if 'WPos' in status:
                        self.position = status['WPos']
                    elif 'MPos' in status:
                        self.position = status['MPos']
                        
                    # Update state
                    grbl_state = status.get('state', 'Unknown')
                    if grbl_state == 'Idle':
                        if self.state == 'moving':
                            self.state = 'idle'
                    elif grbl_state == 'Run':
                        self.state = 'moving'
                    elif grbl_state == 'Alarm':
                        self.logger.error("GRBL in alarm state!")
                        self._announce("ALARM: GRBL controller in alarm state!")
                        self.state = 'alarm'
                        
                    # Publish status update
                    self.publish('status_update', {
                        'position': self.position,
                        'state': self.state,
                        'grbl_state': grbl_state
                    })
                    
            except Exception as e:
                self.logger.error(f"Status monitor error: {e}")
                
            time.sleep(self.status_interval)
    
    def _handle_pickup_assignment(self, assignment_data):
        """Handle new pickup assignment"""
        obj_id = assignment_data.get('object_id')
        lane = assignment_data.get('lane')
        position = assignment_data.get('position')
        
        if not obj_id:
            return
            
        with self.assignment_lock:
            self.current_assignment = {
                'object_id': obj_id,
                'lane': lane,
                'position': position,
                'status': 'preparing'
            }
            
        self.logger.info(f"Received assignment for {obj_id} in lane {lane}")
        self._announce(f"New pickup assignment: Object {obj_id} in lane {lane}")
        
        # Update Redis state
        self._update_cnc_state('preparing')
        
        # Execute preparation route
        if self._execute_route('prepare_lane', lane=lane):
            with self.assignment_lock:
                self.current_assignment['status'] = 'waiting_for_trigger'
                
            self._update_cnc_state('waiting_for_trigger')
            
            # Request trigger watch
            self._request_trigger_watch(obj_id, lane)
        else:
            self.logger.error("Failed to prepare for pickup")
            self._cancel_assignment()
    
    def _handle_trigger_notification(self, trigger_data):
        """Handle trigger camera notification"""
        obj_id = trigger_data.get('object_id')
        
        with self.assignment_lock:
            if not self.current_assignment or self.current_assignment['object_id'] != obj_id:
                self.logger.warning(f"Ignoring trigger for {obj_id} (not my assignment)")
                return
                
            self.current_assignment['status'] = 'executing'
            
        # Calculate timing
        wait_time, pickup_x = self._calculate_pickup_timing(trigger_data)
        
        self.logger.info(f"Trigger received! Waiting {wait_time:.3f}s before pickup...")
        self._announce(f"Object {obj_id} detected - executing pickup in {wait_time:.1f}s")
        
        # Wait for optimal moment
        if wait_time > 0:
            time.sleep(wait_time)
            
        # Execute pickup sequence
        self._update_cnc_state('picking')
        
        if self._execute_route('pickup', pickup_x=pickup_x):
            # Remove object from belt
            self.redis.zrem('objects:active', obj_id)
            self.redis.delete(f'object:{obj_id}')
            
            # Transport and deliver
            self._update_cnc_state('moving_to_bin')
            
            if self._execute_route('transport_to_dropoff'):
                if self._execute_route('dropoff'):
                    # Flash bin
                    self.redis.setex('bin:0:flash', 1, 'true')
                    
                    # Return to ready
                    self._update_cnc_state('returning_home')
                    
                    if self._execute_route('ready'):
                        self._complete_assignment()
                        return
                        
        # If we get here, something failed
        self.logger.error("Pickup sequence failed")
        self._cancel_assignment()
    
    def _calculate_pickup_timing(self, trigger_data):
        """Calculate optimal pickup timing"""
        current_pos = trigger_data.get('current_position', 250)
        velocity = trigger_data.get('velocity', 133.33)
        
        # Get pickup position from config
        pickup_x = self.config['positions']['lanes'][f'lane_0']['x']  # Default pickup X
        
        distance = pickup_x - current_pos
        travel_time = distance / velocity if velocity > 0 else 0
        
        # Subtract pickup routine preparation time
        pickup_prep_time = 0.3
        wait_time = max(0, travel_time - pickup_prep_time)
        
        return wait_time, pickup_x
    
    def _request_trigger_watch(self, obj_id, lane):
        """Request trigger camera to watch for object"""
        obj_pos = float(self.redis.hget(f'object:{obj_id}', 'position_x') or 0)
        distance_to_trigger = 250 - obj_pos
        belt_speed = 50
        expected_time = distance_to_trigger / belt_speed if distance_to_trigger > 0 else 0
        
        event = {
            'event': 'watch_for_object',
            'timestamp': time.time(),
            'data': {
                'object_id': obj_id,
                'lane': lane,
                'timeout': expected_time + 1.0,
                'requested_by': 'movement_agent'
            }
        }
        
        self.redis.publish('events:trigger', json.dumps(event))
        self.logger.info(f"Requested trigger watch for {obj_id}")
        self._announce(f"Monitoring trigger line for object {obj_id}")
    
    def _update_cnc_state(self, status):
        """Update CNC state in Redis"""
        cnc_data = {
            'position_x': str(self.position['x']),
            'position_y': str(self.position['y']),
            'position_z': str(self.position['z']),
            'status': status,
            'has_object': 'true' if status in ['picking', 'moving_to_bin'] else 'false'
        }
        self.redis.hset('cnc:0', mapping=cnc_data)
    
    def _complete_assignment(self):
        """Complete current assignment"""
        with self.assignment_lock:
            if self.current_assignment:
                obj_id = self.current_assignment['object_id']
                self._announce(f"Assignment completed successfully: Object {obj_id} delivered")
                self.current_assignment = None
            
        self._update_cnc_state('idle')
        
        # Wait before accepting new assignment
        time.sleep(1.0)
        
        # Publish ready for assignment
        self.publish('ready_for_assignment', {
            'cnc_id': 'cnc:0',
            'position': self.position
        })
        
        # Also publish to CNC channel for compatibility
        event = {
            'event': 'ready_for_assignment',
            'timestamp': time.time(),
            'data': {
                'cnc_id': 'cnc:0',
                'position': self.position['x']
            }
        }
        self.redis.publish('events:cnc', json.dumps(event))
        self._announce("System ready for next assignment")
    
    def _cancel_assignment(self):
        """Cancel current assignment due to error"""
        with self.assignment_lock:
            if self.current_assignment:
                obj_id = self.current_assignment['object_id']
                self.logger.error(f"Canceling assignment for {obj_id}")
                self._announce(f"Assignment canceled due to error: Object {obj_id}")
                self.current_assignment = None
                
        # Try to recover using error recovery route
        try:
            self._announce("Attempting error recovery...")
            
            # Execute soft error recovery sequence
            if self._execute_route('emergency_ready'):
                self._announce("Error recovery completed")
            else:
                # Fallback to manual recovery commands
                self._send_command("$X")  # Clear alarm
                self._send_command("M5")  # Gripper off
                self._send_command("G28") # Home
                
        except Exception as e:
            self.logger.error(f"Error recovery failed: {e}")
            self._announce("Error recovery failed - manual intervention may be required")
        
        self._update_cnc_state('idle')
        
        # Ready for new assignment after recovery
        time.sleep(2.0)
        self._complete_assignment()
    
    def emergency_stop(self):
        """Emergency stop - halt all operations"""
        self._announce("EMERGENCY STOP ACTIVATED")
        self._send_command('!', wait_for_ok=False)  # Feed hold
        time.sleep(0.1)
        self._send_command('\x18', wait_for_ok=False)  # Soft reset
        
        with self.assignment_lock:
            self.current_assignment = None
            
        self.state = 'emergency_stop'
        self._announce("Emergency stop completed - system halted")

def main():
    """Run movement agent standalone"""
    agent = MovementAgent()
    agent.logger.info("Movement Agent started")
    
    # Publish initial ready state
    time.sleep(2.0)
    agent._complete_assignment()
    
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.logger.info("Movement Agent shutting down")
        agent._announce("Movement agent shutting down")

if __name__ == "__main__":
    main()