#!/usr/bin/env python3
"""
GRBL Interface Agent - Simulates FoxAlien 3S GRBL Controller
Handles serial communication simulation, G-code processing, and status reporting
"""

import time
import threading
import queue
import re
from enum import Enum
from base_agent import BaseAgent

class GrblState(Enum):
    IDLE = "Idle"
    RUN = "Run"
    HOLD = "Hold"
    ALARM = "Alarm"
    DOOR = "Door"
    CHECK = "Check"
    HOME = "Home"
    SLEEP = "Sleep"

class GrblError(Enum):
    NONE = 0
    EXPECTED_COMMAND_LETTER = 1
    BAD_NUMBER_FORMAT = 2
    INVALID_STATEMENT = 3
    NEGATIVE_VALUE = 4
    SETTING_DISABLED = 5
    SETTING_STEP_PULSE_MIN = 6
    SETTING_READ_FAIL = 7
    IDLE_ERROR = 8
    SYSTEM_GC_LOCK = 9
    SOFT_LIMIT_ERROR = 10
    OVERFLOW = 11
    MAX_STEP_RATE_EXCEEDED = 12
    CHECK_DOOR = 13
    LINE_LENGTH_EXCEEDED = 14
    TRAVEL_EXCEEDED = 15
    INVALID_JOG_COMMAND = 16
    SETTING_DISABLED_LASER = 17
    HOMING_DISABLED = 18
    MAX_CHARACTERS_PER_LINE = 20
    MAX_GRBL_SETTINGS_EXCEEDED = 21
    SAFETY_DOOR_DETECTED = 22
    
class GrblInterfaceAgent(BaseAgent):
    """
    GRBL Interface Agent that simulates a FoxAlien 3S controller
    Provides serial-like interface for G-code commands and status reporting
    """
    
    def __init__(self):
        super().__init__(
            name="grbl_interface",
            subscribe_channels=["events:movement"],
            publish_channel="events:grbl"
        )
        
        # GRBL simulation state
        self.state = GrblState.IDLE
        self.version = "1.1h"  # FoxAlien typical firmware
        self.build_info = "['$' for help]"
        
        # Position tracking (machine coordinates)
        self.mpos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        # Work coordinate offset
        self.wco = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        # Work position (mpos - wco)
        self.wpos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # Machine limits (FoxAlien 3S typical workspace)
        self.limits = {
            'x': {'min': 0, 'max': 400},
            'y': {'min': 0, 'max': 400}, 
            'z': {'min': -100, 'max': 0}
        }
        
        # Feed rate and spindle
        self.feed_rate = 0
        self.spindle_speed = 0
        self.spindle_state = False
        
        # Command queue and response queue
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Buffer tracking for character counting protocol
        self.rx_buffer_size = 128  # GRBL RX buffer size
        self.rx_buffer_used = 0
        
        # Settings ($$ parameters)
        self.settings = self._initialize_settings()
        
        # Status report configuration
        self.status_report_mask = 1  # WPos enabled by default
        self.status_interval = 0.1  # 10Hz max status rate
        self.last_status_time = 0
        
        # Motion planning
        self.motion_queue = []
        self.current_motion = None
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.processing_thread.start()
        
    def _initialize_settings(self):
        """Initialize GRBL settings with FoxAlien 3S defaults"""
        return {
            0: 10,      # Step pulse time, microseconds
            1: 25,      # Step idle delay, milliseconds
            2: 0,       # Step pulse invert, mask
            3: 0,       # Step direction invert, mask
            4: 0,       # Invert step enable pin, boolean
            5: 0,       # Invert limit pins, boolean
            6: 0,       # Invert probe pin, boolean
            10: 1,      # Status report options
            11: 0.010,  # Junction deviation, millimeters
            12: 0.002,  # Arc tolerance, millimeters
            13: 0,      # Report in inches, boolean
            20: 0,      # Soft limits enable, boolean
            21: 0,      # Hard limits enable, boolean
            22: 1,      # Homing cycle enable, boolean
            23: 0,      # Homing direction invert, mask
            24: 25.0,   # Homing locate feed rate, mm/min
            25: 500.0,  # Homing search feed rate, mm/min
            26: 250,    # Homing switch debounce delay, milliseconds
            27: 1.0,    # Homing switch pull-off distance, millimeters
            30: 1000,   # Maximum spindle speed, RPM
            31: 0,      # Minimum spindle speed, RPM
            32: 0,      # Laser mode enable, boolean
            100: 80.0,  # X-axis steps per millimeter
            101: 80.0,  # Y-axis steps per millimeter
            102: 400.0, # Z-axis steps per millimeter
            110: 5000.0, # X-axis max rate, mm/min
            111: 5000.0, # Y-axis max rate, mm/min
            112: 500.0,  # Z-axis max rate, mm/min
            120: 200.0,  # X-axis acceleration, mm/sec^2
            121: 200.0,  # Y-axis acceleration, mm/sec^2
            122: 30.0,   # Z-axis acceleration, mm/sec^2
            130: 400.0,  # X-axis max travel, millimeters
            131: 400.0,  # Y-axis max travel, millimeters
            132: 100.0   # Z-axis max travel, millimeters
        }
    
    def handle_message(self, channel, message):
        """Handle incoming messages from movement agent"""
        event_type = message.get('event')
        data = message.get('data', {})
        
        if event_type == 'send_gcode':
            # Movement agent sends G-code command
            gcode = data.get('command')
            if gcode:
                self.send_command(gcode)
                
        elif event_type == 'request_status':
            # Movement agent requests current status
            self._send_status_report()
            
        elif event_type == 'emergency_stop':
            # Emergency stop request
            self._handle_reset()
    
    def send_command(self, command):
        """Add command to processing queue (simulates serial write)"""
        self.command_queue.put(command.strip())
        self.logger.debug(f"Queued command: {command.strip()}")
    
    def get_response(self, timeout=0.1):
        """Get response from GRBL (simulates serial read)"""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _process_commands(self):
        """Main command processing loop (runs in separate thread)"""
        # Send startup message
        time.sleep(0.5)  # Simulate boot time
        self._send_response(f"\r\nGrbl {self.version} {self.build_info}\r\n")
        
        while True:
            try:
                # Check for pending commands
                try:
                    command = self.command_queue.get(timeout=0.01)
                    self._execute_command(command)
                except queue.Empty:
                    pass
                
                # Process ongoing motion if any
                if self.state == GrblState.RUN and self.current_motion:
                    self._update_motion()
                    
                # Send periodic status reports if requested
                self._check_status_report()
                    
            except Exception as e:
                self.logger.error(f"Command processing error: {e}")
    
    def _execute_command(self, command):
        """Execute a single GRBL command"""
        self.logger.debug(f"Executing: {command}")
        
        # Real-time commands (single character)
        if len(command) == 1:
            if command == '?':
                self._send_status_report()
                return
            elif command == '~':  # Cycle start
                if self.state == GrblState.HOLD:
                    self.state = GrblState.RUN
                return
            elif command == '!':  # Feed hold
                if self.state == GrblState.RUN:
                    self.state = GrblState.HOLD
                return
            elif command == '\x18':  # Ctrl-X reset
                self._handle_reset()
                return
        
        # Check for empty command
        if not command:
            self._send_response("ok")
            return
            
        # System commands ($)
        if command.startswith('$'):
            self._handle_system_command(command)
            return
            
        # G-code commands
        try:
            self._parse_and_execute_gcode(command)
            self._send_response("ok")
        except Exception as e:
            error_code = self._get_error_code(str(e))
            self._send_response(f"error:{error_code}")
    
    def _handle_system_command(self, command):
        """Handle GRBL system commands"""
        if command == '$':
            # Help
            self._send_response("[HLP:$$ $# $G $I $N $x=val $Nx=line $J=line $SLP $C $X $H ~ ! ? ctrl-x]")
            self._send_response("ok")
            
        elif command == '$$':
            # View settings
            for key, value in sorted(self.settings.items()):
                self._send_response(f"${key}={value}")
            self._send_response("ok")
            
        elif command == '$#':
            # View parameters
            self._send_response(f"[G54:{self.wco['x']},{self.wco['y']},{self.wco['z']}]")
            self._send_response("[G55:0.000,0.000,0.000]")
            self._send_response("[G56:0.000,0.000,0.000]")
            self._send_response("[G57:0.000,0.000,0.000]")
            self._send_response("[G58:0.000,0.000,0.000]")
            self._send_response("[G59:0.000,0.000,0.000]")
            self._send_response("[G28:0.000,0.000,0.000]")
            self._send_response("[G30:0.000,0.000,0.000]")
            self._send_response("[G92:0.000,0.000,0.000]")
            self._send_response("[TLO:0.000]")
            self._send_response("[PRB:0.000,0.000,0.000:0]")
            self._send_response("ok")
            
        elif command == '$G':
            # Parser state
            self._send_response("[GC:G0 G54 G17 G21 G90 G94 M5 M9 T0 F0 S0]")
            self._send_response("ok")
            
        elif command == '$I':
            # Build info
            self._send_response(f"[VER:{self.version}:FoxAlien 3S]")
            self._send_response("[OPT:V,15,128]")
            self._send_response("ok")
            
        elif command == '$H':
            # Home
            self.state = GrblState.HOME
            self._perform_homing()
            
        elif command == '$X':
            # Kill alarm lock
            if self.state == GrblState.ALARM:
                self.state = GrblState.IDLE
            self._send_response("ok")
            
        elif re.match(r'\$\d+=', command):
            # Setting value
            match = re.match(r'\$(\d+)=(.+)', command)
            if match:
                setting_num = int(match.group(1))
                value = float(match.group(2))
                if setting_num in self.settings:
                    self.settings[setting_num] = value
                    self._send_response("ok")
                else:
                    self._send_response("error:3")
            else:
                self._send_response("error:3")
        else:
            self._send_response("error:3")
    
    def _parse_and_execute_gcode(self, command):
        """Parse and execute G-code command"""
        # For now, simulate basic moves
        if self.state != GrblState.IDLE and self.state != GrblState.RUN:
            raise Exception("Machine not ready")
            
        # Extract G-code components
        tokens = re.findall(r'[A-Z][-+]?\d*\.?\d*', command.upper())
        
        for token in tokens:
            letter = token[0]
            value = token[1:] if len(token) > 1 else ''
            
            if letter == 'G':
                code = int(float(value))
                if code == 0:  # Rapid move
                    self._plan_motion(command, rapid=True)
                elif code == 1:  # Linear move
                    self._plan_motion(command, rapid=False)
                elif code == 4:  # Dwell
                    dwell_time = self._extract_value(command, 'P', 0) / 1000.0
                    time.sleep(dwell_time)
                elif code == 28:  # Home
                    self._perform_homing()
                elif code == 90:  # Absolute mode
                    pass  # Already default
                elif code == 91:  # Relative mode
                    pass  # TODO: Implement
                    
            elif letter == 'M':
                code = int(float(value))
                if code == 3:  # Spindle on CW
                    self.spindle_state = True
                    s_value = self._extract_value(command, 'S', 1000)
                    self.spindle_speed = s_value
                elif code == 5:  # Spindle off
                    self.spindle_state = False
                    self.spindle_speed = 0
                elif code == 8:  # Coolant on
                    pass
                elif code == 9:  # Coolant off
                    pass
    
    def _plan_motion(self, command, rapid=False):
        """Plan a motion command"""
        # Extract target position
        target = self.mpos.copy()
        
        x_val = self._extract_value(command, 'X')
        y_val = self._extract_value(command, 'Y')
        z_val = self._extract_value(command, 'Z')
        f_val = self._extract_value(command, 'F')
        
        if x_val is not None:
            target['x'] = x_val
        if y_val is not None:
            target['y'] = y_val
        if z_val is not None:
            target['z'] = z_val
        if f_val is not None:
            self.feed_rate = f_val
            
        # Check limits
        for axis in ['x', 'y', 'z']:
            if target[axis] < self.limits[axis]['min'] or target[axis] > self.limits[axis]['max']:
                raise Exception("Soft limit error")
                
        # Add to motion queue
        motion = {
            'target': target,
            'start': self.mpos.copy(),
            'rapid': rapid,
            'feed_rate': self.settings[110] if rapid else self.feed_rate,
            'start_time': time.time(),
            'duration': self._calculate_motion_time(self.mpos, target, rapid)
        }
        
        self.motion_queue.append(motion)
        
        # Start motion if idle
        if self.state == GrblState.IDLE:
            self.state = GrblState.RUN
            self.current_motion = self.motion_queue.pop(0)
    
    def _update_motion(self):
        """Update ongoing motion simulation"""
        if not self.current_motion:
            return
            
        elapsed = time.time() - self.current_motion['start_time']
        duration = self.current_motion['duration']
        
        if elapsed >= duration:
            # Motion complete
            self.mpos = self.current_motion['target'].copy()
            self._update_work_position()
            
            # Check for next motion
            if self.motion_queue:
                self.current_motion = self.motion_queue.pop(0)
                self.current_motion['start_time'] = time.time()
            else:
                self.current_motion = None
                self.state = GrblState.IDLE
        else:
            # Interpolate position
            progress = elapsed / duration
            for axis in ['x', 'y', 'z']:
                start = self.current_motion['start'][axis]
                target = self.current_motion['target'][axis]
                self.mpos[axis] = start + (target - start) * progress
            self._update_work_position()
    
    def _calculate_motion_time(self, start, target, rapid=False):
        """Calculate time for motion"""
        # Calculate distance
        distance = 0
        for axis in ['x', 'y', 'z']:
            distance += (target[axis] - start[axis]) ** 2
        distance = distance ** 0.5
        
        if distance == 0:
            return 0
            
        # Get feed rate
        if rapid:
            feed_rate = min(self.settings[110], self.settings[111], self.settings[112])
        else:
            feed_rate = self.feed_rate if self.feed_rate > 0 else 100
            
        # Convert mm/min to mm/sec
        feed_rate_mms = feed_rate / 60.0
        
        # Calculate time
        return distance / feed_rate_mms
    
    def _perform_homing(self):
        """Simulate homing sequence"""
        self.state = GrblState.HOME
        
        # Simulate homing time
        time.sleep(2.0)
        
        # Set to home position
        self.mpos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self._update_work_position()
        
        self.state = GrblState.IDLE
        self._send_response("ok")
    
    def _send_status_report(self):
        """Send current status report"""
        # Format: <State|MPos:x,y,z|WPos:x,y,z|...>
        status = f"<{self.state.value}"
        
        # Add positions based on status mask
        if self.settings[10] & 1:  # WPos
            status += f"|WPos:{self.wpos['x']:.3f},{self.wpos['y']:.3f},{self.wpos['z']:.3f}"
        else:  # MPos
            status += f"|MPos:{self.mpos['x']:.3f},{self.mpos['y']:.3f},{self.mpos['z']:.3f}"
            
        # Add buffer state
        status += f"|Bf:{self.rx_buffer_size - self.rx_buffer_used},{self.rx_buffer_size}"
        
        # Add feed and speeds
        if self.state == GrblState.RUN:
            status += f"|FS:{self.feed_rate},{self.spindle_speed}"
            
        status += ">"
        self._send_response(status)
        self.last_status_time = time.time()
    
    def _check_status_report(self):
        """Check if periodic status report needed"""
        # This would be triggered by external status request interval
        pass
    
    def _update_work_position(self):
        """Update work position from machine position"""
        for axis in ['x', 'y', 'z']:
            self.wpos[axis] = self.mpos[axis] - self.wco[axis]
    
    def _extract_value(self, command, letter, default=None):
        """Extract numeric value for given letter from G-code"""
        pattern = letter + r'([-+]?\d*\.?\d+)'
        match = re.search(pattern, command.upper())
        if match:
            return float(match.group(1))
        return default
    
    def _get_error_code(self, error_msg):
        """Map error message to GRBL error code"""
        if "soft limit" in error_msg.lower():
            return GrblError.SOFT_LIMIT_ERROR.value
        elif "not ready" in error_msg.lower():
            return GrblError.IDLE_ERROR.value
        else:
            return GrblError.INVALID_STATEMENT.value
    
    def _send_response(self, response):
        """Send response to response queue"""
        self.response_queue.put(response)
        self.logger.debug(f"Response: {response}")
        
        # Also publish to Redis for monitoring
        self.publish('grbl_response', {'response': response})
    
    def _handle_reset(self):
        """Handle soft reset"""
        self.logger.info("Soft reset triggered")
        
        # Clear queues
        while not self.command_queue.empty():
            self.command_queue.get()
        self.motion_queue.clear()
        self.current_motion = None
        
        # Reset state
        self.state = GrblState.IDLE
        self.feed_rate = 0
        self.spindle_speed = 0
        self.spindle_state = False
        
        # Send reset message
        self._send_response(f"\r\nGrbl {self.version} {self.build_info}\r\n")
    
    def get_serial_interface(self):
        """
        Get a serial-like interface for external use
        Returns object with write() and readline() methods
        """
        class SerialInterface:
            def __init__(self, agent):
                self.agent = agent
                
            def write(self, data):
                """Write command to GRBL"""
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                self.agent.send_command(data.strip())
                
            def readline(self):
                """Read response from GRBL"""
                response = self.agent.get_response(timeout=0.1)
                if response:
                    return (response + '\r\n').encode('utf-8')
                return b''
                
            def read(self, size=1):
                """Read bytes from GRBL"""
                response = self.agent.get_response(timeout=0.01)
                if response:
                    return response[:size].encode('utf-8')
                return b''
                
            def in_waiting(self):
                """Check if data is waiting"""
                return not self.agent.response_queue.empty()
                
            def flush(self):
                """Flush buffers"""
                pass
                
            def close(self):
                """Close connection"""
                pass
                
        return SerialInterface(self)

def main():
    """Run GRBL interface agent standalone"""
    agent = GrblInterfaceAgent()
    agent.logger.info("GRBL Interface Agent started")
    
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.logger.info("GRBL Interface Agent shutting down")

if __name__ == "__main__":
    main()