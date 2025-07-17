# GRBL Movement System Documentation

## Overview
This document details the implementation of a comprehensive GRBL interface system for FoxAlien 3S CNC controller simulation, including YAML-based route management and Redis announcement integration.

## Redis Setup Configuration

### Memory-Only Configuration
The Pick1 system uses Redis as a memory-only data store for real-time event communication. No persistence is required as all data is transient simulation state.

**Supervisor Configuration** (`/etc/supervisor/conf.d/pick1.conf`):
```
[program:pick1_redis]
command=redis-server --daemonize no --save "" --stop-writes-on-bgsave-error no
```

**Key Settings**:
- `--save ""` - Disables RDB snapshot persistence completely
- `--stop-writes-on-bgsave-error no` - Prevents write blocking on save failures
- `--daemonize no` - Runs in foreground for Supervisor management

**Verification**:
```bash
redis-cli config get save                    # Should return empty
redis-cli config get stop-writes-on-bgsave-error  # Should return "no"
redis-cli ping                               # Should return "PONG"
```

### Event Channels Used
- `events:vision` - Object detection events
- `events:cnc` - CNC machine state and assignments
- `events:scoring` - Object prioritization and targeting
- `events:trigger` - Conveyor trigger zone events
- `events:movement` - GRBL movement commands
- `events:announcements` - System status announcements

## Architecture Change Summary

### Before
```
Movement Agent → GCode Route Manager → GRBL Interface Agent
                ↓
            Redis Events
```

### After
```
Movement Agent (Integrated Route Management) → GRBL Interface Agent
                ↓
            Redis Events + Announcements
```

## New Files Created

### 1. `grbl_interface_agent.py`
**Purpose**: Complete FoxAlien 3S GRBL 1.1 controller simulation

**Key Features**:
- Full GRBL 1.1 protocol implementation
- Serial-like communication interface
- FoxAlien 3S specific configuration (400x400x100mm workspace)
- Real-time command processing (G-code, system commands, real-time commands)
- Status reporting with position tracking
- Error handling with proper GRBL error codes
- Motion planning and simulation
- Settings management with FoxAlien 3S defaults

**GRBL Commands Supported**:
- **G-code**: G0/G1 (moves), G4 (dwell), G28 (home), G90/G91 (modes)
- **M-code**: M3/M5 (spindle), M8/M9 (coolant)
- **System**: $, $$, $#, $G, $I, $H, $X, $N=value
- **Real-time**: ? (status), ~ (resume), ! (hold), ^X (reset)

**Technical Specifications**:
- Firmware Version: GRBL 1.1h (FoxAlien typical)
- Buffer Size: 128 bytes RX buffer simulation
- Communication: 115200 baud equivalent
- Feed Rates: Configurable rapid/positioning/pickup speeds

### 2. `gcode_routes.yaml`
**Purpose**: Predefined G-code routes with announcements for pickup operations

**Structure**:
```yaml
machine:          # Machine configuration
positions:        # Named positions (ready, lanes 0-3, dropoff)
feed_rates:       # Speed settings for different operations
routes:           # Individual route definitions
sequences:        # Combined route sequences
timing:           # Estimated execution times
safety:           # Safety limits and validation
error_recovery:   # Recovery procedures
```

**Key Routes**:
- **ready**: Move to home/ready position
- **prepare_lane**: Position for pickup at specified lane (0-3)
- **pickup**: Execute object pickup with gripper control
- **transport_to_dropoff**: Move object to dropoff bin
- **dropoff**: Release object at dropoff location

**Announcement System**:
Each route includes:
- `announce_start`: Message when route begins
- `announce_complete`: Message when route completes
- Parameter substitution: `{lane}`, `{pickup_x}` etc.

**Example Route**:
```yaml
prepare_lane:
  announce_start: "Preparing for pickup - moving to lane {lane}..."
  announce_complete: "Lane {lane} position reached - ready for pickup trigger"
  parameters:
    - name: "lane"
      type: "int"
      range: [0, 3]
  commands:
    - gcode: "G1"
      params: "Z{safe_z} F{feed_rates.positioning}"
      description: "Raise Z to safe height"
```

### 3. `test_grbl_interface.py`
**Purpose**: Comprehensive testing utilities for GRBL interface

**Test Coverage**:
- Command execution and response validation
- Status report parsing and monitoring
- Real-time command testing (hold/resume)
- Error handling validation
- Interactive mode for manual testing

**Test Functions**:
- `test_grbl_interface()`: Basic command testing
- `test_error_handling()`: Error condition testing
- `interactive_test()`: Manual G-code console

## Modified Files

### 4. `movement_agent.py` (Completely Rewritten)
**Major Changes**:

#### Before:
- Used separate `GCodeRouteManager` for YAML processing
- Basic G-code execution through old `gcode_routines.py`
- Limited announcement system

#### After:
- **Integrated Route Management**: Handles YAML loading and processing internally
- **Enhanced Announcements**: Publishes to both `events:movement` and `events:announcements`
- **GRBL Communication**: Direct serial-like interface with GRBL agent
- **Parameter Substitution**: Advanced `{key.subkey}` pattern replacement
- **Error Recovery**: YAML-defined recovery sequences
- **Status Monitoring**: Real-time GRBL status parsing and position tracking

**New Methods Added**:
```python
_load_routes()              # Load YAML configuration
_substitute_parameters()    # Handle parameter replacement
_get_route()               # Retrieve and process routes
_process_route()           # Process individual routes
_process_sequence()        # Handle route sequences
_announce()                # Publish announcements
_execute_route()           # Execute YAML-defined routes
```

**Route Execution Flow**:
1. Load route from YAML with parameters
2. Substitute parameters in commands and announcements
3. Announce route start
4. Execute each G-code command through GRBL interface
5. Handle dwell commands with actual delays
6. Announce route completion
7. Return success/failure status

### 5. `requirements.txt` (Updated)
**Added Dependencies**:
```
PyYAML==6.0    # For YAML route configuration parsing
```

## Removed Files

### Files Deleted:
1. **`gcode_route_manager.py`** - Functionality integrated into Movement Agent
2. **`gcode_agent.py`** - Redundant with GRBL Interface Agent
3. **`movement_agent_yaml.py`** - Temporary file, functionality moved to main agent

**Rationale**: Eliminated unnecessary abstraction layers and redundant functionality to create a cleaner, more maintainable architecture.

## Technical Implementation Details

### GRBL Interface Communication
```python
# Serial-like interface
grbl_serial = grbl.get_serial_interface()
grbl_serial.write('G1 X10 Y10 F1000\n')
response = grbl_serial.readline()
```

### YAML Route Processing
```python
# Parameter substitution example
text = "Move to lane {lane} at {positions.lanes.lane_{lane}.x}"
result = self._substitute_parameters(text, lane=2)
# Result: "Move to lane 2 at 337.5"
```

### Announcement System
```python
# Dual announcement publishing
self.publish('announcement', announcement)  # Movement events
self.redis.publish('events:announcements', json.dumps(announcement))  # General
```

### Route Execution
```python
# Complete route execution
route_data = self._get_route('prepare_lane', lane=2)
if route_data['announce_start']:
    self._announce(route_data['announce_start'])
    
for gcode, params, desc in route_data['commands']:
    command = self._format_gcode(gcode, params)
    success = self._send_command(command)
```

## Configuration Examples

### Machine Configuration
```yaml
machine:
  name: "FoxAlien 3S"
  workspace:
    x_max: 400
    y_max: 400
    z_max: 0
    z_min: -100
  safe_z: -10
  pickup_z: -85
```

### Position Definitions
```yaml
positions:
  ready:
    x: 200
    y: 200
    z: -10
  lanes:
    lane_0: { x: 337.5, y: 50, z: -10 }
    lane_1: { x: 337.5, y: 150, z: -10 }
    lane_2: { x: 337.5, y: 250, z: -10 }
    lane_3: { x: 337.5, y: 350, z: -10 }
  dropoff:
    x: 50
    y: 200
    z: -10
```

### Complete Pickup Sequence
```yaml
sequences:
  complete_pickup_cycle:
    announce_start: "Starting complete pickup cycle for lane {lane}..."
    announce_complete: "Pickup cycle completed successfully - system ready for next assignment"
    routes:
      - route: "prepare_lane"
        params: { lane: "{lane}" }
      - route: "pickup"
        params: { pickup_x: "{pickup_x}" }
      - route: "transport_to_dropoff"
      - route: "dropoff"
      - route: "ready"
```

## Integration Points

### Redis Events
**Published Channels**:
- `events:movement` - Movement agent status and announcements
- `events:announcements` - General system announcements
- `events:grbl` - GRBL interface responses (for debugging)

**Subscribed Channels**:
- `events:cnc` - Pickup assignments from scoring agent
- `events:trigger` - Object detection from trigger camera
- `events:grbl` - GRBL responses for logging

### Event Flow
```
1. Scoring Agent → events:cnc → pickup_assignment
2. Movement Agent → Executes prepare_lane route
3. Movement Agent → events:trigger → watch_for_object
4. Trigger Camera → events:trigger → object_approaching  
5. Movement Agent → Executes pickup sequence
6. Movement Agent → events:announcements → progress updates
```

## Benefits of New System

### 1. **Simplified Architecture**
- Reduced from 3 components to 2 main components
- Eliminated unnecessary abstraction layers
- All movement logic in one cohesive agent

### 2. **Enhanced Configurability**
- YAML-based route definitions
- Easy modification without code changes
- Parameter-driven route execution
- Configurable announcements

### 3. **Improved Monitoring**
- Real-time announcements for all operations
- Detailed progress tracking
- GRBL status integration
- Error reporting and recovery

### 4. **Better Maintainability**
- Related functionality kept together
- Clear separation of concerns
- Comprehensive error handling
- Standardized route format

### 5. **Realistic GRBL Simulation**
- Accurate FoxAlien 3S controller behavior
- Proper GRBL 1.1 protocol implementation
- Real timing and motion simulation
- Complete command set support

## Usage Examples

### Starting the System
```bash
# Start Movement Agent
python movement_agent.py

# Run GRBL Interface Tests
python test_grbl_interface.py
```

### Manual Route Testing
```python
# Create movement agent
agent = MovementAgent()

# Execute individual routes
agent._execute_route('ready')
agent._execute_route('prepare_lane', lane=2)
agent._execute_route('pickup', pickup_x=337.5)
```

### Custom Route Configuration
```yaml
# Add new route to gcode_routes.yaml
routes:
  custom_move:
    announce_start: "Executing custom move..."
    announce_complete: "Custom move completed"
    commands:
      - gcode: "G1"
        params: "X{target_x} Y{target_y} F2000"
        description: "Move to target position"
```

## Error Handling

### Recovery Sequences
```yaml
error_recovery:
  soft_error:
    commands:
      - gcode: "M5"
        description: "Ensure gripper is off"
      - gcode: "G28"
        description: "Home all axes"
      - gcode: "G1"
        params: "X{positions.ready.x} Y{positions.ready.y} Z{positions.ready.z}"
        description: "Move to ready position"
```

### Error Types Handled
- GRBL communication timeouts
- Invalid G-code commands
- Motion limit violations
- Gripper/spindle errors
- Emergency stop conditions

## Future Enhancements

### Potential Improvements
1. **Route Validation**: Pre-execution route validation
2. **Performance Metrics**: Route execution timing analysis
3. **Route Optimization**: Automatic path optimization
4. **Advanced Recovery**: Context-aware error recovery
5. **Route Variants**: Conditional route execution
6. **Real Hardware**: Actual FoxAlien 3S integration

### Extensibility
- Easy addition of new routes via YAML
- Configurable machine parameters
- Pluggable error recovery strategies
- Modular announcement system

## Conclusion

The new GRBL Movement System provides a robust, configurable, and maintainable solution for CNC control with comprehensive FoxAlien 3S simulation. The integration of route management directly into the Movement Agent eliminates unnecessary complexity while providing enhanced functionality and monitoring capabilities.

The system successfully bridges the gap between high-level pickup logic and low-level GRBL control, providing a realistic simulation environment for development and testing of automated pickup systems.