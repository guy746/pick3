# CNC & Trigger Camera Architecture

## Overview

The Pick1 system now features a fully modular architecture with separate agents for each major function:

1. **Vision Agent** - Detects objects at 50mm line
2. **Trigger Camera Agent** - Monitors specific objects at 250mm trigger line
3. **CNC Agent** - Executes pickup operations with G-code routines
4. **Scoring Agent** - Prioritizes and assigns work

## Component Details

### CNC Agent (`cnc_agent.py`)

The CNC Agent is an event-driven controller that:
- Receives pickup assignments from the Scoring Agent
- Pre-positions to the assigned lane immediately
- Requests the Trigger Camera to watch for the specific object
- Calculates precise timing when triggered
- Executes simple G-code routines for pickup
- Publishes ready status after completing operations

**Key Features:**
- No polling - fully event-driven
- Dynamic timing calculation based on object position/velocity
- Simple G-code execution (move down, grab, move up)
- Assumes all pickups succeed (simplified for simulation)

### Trigger Camera Agent (`trigger_camera_agent.py`)

The Trigger Camera monitors the 250mm trigger line:
- Receives watch requests from CNC with specific object ID and lane
- Only monitors the requested lane (efficient)
- Detects when watched object crosses trigger line
- Publishes object position and velocity for CNC timing
- Automatic timeout for watches (prevents stale requests)

**Key Features:**
- Lane-specific monitoring (no wasted processing)
- High-frequency checking (50Hz) for precision
- Automatic cleanup of expired watches
- Velocity calculation for timing

### G-code Routines (`gcode_routines.py`)

Defines simple pickup sequences:
- **Prepare**: Move to safe height and lane position
- **Pickup**: Lower, activate gripper, raise
- **Deliver**: Move to bin, release
- **Home**: Return to home position

## Event Flow

```
1. Scoring Agent assigns object to CNC
   └─> Publishes pickup_assignment to events:cnc
       {
         "object_id": "obj_0001",
         "lane": 2,
         "position": 325.5
       }

2. CNC Agent receives assignment
   └─> Pre-positions to assigned lane
   └─> Publishes watch_for_object to events:trigger
       {
         "object_id": "obj_0001",
         "lane": 2,
         "timeout": 3.0
       }

3. Trigger Camera adds to watch list
   └─> Monitors only lane 2 for obj_0001

4. Object crosses 250mm trigger line
   └─> Publishes object_approaching to events:trigger
       {
         "object_id": "obj_0001",
         "current_position": 250,
         "velocity": 133.33,
         "eta_to_pickup": 0.375
       }

5. CNC Agent receives trigger
   └─> Calculates wait time dynamically
   └─> Executes G-code pickup routine
   └─> Removes object from belt
   └─> Returns to home

6. CNC publishes ready_for_assignment
   └─> Cycle repeats
```

## Timing Calculation

The CNC dynamically calculates when to start the pickup:

```python
# Distance from trigger to pickup zone center
distance = 337.5 - 250 = 87.5mm

# Time for object to travel
travel_time = distance / velocity = 0.656s

# Account for pickup descent time
pickup_prep_time = 0.3s

# Wait time after trigger
wait_time = travel_time - pickup_prep_time = 0.356s
```

## Configuration

### Belt Layout
- Vision line: 50mm
- Trigger line: 250mm  
- Pickup zone: 300-375mm (center: 337.5mm)
- Belt speed: 133.33mm/s

### CNC Home Position
- X: 337.5mm (center of pickup zone)
- Y: 200mm (center lane)
- Z: 100mm (safe height)

## Running the System

```bash
# Start all agents
./run_with_vision.sh

# Agents start in order:
1. Vision Agent
2. Trigger Camera Agent  
3. CNC Agent
4. Test Data (with Scoring Agent)
5. Flask Web App
```

## Monitoring

Each agent provides status updates:
- `[Vision detected]` - Objects at 50mm
- `[TriggerCamera]` - Watch list and triggers
- `[CNC]` - Assignments and operations
- `[ScoringAgent]` - Tracking and assignments

## Benefits

1. **Realistic Simulation**: Mimics industrial pick-and-place systems
2. **Modular Design**: Each agent has single responsibility
3. **Precise Timing**: Dynamic calculation for accurate pickup
4. **Event-Driven**: No polling, reactive system
5. **Scalable**: Easy to add multiple CNCs or cameras

## Future Enhancements

- Multiple CNC support with load balancing
- Vision-based position correction
- Failure handling and retry logic
- Advanced G-code routines for different object types
- Performance metrics and optimization