# Event-Driven Architecture with Scoring Agent

## Overview

The Pick1 system now uses a fully event-driven architecture where:
- Vision Agent detects objects
- Scoring Agent tracks and prioritizes pickable objects
- CNC publishes "ready" events when idle
- Scoring Agent assigns work based on FIFO scoring

## Components

### 1. Vision Agent (`vision_agent.py`)
- Monitors 50mm vision line
- Publishes to `events:vision` when objects cross
- Detection includes: ID, type, position, lane, area, height

### 2. Scoring Agent (in `test_data.py`)
- **Subscribes to**: `events:vision` for detections
- **Tracks**: Only green (pickable) objects
- **Scoring**: 100% FIFO by default (configurable)
- **Assignment**: Selects best object when CNC is ready
- **Memory**: Clears all tracked objects after each assignment (fresh start)

### 3. CNC Controller (in `test_data.py`)
- **Event-driven**: No polling, waits for assignments
- **Publishes**: `ready_for_assignment` when idle
- **Receives**: `pickup_assignment` with target object
- **Executes**: Full pickup cycle then announces ready again

## Event Flow

```
1. Object crosses 50mm line
   └─> Vision Agent publishes detection to events:vision

2. Scoring Agent receives detection
   └─> Adds green objects to tracking list with timestamp

3. CNC completes pickup (or starts idle)
   └─> Publishes ready_for_assignment to events:cnc

4. Scoring Agent receives ready event
   └─> Evaluates tracked objects in pickup zone (300-375mm)
   └─> Calculates FIFO score (time since detection)
   └─> Publishes pickup_assignment to events:cnc
   └─> Clears all tracked objects (fresh start)

5. CNC receives assignment
   └─> Executes pickup sequence
   └─> Returns to idle and publishes ready again
```

## Event Messages

### Vision Detection Event
```json
{
  "event": "object_detected",
  "timestamp": 1234567890.123,
  "data": {
    "id": "obj_0001",
    "type": "green",
    "lane": 2,
    "position_x": 50.0,
    "area": 1500,
    "height": 32.5,
    "confidence": 1.0
  }
}
```

### CNC Ready Event
```json
{
  "event": "ready_for_assignment",
  "timestamp": 1234567890.123,
  "data": {
    "cnc_id": "cnc:0",
    "position": 337.5
  }
}
```

### Pickup Assignment Event
```json
{
  "event": "pickup_assignment",
  "timestamp": 1234567890.123,
  "data": {
    "cnc_id": "cnc:0",
    "object_id": "obj_0001",
    "position": 325.5,
    "lane": 2,
    "score": 0.95
  }
}
```

## Scoring Configuration

The Scoring Agent uses configurable weights (in `test_data.py`):

```python
scoring_weights = {
    'fifo': 1.0,      # 100% - First detected, first picked
    'position': 0.0,   # 0% - Distance from pickup zone center
    'urgency': 0.0     # 0% - How close to leaving pickup zone
}
```

### FIFO Scoring (Default)
- Objects detected earlier get higher scores
- Score increases with time since detection
- Normalized to 0-1 range (max 10 seconds)

### Position Scoring (Optional)
- Objects closer to pickup zone center get higher scores
- Useful for optimal CNC positioning

### Urgency Scoring (Optional)
- Objects about to leave pickup zone get higher scores
- Prevents objects from passing through unpicked

## Key Design Decisions

1. **Only track green objects**: Reduces memory usage and complexity
2. **Clear after assignment**: Prevents stale data and edge cases
3. **FIFO primary**: Ensures fair processing order
4. **Event-driven CNC**: No polling, reactive to work availability
5. **Immediate assignment**: No waiting when CNC is ready

## Running the System

```bash
# Start all components
./run_with_vision.sh

# Or manually:
python vision_agent.py    # Terminal 1
python test_data.py       # Terminal 2 (includes scoring agent)
python app.py             # Terminal 3
```

## Monitoring

The system provides real-time feedback:
- `[Vision detected]` - Objects crossing vision line
- `[ScoringAgent]` - Tracking and assignment decisions
- `[CNC]` - Assignment execution
- `[Monitor]` - System statistics every 10 seconds

## Benefits

1. **Realistic simulation**: Mimics industrial automation patterns
2. **Clean separation**: Each component has single responsibility
3. **No missed objects**: Vision detection required for pickup
4. **Fair processing**: FIFO ensures no object waits too long
5. **Scalable**: Easy to add multiple CNCs or different scoring strategies