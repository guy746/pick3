# Vision Agent Architecture

The Pick1 system now uses a modular architecture with a dedicated Vision Agent for object detection.

## Architecture Overview

### Components

1. **Vision Agent** (`vision_agent.py`)
   - Monitors objects crossing the 50mm vision detection line
   - Publishes detection events to Redis channel `events:vision`
   - Tracks detected objects to prevent duplicate events
   - Provides perfect detection (100% accuracy)

2. **Test Data Generator** (`test_data.py`)
   - Creates and moves objects along the conveyor belt
   - No longer handles vision detection or ring management
   - Publishes object positions and handles CNC pickup operations

3. **Web Application** (`app.py`)
   - Subscribes to vision events
   - Updates object rings when green objects are detected
   - Manages visualization and client connections

## Communication Flow

1. Test Data creates objects and moves them along the belt
2. Vision Agent detects objects crossing 50mm line
3. Vision Agent publishes detection event:
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
4. App.py receives event and adds yellow ring to green objects
5. Frontend displays objects with appropriate visual indicators

## Running the System

### Option 1: Use the integrated startup script
```bash
./run_with_vision.sh
```

### Option 2: Run components manually
```bash
# Terminal 1 - Vision Agent
python vision_agent.py

# Terminal 2 - Test Data Generator
python test_data.py

# Terminal 3 - Web Application
python app.py
```

## Testing

### Test Vision Agent Standalone
```bash
python test_vision_agent.py
```

This creates test objects and monitors vision detection events.

## Benefits of This Architecture

1. **Separation of Concerns**: Each component has a single responsibility
2. **Scalability**: Vision processing can be scaled independently
3. **Flexibility**: Easy to add more detection zones or vision capabilities
4. **Realistic Simulation**: Mimics real computer vision systems
5. **Maintainability**: Easier to debug and modify individual components

## Future Enhancements

- Trigger Agent: Monitor 250mm trigger line
- Pickup Agent: Handle CNC coordination
- Quality Agent: Post-pickup verification
- Configuration Manager: Dynamic system configuration

## Redis Channels

- `events:vision` - Vision detection events
- `events:spawn` - Object creation events
- `events:trigger` - Trigger line events (future)
- `events:motion` - Motion detection events
- `events:system` - System status events