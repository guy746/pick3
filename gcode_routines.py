#!/usr/bin/env python3
"""
G-code routines for CNC pickup operations
Simple movement sequences for each lane
"""

# Lane positions (Y-axis) for 4 lanes with 100mm spacing
LANE_POSITIONS = {
    0: 0,
    1: 100,
    2: 200,
    3: 300
}

# Pickup zone center (X-axis)
PICKUP_ZONE_CENTER = 337.5  # Center of 300-375mm zone

# Simple pickup routines for each lane
# Format: (G-code, parameters, description)
PICKUP_ROUTINES = {
    'prepare': [
        ('G00', 'Z100', 'Raise to safe height'),
        ('G00', 'X{pickup_x} Y{lane_y}', 'Move to lane position'),
        ('G00', 'Z50', 'Lower to approach height'),
    ],
    
    'pickup': [
        ('G01', 'Z10 F100', 'Slow descent to pickup'),
        ('M03', '', 'Activate gripper'),
        ('G04', 'P0.3', 'Dwell 300ms for grip'),
        ('G00', 'Z100', 'Raise with object'),
    ],
    
    'deliver': [
        ('G00', 'X337.5 Y200', 'Move to bin position'),
        ('G00', 'Z50', 'Lower to drop height'),
        ('M05', '', 'Release gripper'),
        ('G04', 'P0.2', 'Dwell 200ms for release'),
        ('G00', 'Z100', 'Raise to safe height'),
    ],
    
    'home': [
        ('G00', 'X337.5 Y200 Z100', 'Return to home position'),
    ]
}

# Timing parameters for movements
MOVEMENT_TIMES = {
    'prepare': 0.4,      # Time to move to lane position
    'pickup': 0.6,       # Time for pickup sequence
    'deliver': 0.5,      # Time to deliver to bin
    'home': 0.3,         # Time to return home
}

def get_routine(routine_name, lane=None, pickup_x=None):
    """
    Get G-code routine with parameters filled in
    
    Args:
        routine_name: Name of routine (prepare, pickup, deliver, home)
        lane: Lane number (0-3) for position
        pickup_x: X position for pickup (defaults to zone center)
    
    Returns:
        List of (gcode, params, description) tuples
    """
    if routine_name not in PICKUP_ROUTINES:
        return []
    
    routine = PICKUP_ROUTINES[routine_name].copy()
    
    # Fill in parameters
    if lane is not None:
        lane_y = LANE_POSITIONS.get(lane, 200)
    else:
        lane_y = 200  # Default center lane
        
    if pickup_x is None:
        pickup_x = PICKUP_ZONE_CENTER
    
    # Replace placeholders in routine
    filled_routine = []
    for gcode, params, desc in routine:
        filled_params = params.format(
            pickup_x=pickup_x,
            lane_y=lane_y
        )
        filled_routine.append((gcode, filled_params, desc))
    
    return filled_routine

def get_routine_time(routine_name):
    """Get estimated execution time for a routine"""
    return MOVEMENT_TIMES.get(routine_name, 0.5)

def format_gcode_line(gcode, params):
    """Format a G-code line for execution"""
    if params:
        return f"{gcode} {params}"
    return gcode

def calculate_movement_time(from_pos, to_pos, speed=200):
    """
    Calculate time for a movement based on distance and speed
    
    Args:
        from_pos: (x, y, z) tuple
        to_pos: (x, y, z) tuple  
        speed: Movement speed in mm/s
    
    Returns:
        Time in seconds
    """
    import math
    
    # Calculate 3D distance
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    dz = to_pos[2] - from_pos[2]
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Time = distance / speed
    return distance / speed if speed > 0 else 0

# Example usage
if __name__ == "__main__":
    print("G-code Pickup Routines")
    print("=" * 50)
    
    # Show routine for lane 2
    lane = 2
    print(f"\nPickup routine for lane {lane}:")
    
    for routine_name in ['prepare', 'pickup', 'deliver', 'home']:
        print(f"\n{routine_name.upper()}:")
        routine = get_routine(routine_name, lane=lane)
        for gcode, params, desc in routine:
            line = format_gcode_line(gcode, params)
            print(f"  {line:<20} ; {desc}")
        print(f"  Estimated time: {get_routine_time(routine_name)}s")