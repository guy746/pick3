#!/usr/bin/env python3
"""
Add visual alert when green objects pass through without being picked up
Post-pick monitor line flashes red when a miss is detected
"""

def update_canvas_js():
    """Update canvas.js to show missed pickup alerts"""
    print("Updating static/js/canvas.js...")
    
    # Read the current file
    with open('static/js/canvas.js', 'r') as f:
        content = f.read()
    
    # Add missed pickup tracking variables after binFlashTime
    flash_vars = """let animationFrame = null;
let socket = null;
let binFlashTime = 0;
let missedPickupTime = 0;  // Track when a missed pickup was detected"""
    
    content = content.replace("let binFlashTime = 0;", "let binFlashTime = 0;\nlet missedPickupTime = 0;  // Track when a missed pickup was detected")
    
    # Update the world_update handler to check for missed pickups
    update_handler = """    socket.on('world_update', (state) => {
        worldState = state;
        if (state.bin_flash) {
            binFlashTime = Date.now();
        }
        if (state.missed_pickup_alert) {
            missedPickupTime = Date.now();
            // Auto-clear the alert from Redis after reading
            socket.emit('clear_missed_alert');
        }
        updateStatus();
    });"""
    
    # Find and replace the world_update handler
    world_update_start = "socket.on('world_update', (state) => {"
    world_update_idx = content.find(world_update_start)
    if world_update_idx != -1:
        # Find the end of this handler
        brace_count = 1
        i = world_update_idx + len(world_update_start)
        end_idx = i
        while i < len(content) and brace_count > 0:
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
            i += 1
        
        # Replace with updated handler
        content = content[:world_update_idx] + update_handler + content[end_idx + 2:]  # +2 to skip ");"
    
    
    # Find and update the drawConveyor function
    new_draw_conveyor = '''// Draw conveyor belt
function drawConveyor() {
    // Belt background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 50, CANVAS_WIDTH, 200);
    
    // Belt edges
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 50, CANVAS_WIDTH, 200);
    
    // Zone markers
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    
    // Vision zone line at 50mm = 100px
    const visionX = mmToPixels(CONFIG.vision_zone);
    ctx.beginPath();
    ctx.moveTo(visionX, 50);
    ctx.lineTo(visionX, 250);
    ctx.stroke();
    
    // Trigger zone line at 300mm = 600px (purple)
    const triggerX = mmToPixels(CONFIG.trigger_zone);
    ctx.strokeStyle = '#9C27B0';  // Purple for trigger
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(triggerX, 50);
    ctx.lineTo(triggerX, 250);
    ctx.stroke();
    
    // Pickup zone at 375-425mm = 750-850px (shaded area)
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    const pickupStartX = mmToPixels(CONFIG.pickup_zone_start);
    const pickupEndX = mmToPixels(CONFIG.pickup_zone_end);
    ctx.fillStyle = 'rgba(255, 255, 0, 0.1)';
    ctx.fillRect(pickupStartX, 50, pickupEndX - pickupStartX, 200);
    
    // Pickup zone boundary lines
    ctx.strokeStyle = '#FFD700';  // Gold for pickup boundaries
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(pickupStartX, 50);
    ctx.lineTo(pickupStartX, 250);
    ctx.moveTo(pickupEndX, 50);
    ctx.lineTo(pickupEndX, 250);
    ctx.stroke();
    
    // Post-pick monitor line at 475mm = 950px
    // Flash RED if a missed pickup was detected recently
    const postPickX = mmToPixels(CONFIG.post_pick_zone);
    const isFlashingRed = (Date.now() - missedPickupTime) < 2000; // Flash for 2 seconds
    
    if (isFlashingRed) {
        // Pulse effect for missed pickup
        const flashIntensity = Math.sin((Date.now() - missedPickupTime) * 0.01) * 0.5 + 0.5;
        ctx.strokeStyle = `rgba(255, 0, 0, ${0.5 + flashIntensity * 0.5})`;
        ctx.lineWidth = 3 + flashIntensity * 2;
        
        // Draw red glow effect
        ctx.shadowBlur = 10;
        ctx.shadowColor = 'red';
    } else {
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.shadowBlur = 0;
    }
    
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(postPickX, 50);
    ctx.lineTo(postPickX, 250);
    ctx.stroke();
    
    // Reset shadow
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);
    
    // Zone labels with alert for missed pickup
    ctx.fillStyle = '#999';
    ctx.font = '12px Arial';
    ctx.fillText('Vision', visionX + 5, 45);
    ctx.fillText('Trigger', triggerX + 5, 45);
    ctx.fillText('Pickup', pickupStartX + 5, 45);
    
    if (isFlashingRed) {
        ctx.fillStyle = '#FF0000';
        ctx.font = 'bold 12px Arial';
        ctx.fillText('MISSED!', postPickX - 35, 45);
    } else {
        ctx.fillStyle = '#999';
        ctx.font = '12px Arial';
        ctx.fillText('Monitor', postPickX - 35, 45);
    }
    
    // Add scale reference at bottom
    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.fillText('0mm', 5, 265);
    ctx.fillText('500mm', CANVAS_WIDTH - 35, 265);
}'''
    
    # Find and replace drawConveyor function
    start_marker = "// Draw conveyor belt"
    start_idx = content.find(start_marker)
    
    if start_idx != -1:
        # Find the end of the function
        brace_count = 0
        i = content.find('{', start_idx)
        end_idx = i
        
        while i < len(content):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
            i += 1
        
        # Replace the function
        content = content[:start_idx] + new_draw_conveyor + content[end_idx:]
    
    # Write the updated content
    with open('static/js/canvas.js', 'w') as f:
        f.write(content)
    
    print("✓ static/js/canvas.js updated with missed pickup alerts")

def update_app_py():
    """Update app.py to handle missed pickup alerts"""
    print("Updating app.py to support missed pickup alerts...")
    
    # Read the current file
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Add missed pickup alert to get_world_state
    world_state_update = """        # Get bin state
        bin_flash = redis_client.get('bin:0:flash')
        if bin_flash:
            state['bin_flash'] = True
            # Auto-clear flash after reading
            redis_client.delete('bin:0:flash')
            
        # Get missed pickup alert
        missed_alert = redis_client.get('missed_pickup:alert')
        if missed_alert:
            state['missed_pickup_alert'] = True"""
    
    # Find and update the section
    bin_flash_idx = content.find("# Get bin state")
    if bin_flash_idx != -1:
        # Find the end of this section
        end_marker = "except Exception as e:"
        end_idx = content.find(end_marker, bin_flash_idx)
        if end_idx != -1:
            # Replace the section
            new_section = world_state_update + "\n            \n    "
            content = content[:bin_flash_idx] + new_section + content[end_idx:]
    
    # Add handler to clear missed alert
    clear_handler = """
@socketio.on('clear_missed_alert')
def handle_clear_missed_alert():
    \"\"\"Clear the missed pickup alert\"\"\"
    redis_client.delete('missed_pickup:alert')
"""
    
    # Add before the main block
    main_idx = content.find("if __name__ == '__main__':")
    if main_idx != -1:
        content = content[:main_idx] + clear_handler + "\n" + content[main_idx:]
    
    # Write the updated content
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("✓ app.py updated")

def create_enhanced_test():
    """Create test with missed pickup detection"""
    print("Creating enhanced_pickup_test.py with miss detection...")
    
    content = '''#!/usr/bin/env python3
"""
Enhanced pickup test with missed pickup alerts
Post-pick monitor line flashes red when green objects are missed
"""

import redis
import time
import random
import threading

# Using host.docker.internal for Docker container networking
r = redis.Redis(host='host.docker.internal', port=6379, decode_responses=True)

# Tracking
assigned_objects = {}  # Track which objects were assigned
pickup_stats = {'successful': 0, 'missed': 0}

# Belt configuration
BELT_SPEED = 100  # mm/sec
UPDATE_INTERVAL = 0.1
MOVEMENT_PER_UPDATE = BELT_SPEED * UPDATE_INTERVAL

def setup():
    """Setup conveyor and CNC"""
    r.hset('conveyor:config', mapping={
        'belt_speed': str(BELT_SPEED),
        'length': '500',
        'vision_zone': '50',
        'trigger_zone': '300',
        'pickup_zone_start': '375',
        'pickup_zone_end': '425',
        'post_pick_zone': '475'
    })
    
    r.hset('cnc:0', mapping={
        'position_x': '400',
        'position_y': '200',
        'position_z': '100',
        'status': 'idle',
        'has_object': 'false'
    })

def create_object(obj_id, pos, lane, color):
    """Create an object"""
    r.hset(f'object:{obj_id}', mapping={
        'position_x': str(pos),
        'lane': str(lane),
        'type': color,
        'status': 'moving',
        'has_ring': 'false',
        'ring_color': 'yellow',
        'assigned': 'false'  # Track assignment
    })
    r.zadd('objects:active', {obj_id: pos})
    print(f"Created {color} object {obj_id} in lane {lane}")

def move_objects():
    """Move objects and check for misses"""
    while True:
        objects = r.zrange('objects:active', 0, -1, withscores=True)
        pipe = r.pipeline()
        
        for obj_id, pos in objects:
            new_pos = pos + MOVEMENT_PER_UPDATE
            
            # Check if object passed post-pick monitor
            if 470 < pos < 480:  # Just passing monitor line
                obj_type = r.hget(f'object:{obj_id}', 'type')
                was_assigned = r.hget(f'object:{obj_id}', 'assigned') == 'true'
                
                # Check if this was a missed green object
                if obj_type == 'green' and was_assigned:
                    print(f"\\n⚠️  MISSED PICKUP DETECTED: {obj_id}")
                    print(f"   Green object passed through without pickup!")
                    
                    # Set alert for visualization
                    r.setex('missed_pickup:alert', 3, 'true')
                    pickup_stats['missed'] += 1
                    
                    # Remove from assigned list
                    if obj_id in assigned_objects:
                        del assigned_objects[obj_id]
            
            if new_pos > 500:
                # Remove if past belt
                pipe.zrem('objects:active', obj_id)
                pipe.delete(f'object:{obj_id}')
                if obj_id in assigned_objects:
                    del assigned_objects[obj_id]
            else:
                # Update position
                pipe.hset(f'object:{obj_id}', 'position_x', str(new_pos))
                pipe.zadd('objects:active', {obj_id: new_pos})
                
                # Update ring for green objects
                obj_type = r.hget(f'object:{obj_id}', 'type')
                if obj_type == 'green':
                    if 50 < new_pos < 300:
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'yellow')
                    elif new_pos >= 300:
                        pipe.hset(f'object:{obj_id}', 'has_ring', 'true')
                        pipe.hset(f'object:{obj_id}', 'ring_color', 'red')
        
        pipe.execute()
        time.sleep(UPDATE_INTERVAL)

def simulate_assignment():
    """Simulate scoring agent assignments"""
    while True:
        time.sleep(0.5)
        
        # Find green objects approaching trigger
        approaching = r.zrangebyscore('objects:active', 250, 295, withscores=True)
        
        for obj_id, pos in approaching:
            obj_type = r.hget(f'object:{obj_id}', 'type')
            already_assigned = r.hget(f'object:{obj_id}', 'assigned') == 'true'
            
            if obj_type == 'green' and not already_assigned:
                # Check if CNC is available
                cnc_status = r.hget('cnc:0', 'status')
                
                if cnc_status == 'idle' and len(assigned_objects) == 0:
                    # Assign the object
                    r.hset(f'object:{obj_id}', 'assigned', 'true')
                    assigned_objects[obj_id] = {
                        'lane': int(r.hget(f'object:{obj_id}', 'lane')),
                        'assigned_at': pos
                    }
                    print(f"[Assignment] Green object {obj_id} assigned for pickup")
                    break

def cnc_pickup_cycle():
    """CNC pickup with miss capability"""
    while True:
        time.sleep(0.3)
        
        if not assigned_objects:
            continue
            
        cnc_status = r.hget('cnc:0', 'status')
        if cnc_status != 'idle':
            continue
        
        # Get next assigned object
        obj_id = next(iter(assigned_objects))
        assignment = assigned_objects[obj_id]
        
        # Check if object still exists and is in pickup zone
        obj_pos = r.zscore('objects:active', obj_id)
        
        if obj_pos and 375 <= obj_pos <= 425:
            # Simulate 85% success rate
            if random.random() < 0.85:
                # Successful pickup sequence
                print(f"\\n✅ PICKUP SEQUENCE: {obj_id}")
                
                # Execute pickup
                r.hset('cnc:0', 'status', 'moving_to_position')
                r.hset('cnc:0', 'position_x', str(obj_pos))
                time.sleep(0.2)
                
                r.hset('cnc:0', 'status', 'picking')
                r.hset('cnc:0', 'has_object', 'true')
                r.zrem('objects:active', obj_id)
                r.delete(f'object:{obj_id}')
                time.sleep(0.3)
                
                r.hset('cnc:0', 'status', 'moving_to_bin')
                r.hset('cnc:0', 'position_x', '400')
                time.sleep(0.3)
                
                r.hset('cnc:0', 'status', 'dropping')
                r.hset('cnc:0', 'has_object', 'false')
                r.setex('bin:0:flash', 1, 'true')
                time.sleep(0.2)
                
                r.hset('cnc:0', 'status', 'idle')
                pickup_stats['successful'] += 1
                
                del assigned_objects[obj_id]
                print(f"   Pickup complete! Total: {pickup_stats['successful']}")
            else:
                # Simulate miss (e.g., vacuum failure, timing issue)
                print(f"\\n❌ PICKUP FAILED: {obj_id} (simulated miss)")
                r.hset('cnc:0', 'status', 'idle')
                # Don't delete object, let it pass through
                time.sleep(0.5)

def spawn_objects():
    """Spawn objects"""
    counter = 1
    
    while True:
        time.sleep(random.uniform(3, 4))
        
        obj_id = f'obj_{counter:04d}'
        lane = random.randint(0, 3)
        
        # 50% green objects
        color = 'green' if random.random() < 0.5 else random.choice(['blue', 'red', 'yellow', 'orange'])
        
        create_object(obj_id, 0, lane, color)
        counter += 1

def status_monitor():
    """Show statistics"""
    while True:
        time.sleep(10)
        
        total = pickup_stats['successful'] + pickup_stats['missed']
        if total > 0:
            success_rate = (pickup_stats['successful'] / total) * 100
            print(f"\\n📊 STATISTICS:")
            print(f"   Successful: {pickup_stats['successful']}")
            print(f"   Missed: {pickup_stats['missed']}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Watch for RED FLASHING monitor line when misses occur!\\n")

def main():
    print("="*60)
    print("Enhanced Pickup Test with Missed Pickup Alerts")
    print("="*60)
    print("Features:")
    print("- Post-pick monitor line flashes RED for missed green objects")
    print("- 85% pickup success rate (simulated)")
    print("- Clear statistics tracking")
    print("="*60 + "\\n")
    
    r.flushdb()
    setup()
    
    # Initial objects
    create_object('obj_0001', 100, 0, 'green')
    create_object('obj_0002', 200, 1, 'blue')
    create_object('obj_0003', 300, 2, 'green')
    
    # Start threads
    threads = [
        threading.Thread(target=move_objects, daemon=True),
        threading.Thread(target=simulate_assignment, daemon=True),
        threading.Thread(target=cnc_pickup_cycle, daemon=True),
        threading.Thread(target=spawn_objects, daemon=True),
        threading.Thread(target=status_monitor, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    print("Running... Watch for RED FLASHING when green objects are missed!\\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\nFinal Statistics:")
        print(f"Successful pickups: {pickup_stats['successful']}")
        print(f"Missed pickups: {pickup_stats['missed']}")

if __name__ == '__main__':
    main()
'''
    
    with open('enhanced_pickup_test.py', 'w') as f:
        f.write(content)
    
    print("✓ Created enhanced_pickup_test.py")

def main():
    print("Adding Missed Pickup Alert System")
    print("="*60)
    print("Post-pick monitor line will flash RED when:")
    print("- A GREEN object that was assigned passes through")
    print("- The object wasn't picked up successfully")
    print("="*60 + "\n")
    
    try:
        update_canvas_js()
        update_app_py()
        create_enhanced_test()
        
        print("\n✅ Updates complete!")
        print("\nNew visual alerts:")
        print("- Post-pick line flashes RED for 2 seconds")
        print("- 'MISSED!' text appears during flash")
        print("- Red glow effect for emphasis")
        
        print("\nTo see it in action:")
        print("1. Restart app.py")
        print("2. Refresh your browser")
        print("3. Run: python enhanced_pickup_test.py")
        print("\nThe test has an 85% success rate, so you'll see misses!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()