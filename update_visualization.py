#!/usr/bin/env python3
"""
Script to update Pick1 visualization files for shorter belt with trigger line.
Run this from the C:\pick1-visualization directory.
"""

import os
import re

def update_app_py():
    """Update app.py with new conveyor configuration"""
    print("Updating app.py...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Replace CONVEYOR_CONFIG
    old_config = r"CONVEYOR_CONFIG = \{[^}]+\}"
    new_config = """CONVEYOR_CONFIG = {
    'length': 1000,  # mm (reduced by half)
    'width': 400,    # mm (100mm per lane)
    'lanes': 4,
    'belt_speed': 133.33,  # mm/sec (1000mm in 7.5 seconds)
    'vision_zone': 100,     # mm from start (10%)
    'trigger_zone': 700,    # mm (new - separate trigger line)
    'pickup_zone_start': 750,  # moved closer
    'pickup_zone_end': 800,    # smaller zone
    'post_pick_zone': 900      # near end
}"""
    
    content = re.sub(old_config, new_config, content, flags=re.DOTALL)
    
    with open('app.py', 'w') as f:
        f.write(content)
    print("✓ app.py updated")

def update_test_data_py():
    """Update test_data.py with new configurations"""
    print("Updating test_data.py...")
    
    with open('test_data.py', 'r') as f:
        content = f.read()
    
    # Update setup_conveyor_config
    old_setup = r"def setup_conveyor_config\(\):[^}]+\}"
    new_setup = """def setup_conveyor_config():
    \"\"\"Set up conveyor configuration\"\"\"
    config = {
        'belt_speed': 133.33,
        'length': 1000,  # Reduced by half
        'lanes': 4,
        'lane_width': 100,
        'vision_zone': 100,
        'trigger_zone': 700,  # New trigger zone
        'pickup_zone_start': 750,
        'pickup_zone_end': 800,
        'post_pick_zone': 900
    }"""
    
    content = re.sub(old_setup, new_setup, content, flags=re.DOTALL)
    
    # Update belt length check
    content = re.sub(r'if new_position > 2000:', 'if new_position > 1000:', content)
    
    # Update ring logic
    old_ring_logic = r"# Update ring status based on position\s+obj_type = r\.hget[^}]+\}"
    new_ring_logic = """# Update ring status based on position
                obj_type = r.hget(f'object:{obj_id}', 'type')
                if obj_type == 'green':
                    if new_position > 100 and new_position < 700:
                        r.hset(f'object:{obj_id}', 'has_ring', 'true')
                        r.hset(f'object:{obj_id}', 'ring_color', 'yellow')
                    elif new_position >= 700 and new_position < 750:
                        # Trigger zone - keep yellow
                        r.hset(f'object:{obj_id}', 'ring_color', 'yellow')
                    elif new_position >= 750 and new_position < 800:
                        # Pickup zone - turn red
                        r.hset(f'object:{obj_id}', 'ring_color', 'red')"""
    
    content = re.sub(old_ring_logic, new_ring_logic, content, flags=re.DOTALL)
    
    # Update pickup zone range
    content = re.sub(r'zrangebyscore\(\'objects:active\', 1500, 1600', 
                     'zrangebyscore(\'objects:active\', 750, 800', content)
    
    # Update CNC positions
    content = re.sub(r"'position_x': 1550,", "'position_x': 775,", content)
    content = re.sub(r"r\.hset\('cnc:0', 'position_x', 1550\)", 
                     "r.hset('cnc:0', 'position_x', 775)", content)
    
    with open('test_data.py', 'w') as f:
        f.write(content)
    print("✓ test_data.py updated")

def update_canvas_js():
    """Update canvas.js with new drawing functions"""
    print("Updating static/js/canvas.js...")
    
    with open('static/js/canvas.js', 'r') as f:
        content = f.read()
    
    # Replace drawConveyor function
    old_draw_conveyor = r"// Draw conveyor belt\s*function drawConveyor\(\) \{[^}]+\}"
    new_draw_conveyor = """// Draw conveyor belt
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
    
    // Vision zone line
    const visionX = mmToPixels(CONFIG.vision_zone);
    ctx.beginPath();
    ctx.moveTo(visionX, 50);
    ctx.lineTo(visionX, 250);
    ctx.stroke();
    
    // Trigger zone line (new - in purple)
    const triggerX = mmToPixels(CONFIG.trigger_zone);
    ctx.strokeStyle = '#9C27B0';  // Purple for trigger
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(triggerX, 50);
    ctx.lineTo(triggerX, 250);
    ctx.stroke();
    
    // Pickup zone (shaded area)
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    const pickupStartX = mmToPixels(CONFIG.pickup_zone_start);
    const pickupEndX = mmToPixels(CONFIG.pickup_zone_end);
    ctx.fillStyle = 'rgba(255, 255, 0, 0.1)';
    ctx.fillRect(pickupStartX, 50, pickupEndX - pickupStartX, 200);
    
    // Post-pick monitor line
    const postPickX = mmToPixels(CONFIG.post_pick_zone);
    ctx.beginPath();
    ctx.moveTo(postPickX, 50);
    ctx.lineTo(postPickX, 250);
    ctx.stroke();
    
    ctx.setLineDash([]);
    
    // Zone labels
    ctx.fillStyle = '#999';
    ctx.font = '10px Arial';
    ctx.fillText('Vision', visionX + 5, 45);
    ctx.fillText('Trigger', triggerX + 5, 45);
    ctx.fillText('Pickup', pickupStartX + 5, 45);
    ctx.fillText('Monitor', postPickX + 5, 45);
}"""
    
    # Due to regex complexity with nested braces, let's do a simpler replacement
    # Find the start and end of the function
    start_idx = content.find('// Draw conveyor belt\nfunction drawConveyor() {')
    if start_idx == -1:
        start_idx = content.find('function drawConveyor() {')
    
    # Find the matching closing brace
    brace_count = 0
    i = content.find('{', start_idx)
    end_idx = i
    
    while i < len(content) and brace_count >= 0:
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
        i += 1
    
    # Replace the function
    if start_idx != -1 and end_idx > start_idx:
        content = content[:start_idx] + new_draw_conveyor + content[end_idx:]
    
    # Update bin position
    content = re.sub(r'const binX = mmToPixels\(1550\);', 
                     'const binX = mmToPixels(775);', content)
    
    with open('static/js/canvas.js', 'w') as f:
        f.write(content)
    print("✓ static/js/canvas.js updated")

def update_index_html():
    """Update index.html with new zone information"""
    print("Updating templates/index.html...")
    
    with open('templates/index.html', 'r') as f:
        content = f.read()
    
    # Replace zones section
    old_zones = r'<div class="zones">.*?</div>\s*</div>'
    new_zones = """<div class="zones">
                <h3>Zones</h3>
                <div>Vision Detection: 100mm</div>
                <div>Trigger Zone: 700mm</div>
                <div>Pickup Zone: 750-800mm</div>
                <div>Post-Pick Monitor: 900mm</div>
            </div>"""
    
    content = re.sub(old_zones, new_zones + '\n        </div>', content, flags=re.DOTALL)
    
    with open('templates/index.html', 'w') as f:
        f.write(content)
    print("✓ templates/index.html updated")

def main():
    """Run all updates"""
    print("Starting Pick1 visualization updates...\n")
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("Error: Make sure you're running this from the C:\\pick1-visualization directory!")
        return
    
    try:
        update_app_py()
        update_test_data_py()
        update_canvas_js()
        update_index_html()
        
        print("\n✅ All files updated successfully!")
        print("\nNext steps:")
        print("1. Stop both running programs with Ctrl+C")
        print("2. Restart the app: python app.py")
        print("3. Refresh your browser")
        print("4. Restart test data: python test_data.py")
        
    except Exception as e:
        print(f"\n❌ Error updating files: {e}")
        print("Make sure all files exist and are not in use.")

if __name__ == '__main__':
    main()