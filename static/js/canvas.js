// Canvas setup
const canvas = document.getElementById('conveyor-canvas');
const ctx = canvas.getContext('2d');

// Canvas dimensions
const CANVAS_WIDTH = 1000;
const CANVAS_HEIGHT = 300;
const SCALE = CANVAS_WIDTH / CONFIG.length;  // Scale factor for mm to pixels

canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;

// State
let worldState = {
    objects: {},
    cnc: {
        position_x: 400,  // Center of pickup zone
        position_y: 200,
        position_z: 100,
        status: 'idle',
        has_object: false
    }
};

let animationFrame = null;
let socket = null;
let binFlashTime = 0;
let missedPickupTime = 0;  // Track when a missed pickup was detected
let triggerFlashTime = 0;  // Track when trigger line should flash
let monitorFlashTime = 0;  // Track when monitor line should flash
let scoringLaneInfo = null;  // Store scoring lane info with timestamp
let scoringDisplayTimeout = null;  // Timeout for clearing scoring display

// Robot arm animation state
let robotArm = {
    currentX: 600,  // Start at pickup zone edge (300mm)
    currentY: 150,  // Center of belt
    targetX: 600,
    targetY: 150,
    animationSpeed: 0.15  // Interpolation speed (0.1 = slow, 0.3 = fast)
};

// Object type colors
const OBJECT_COLORS = {
    'green': '#4CAF50',
    'blue': '#2196F3',
    'red': '#F44336',
    'yellow': '#FFEB3B',
    'orange': '#FF9800'
};

// Initialize socket connection
function initSocket() {
    socket = io();
    
    socket.on('connect', () => {
        document.getElementById('connection-status').textContent = 'Connected';
        document.getElementById('connection-status').className = 'connected';
    });
    
    socket.on('disconnect', () => {
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.getElementById('connection-status').className = 'disconnected';
    });
    
    socket.on('world_state', (state) => {
        worldState = state;
        if (state.bin_flash) {
            binFlashTime = Date.now();
        }
        if (state.trigger_flash) {
            triggerFlashTime = Date.now();
        }
        if (state.monitor_flash) {
            monitorFlashTime = Date.now();
        }
        updateStatus();
    });
    
        socket.on('world_update', (state) => {
        worldState = state;
        if (state.bin_flash) {
            binFlashTime = Date.now();
        }
        if (state.trigger_flash) {
            triggerFlashTime = Date.now();
        }
        if (state.monitor_flash) {
            monitorFlashTime = Date.now();
        }
        if (state.missed_pickup_alert) {
            missedPickupTime = Date.now();
            // Auto-clear the alert from Redis after reading
            socket.emit('clear_missed_alert');
        }
        updateStatus();
    });
    
    socket.on('event', (event) => {
        console.log('Event:', event);
    });
    
    socket.on('status_message', (message) => {
        addStatusMessage(message);
        updateAgentStatus(message);
    });
}

// Track last data update time
let lastDataUpdate = 0;

// Update status display
function updateStatus() {
    const objectCount = Object.keys(worldState.objects).length;
    document.getElementById('object-count').textContent = `Objects: ${objectCount}`;
    document.getElementById('cnc-status').textContent = `CNC: ${worldState.cnc?.status || 'Unknown'}`;
    
    // Update confirmed target status
    const confirmedTargetElement = document.getElementById('confirmed-target-status');
    if (worldState.confirmed_target) {
        const { object_id, lane } = worldState.confirmed_target;
        confirmedTargetElement.textContent = `Confirmed: ${object_id} Lane ${lane}`;
    } else {
        confirmedTargetElement.textContent = 'Confirmed: None';
    }
    
    // Update scoring status with lane assignment (with 2-second persistence)
    const assignedLane = worldState.cnc?.assigned_lane;
    
    if (assignedLane !== undefined) {
        // New lane assignment - store it and display immediately
        scoringLaneInfo = {
            lane: assignedLane,
            timestamp: Date.now()
        };
        document.getElementById('scoring-status').textContent = `Scoring: Lane ${assignedLane}`;
        
        // Clear any existing timeout
        if (scoringDisplayTimeout) {
            clearTimeout(scoringDisplayTimeout);
        }
    } else if (scoringLaneInfo) {
        // No current assignment, but we have stored info
        const timeSinceAssignment = Date.now() - scoringLaneInfo.timestamp;
        
        if (timeSinceAssignment < 2000) {
            // Keep showing the lane for 2 seconds
            document.getElementById('scoring-status').textContent = `Scoring: Lane ${scoringLaneInfo.lane}`;
        } else {
            // More than 2 seconds - clear to idle
            scoringLaneInfo = null;
            document.getElementById('scoring-status').textContent = 'Scoring: Idle';
        }
    } else {
        // No assignment and no stored info
        document.getElementById('scoring-status').textContent = 'Scoring: Idle';
    }
    
    // Update connection status based on data flow
    updateConnectionStatus(objectCount);
    
    // Update status messages if they're in the world state
    if (worldState.status_messages) {
        updateStatusMessages(worldState.status_messages);
    }
}

// Update connection status based on Redis data flow
function updateConnectionStatus(objectCount) {
    const now = Date.now();
    const connectionStatus = document.getElementById('connection-status');
    
    // If we have objects or recent world state updates, we're connected
    if (objectCount > 0 || (worldState.timestamp && (now - (worldState.timestamp * 1000)) < 5000)) {
        lastDataUpdate = now;
        connectionStatus.textContent = 'Connected';
        connectionStatus.className = 'connected';
    } else if (now - lastDataUpdate > 10000) {
        // No data for 10 seconds = disconnected
        connectionStatus.textContent = 'Disconnected';
        connectionStatus.className = 'disconnected';
    }
}

// Add a status message to the display
function addStatusMessage(message) {
    const statusLog = document.getElementById('status-log');
    if (!statusLog) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `status-message ${message.agent}`;
    
    const timestamp = document.createElement('span');
    timestamp.className = 'timestamp';
    timestamp.textContent = `[${message.timestamp}] `;
    
    const content = document.createElement('span');
    content.textContent = `${message.agent.toUpperCase()}: ${message.message}`;
    
    messageDiv.appendChild(timestamp);
    messageDiv.appendChild(content);
    
    statusLog.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    statusLog.scrollTop = statusLog.scrollHeight;
    
    // Keep only last 50 messages
    while (statusLog.children.length > 50) {
        statusLog.removeChild(statusLog.firstChild);
    }
}

// Update status messages from world state
function updateStatusMessages(messages) {
    const statusLog = document.getElementById('status-log');
    if (!statusLog || !messages) return;
    
    // Only update if we have new messages
    const currentCount = statusLog.children.length;
    if (messages.length <= currentCount) return;
    
    // Add new messages
    for (let i = currentCount; i < messages.length; i++) {
        addStatusMessage(messages[i]);
    }
}

// Update agent status under conveyor
function updateAgentStatus(message) {
    const agent = message.agent.toLowerCase();
    const text = message.message.toLowerCase();
    
    let statusElement, statusText;
    
    if (agent === 'vision') {
        statusElement = document.getElementById('vision-agent-status');
        if (text.includes('detected')) {
            statusText = `Vision: Detected ${getObjectFromMessage(text)}`;
        }
    } else if (agent === 'scoring') {
        statusElement = document.getElementById('scoring-agent-status');
        if (text.includes('assigned')) {
            statusText = `Score: Assigned ${getLaneFromMessage(text)}`;
        } else if (text.includes('confirmed')) {
            statusText = `Score: Confirmed ${getObjectFromMessage(text)}`;
        }
    } else if (agent === 'trigger') {
        statusElement = document.getElementById('trigger-agent-status');
        if (text.includes('watching')) {
            statusText = `Trigger: Watching ${getObjectFromMessage(text)}`;
        } else if (text.includes('approaching')) {
            statusText = `Trigger: Object approaching`;
        }
    } else if (agent === 'cnc') {
        statusElement = document.getElementById('cnc-agent-status');
        if (text.includes('ready')) {
            statusText = 'CNC R: Ready';
            // Show vision lane flash when CNC sends ready notice
            const laneMatch = text.match(/lane (\d+)/);
            if (laneMatch) {
                showVisionLaneFlash(laneMatch[1]);
            }
        } else if (text.includes('assigned')) {
            statusText = `CNC: Assigned ${getLaneFromMessage(text)}`;
        } else if (text.includes('pick')) {
            statusText = 'CNC: Pick';
        } else if (text.includes('moving') || text.includes('position')) {
            statusText = 'CNC: Moving';
        } else if (text.includes('dropping') || text.includes('drop')) {
            statusText = 'CNC: Dropping';
        } else if (text.includes('homing') || text.includes('home')) {
            statusText = 'CNC: Homing';
        }
    }
    
    if (statusElement && statusText) {
        showSimpleStatus(statusElement, statusText);
    }
}

// Helper functions to extract info from message
function getObjectFromMessage(text) {
    const match = text.match(/obj_\w+/);
    return match ? match[0] : '';
}

function getLaneFromMessage(text) {
    const match = text.match(/lane (\d+)/);
    return match ? `Lane ${match[1]}` : '';
}

// Show simple status update
function showSimpleStatus(element, statusText) {
    if (!element) return;
    
    element.textContent = statusText;
    element.classList.add('status-active');
    
    // Clear after 3 seconds
    setTimeout(() => {
        const agent = element.id.split('-')[0];
        element.textContent = `${agent.charAt(0).toUpperCase() + agent.slice(1)}: Idle`;
        element.classList.remove('status-active');
    }, 3000);
}

// Show vision lane flash for 2 seconds
function showVisionLaneFlash(laneNumber) {
    const visionLaneElement = document.getElementById('vision-lane');
    if (!visionLaneElement) return;
    
    visionLaneElement.textContent = `Vision: Lane ${laneNumber}`;
    visionLaneElement.style.display = 'inline';
    visionLaneElement.style.color = '#FFFF00';  // Yellow flash
    visionLaneElement.style.fontWeight = 'bold';
    
    // Flash effect
    let flashCount = 0;
    const flashInterval = setInterval(() => {
        visionLaneElement.style.visibility = visionLaneElement.style.visibility === 'hidden' ? 'visible' : 'hidden';
        flashCount++;
        if (flashCount >= 8) {  // Flash 4 times (8 visibility changes)
            clearInterval(flashInterval);
            visionLaneElement.style.visibility = 'visible';
        }
    }, 250);
    
    // Hide after 2 seconds
    setTimeout(() => {
        clearInterval(flashInterval);
        visionLaneElement.style.display = 'none';
        visionLaneElement.style.color = '';
        visionLaneElement.style.fontWeight = '';
        visionLaneElement.style.visibility = 'visible';
    }, 2000);
}

// Convert mm to canvas pixels
function mmToPixels(mm) {
    // With 500mm belt and 1000px canvas, scale is 2
    return mm * SCALE;
}

// Draw conveyor belt
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
    
    // Trigger zone line at 250mm = 500px (purple, flashes yellow)
    const triggerX = mmToPixels(CONFIG.trigger_zone);
    const isTriggerFlashing = (Date.now() - triggerFlashTime) < 500; // Flash for 0.5 seconds
    
    if (isTriggerFlashing) {
        // Flash yellow when green object detected
        ctx.strokeStyle = '#FFFF00';  // Bright yellow
        ctx.lineWidth = 4;
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'yellow';
    } else {
        ctx.strokeStyle = '#9C27B0';  // Purple for trigger
        ctx.lineWidth = 2;
        ctx.shadowBlur = 0;
    }
    
    ctx.beginPath();
    ctx.moveTo(triggerX, 50);
    ctx.lineTo(triggerX, 250);
    ctx.stroke();
    
    // Reset shadow
    ctx.shadowBlur = 0;
    
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
    // Flash RED if a missed pickup was detected recently, YELLOW for motion detection
    const postPickX = mmToPixels(CONFIG.post_pick_zone);
    const isFlashingRed = (Date.now() - missedPickupTime) < 2000; // Flash for 2 seconds
    const isFlashingYellow = (Date.now() - monitorFlashTime) < 500; // Flash for 0.5 seconds
    
    if (isFlashingRed) {
        // Pulse effect for missed pickup (RED takes priority)
        const flashIntensity = Math.sin((Date.now() - missedPickupTime) * 0.01) * 0.5 + 0.5;
        ctx.strokeStyle = `rgba(255, 0, 0, ${0.5 + flashIntensity * 0.5})`;
        ctx.lineWidth = 3 + flashIntensity * 2;
        
        // Draw red glow effect
        ctx.shadowBlur = 10;
        ctx.shadowColor = 'red';
    } else if (isFlashingYellow) {
        // Flash yellow for motion detection
        ctx.strokeStyle = '#FFFF00';  // Bright yellow
        ctx.lineWidth = 4;
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'yellow';
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
}

// Draw object on belt
function drawObject(obj) {
    const x = mmToPixels(obj.position_x);
    const laneHeight = 200 / CONFIG.lanes;
    const y = 50 + (obj.lane * laneHeight) + (laneHeight / 2);
    const radius = 15;
    
    // Draw object circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = OBJECT_COLORS[obj.type] || '#999';
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Draw ring if present
    if (obj.has_ring) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 5, 0, 2 * Math.PI);
        ctx.strokeStyle = obj.ring_color || 'yellow';
        ctx.lineWidth = 3;
        ctx.stroke();
    }
    
    // Draw red checkmark if this object is assigned to CNC
    if (worldState.confirmed_target && worldState.confirmed_target.object_id === obj.id) {
        ctx.strokeStyle = '#FF0000'; // Red color
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        
        // Draw checkmark (two lines forming a check)
        ctx.beginPath();
        // Short line (bottom left to middle)
        ctx.moveTo(x - 6, y);
        ctx.lineTo(x - 2, y + 4);
        // Long line (middle to top right)
        ctx.moveTo(x - 2, y + 4);
        ctx.lineTo(x + 6, y - 4);
        ctx.stroke();
        
        ctx.lineCap = 'butt'; // Reset line cap
    }
    
    // Debug: show position (remove in production)
    ctx.fillStyle = '#fff';
    ctx.font = '8px Arial';
    ctx.fillText(Math.round(obj.position_x), x - 10, y + 3);
}

// Draw CNC picker
function drawCNC() {
    if (!worldState.cnc) return;
    
    const x = mmToPixels(worldState.cnc.position_x);
    const size = 30;
    
    // Draw vertical line showing CNC position
    ctx.strokeStyle = 'rgba(139, 69, 19, 0.3)';  // Transparent brown
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(x, 50);
    ctx.lineTo(x, 250);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // CNC arm (brown square) positioned above belt
    ctx.fillStyle = '#8B4513';
    ctx.fillRect(x - size/2, 20 - size/2, size, size);
    
    // Add yellow outline when CNC is activated (not idle)
    if (worldState.cnc.status !== 'idle') {
        ctx.strokeStyle = '#FFFF00';  // Bright yellow
        ctx.lineWidth = 3;
        ctx.shadowBlur = 6;
        ctx.shadowColor = 'yellow';
    } else {
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        ctx.shadowBlur = 0;
    }
    ctx.strokeRect(x - size/2, 20 - size/2, size, size);
    
    // Reset shadow
    ctx.shadowBlur = 0;
    
    // Show status text
    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(worldState.cnc.status, x, 10);
    
    // Show assigned lane number as text under status
    if (worldState.cnc && worldState.cnc.assigned_lane !== undefined) {
        ctx.fillStyle = '#000';  // Black text
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Assigned: Lane ${worldState.cnc.assigned_lane}`, x, 50);
    }
    
    ctx.textAlign = 'start';
    
    // Show if carrying object
    if (worldState.cnc.has_object) {
        ctx.beginPath();
        ctx.arc(x, 20, 8, 0, 2 * Math.PI);
        ctx.fillStyle = OBJECT_COLORS.green;
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// Draw robot arm icon that tracks operational locations
function drawRobotArm() {
    if (!worldState.cnc) return;
    
    let newTargetX, newTargetY;
    const armSize = 40; // Increased size for visibility
    
    // Determine target position based on CNC status and assigned lane
    switch (worldState.cnc.status) {
        case 'preparing':
        case 'waiting_for_trigger':
            // Show at assigned lane position
            if (worldState.cnc.assigned_lane !== undefined) {
                newTargetX = mmToPixels(337.5); // Pickup zone center
                const laneHeight = 200 / CONFIG.lanes;
                newTargetY = 50 + (worldState.cnc.assigned_lane * laneHeight) + (laneHeight / 2);
            } else {
                newTargetX = mmToPixels(337.5);
                newTargetY = 150; // Center
            }
            break;
            
        case 'picking':
            // Show at pickup location (assigned lane)
            if (worldState.cnc.assigned_lane !== undefined) {
                newTargetX = mmToPixels(337.5);
                const laneHeight = 200 / CONFIG.lanes;
                newTargetY = 50 + (worldState.cnc.assigned_lane * laneHeight) + (laneHeight / 2);
            } else {
                newTargetX = mmToPixels(337.5);
                newTargetY = 150;
            }
            break;
            
        case 'moving_to_bin':
            // Show at bin location
            newTargetX = mmToPixels(337.5);
            newTargetY = 270; // Bin area
            break;
            
        case 'returning_home':
            // Show returning to home position at pickup zone start, center of belt
            newTargetX = mmToPixels(300); // At pickup zone start (first pickup line)
            newTargetY = 150; // Center of belt
            break;
            
        case 'idle':
        default:
            // Show at ready position at pickup zone start, center of belt
            newTargetX = mmToPixels(300); // At pickup zone start (first pickup line)
            newTargetY = 150; // Center of belt
            break;
    }
    
    // Update target position
    robotArm.targetX = newTargetX;
    robotArm.targetY = newTargetY;
    
    // Smooth interpolation to target position
    robotArm.currentX += (robotArm.targetX - robotArm.currentX) * robotArm.animationSpeed;
    robotArm.currentY += (robotArm.targetY - robotArm.currentY) * robotArm.animationSpeed;
    
    // Use current position for drawing
    const targetX = robotArm.currentX;
    const targetY = robotArm.currentY;
    
    // Draw robot arm claw icon
    ctx.save();
    
    // Main arm body
    ctx.fillStyle = '#000000'; // Black for high contrast
    ctx.strokeStyle = '#FFFFFF'; // White border
    ctx.lineWidth = 3;
    
    // Draw arm base (rectangular)
    ctx.fillRect(targetX - armSize/2, targetY - armSize/2, armSize, armSize/2);
    ctx.strokeRect(targetX - armSize/2, targetY - armSize/2, armSize, armSize/2);
    
    // Draw claw parts (two small rectangles)
    const clawSize = 8; // Larger claw
    ctx.fillStyle = '#FF0000'; // Bright red for claw
    
    // Left claw
    ctx.fillRect(targetX - clawSize - 2, targetY, clawSize, 4);
    ctx.strokeRect(targetX - clawSize - 2, targetY, clawSize, 4);
    
    // Right claw
    ctx.fillRect(targetX + 2, targetY, clawSize, 4);
    ctx.strokeRect(targetX + 2, targetY, clawSize, 4);
    
    // Add status-based effects
    if (worldState.cnc.status === 'picking') {
        // Highlight when picking
        ctx.shadowBlur = 8;
        ctx.shadowColor = '#E74C3C';
        ctx.strokeStyle = '#E74C3C';
        ctx.lineWidth = 3;
        ctx.strokeRect(targetX - armSize/2 - 2, targetY - armSize/2 - 2, armSize + 4, armSize/2 + 4);
    } else if (worldState.cnc.status !== 'idle') {
        // Subtle glow when active
        ctx.shadowBlur = 4;
        ctx.shadowColor = '#3498DB';
    }
    
    // Show carried object
    if (worldState.cnc.has_object) {
        ctx.beginPath();
        ctx.arc(targetX, targetY + armSize/2 + 5, 5, 0, 2 * Math.PI);
        ctx.fillStyle = OBJECT_COLORS.green;
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    ctx.restore();
}

// Draw collection bin
function drawBin() {
    // Bin positioned at center of pickup zone (337.5mm = 675px)
    const binX = mmToPixels(337.5);
    const binY = 270;
    const binWidth = 60;
    const binHeight = 25;
    
    // Check for flash
    const isFlashing = (Date.now() - binFlashTime) < 300;
    
    // Bin body
    ctx.fillStyle = isFlashing ? '#4CAF50' : '#666';
    ctx.fillRect(binX - binWidth/2, binY, binWidth, binHeight);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.strokeRect(binX - binWidth/2, binY, binWidth, binHeight);
    
    // Bin label
    ctx.fillStyle = '#999';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('BIN', binX, binY + binHeight + 12);
    ctx.textAlign = 'start';
}


// Main render loop
function render() {
    // Clear canvas
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Draw conveyor
    drawConveyor();
    
    // Draw collection bin
    drawBin();
    
    // Draw objects
    if (worldState.objects) {
        Object.values(worldState.objects).forEach(obj => {
            drawObject(obj);
        });
    }
    
    // Draw CNC
    drawCNC();
    
    // Draw robot arm tracking icon (on top of everything)
    drawRobotArm();
    
    // Continue animation
    animationFrame = requestAnimationFrame(render);
}

// Start visualization
initSocket();
render();
