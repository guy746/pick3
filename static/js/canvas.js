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
    });
    
        socket.on('world_update', (state) => {
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
    });
    
    socket.on('event', (event) => {
        console.log('Event:', event);
    });
}

// Update status display
function updateStatus() {
    const objectCount = Object.keys(worldState.objects).length;
    document.getElementById('object-count').textContent = `Objects: ${objectCount}`;
    document.getElementById('cnc-status').textContent = `CNC: ${worldState.cnc?.status || 'Unknown'}`;
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
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.strokeRect(x - size/2, 20 - size/2, size, size);
    
    // Show status text
    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(worldState.cnc.status, x, 10);
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

// Draw collection bin
function drawBin() {
    // Bin positioned at center of pickup zone (400mm = 800px)
    const binX = mmToPixels(400);
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
    
    // Continue animation
    animationFrame = requestAnimationFrame(render);
}

// Start visualization
initSocket();
render();
