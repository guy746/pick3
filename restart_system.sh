#!/bin/bash
# Restart script for Pick1WebSim system

echo "=== Pick1WebSim System Restart ==="

# Kill existing processes
echo "Stopping existing processes..."
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "python.*test_data.py" 2>/dev/null || true  
pkill -f "python.*scoring_agent.py" 2>/dev/null || true
pkill -f "python.*vision_agent.py" 2>/dev/null || true
pkill -f "python.*cnc_agent.py" 2>/dev/null || true

# Wait for processes to stop
sleep 3

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Starting Redis..."
    redis-server --daemonize yes
    sleep 2
fi

# Activate virtual environment
source venv/bin/activate

# Clear Redis state (optional)
echo "Clearing Redis state..."
redis-cli flushall

# Start services in order
echo "Starting Flask app on port 5001..."
PORT=5001 python app.py &
APP_PID=$!
sleep 3

echo "Starting scoring agent..."
python scoring_agent.py &
SCORING_PID=$!
sleep 2

echo "Starting vision agent..."
python vision_agent.py &
VISION_PID=$!
sleep 2

echo "Starting CNC agent..."
python cnc_agent.py &
CNC_PID=$!
sleep 2

echo "Starting trigger camera agent..."
python trigger_camera_agent.py &
TRIGGER_PID=$!
sleep 2

echo "Starting test data generator..."
python test_data.py &
TEST_DATA_PID=$!
sleep 2

# Display running processes
echo "=== System Status ==="
echo "Flask app (PID: $APP_PID) - http://localhost:5001"
echo "Scoring agent (PID: $SCORING_PID)"
echo "Vision agent (PID: $VISION_PID)"
echo "CNC agent (PID: $CNC_PID)"
echo "Trigger agent (PID: $TRIGGER_PID)"
echo "Test data (PID: $TEST_DATA_PID)"
echo ""
echo "Redis status: $(redis-cli ping)"
echo ""
echo "Running Python processes:"
ps aux | grep python | grep -E "(app\.py|scoring_agent\.py|test_data\.py)" | grep -v grep

echo ""
echo "=== System started successfully! ==="
echo "Access the web interface at: http://localhost:5001"