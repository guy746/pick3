#!/bin/bash
# Run the Pick1 system with all agents (Vision, Trigger Camera, CNC, Post-Pick Monitor, Scoring)

echo "Starting Pick1 system with all modular agents..."
echo "================================================="
echo "Agents: Vision, Trigger, CNC, Monitor, Scoring"
echo "================================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "ERROR: Redis is not running!"
    echo "Please start Redis first: redis-server"
    exit 1
fi

# Clear any existing Redis data (optional)
echo "Clearing old Redis data..."
redis-cli FLUSHDB

# Start vision agent in background
echo "Starting Vision Agent..."
cd pick1websim && python3 vision_agent.py &
VISION_PID=$!
echo "Vision Agent PID: $VISION_PID"

# Give vision agent time to start
sleep 1

# Start trigger camera agent in background
echo "Starting Trigger Camera Agent..."
cd pick1websim && python3 trigger_camera_agent.py &
TRIGGER_PID=$!
echo "Trigger Camera PID: $TRIGGER_PID"

# Give trigger camera time to start
sleep 1

# Start CNC agent in background
echo "Starting CNC Agent..."
cd pick1websim && python3 cnc_agent.py &
CNC_PID=$!
echo "CNC Agent PID: $CNC_PID"

# Give CNC agent time to start
sleep 1

# Start post-pick monitor agent in background
echo "Starting Post-Pick Monitor Agent..."
cd pick1websim && python3 post_pick_monitor_agent.py &
MONITOR_PID=$!
echo "Post-Pick Monitor PID: $MONITOR_PID"

# Give monitor agent time to start
sleep 1

# Start test data generator in background
echo "Starting Test Data Generator..."
cd pick1websim && python3 test_data.py &
TEST_PID=$!
echo "Test Data PID: $TEST_PID"

# Give test data time to initialize
sleep 1

# Start scoring agent in background
echo "Starting Scoring Agent..."
cd pick1websim && python3 scoring_agent.py &
SCORING_PID=$!
echo "Scoring Agent PID: $SCORING_PID"

# Start Flask app (this will block)
echo "Starting Flask Web App..."
echo ""
echo "Open http://localhost:5000 in your browser"
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $VISION_PID 2>/dev/null
    kill $TRIGGER_PID 2>/dev/null
    kill $CNC_PID 2>/dev/null
    kill $MONITOR_PID 2>/dev/null
    kill $TEST_PID 2>/dev/null
    kill $SCORING_PID 2>/dev/null
    echo "All services stopped."
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup INT

# Run Flask app with development server (this blocks)
# NOTE: Gunicorn commented out due to WebSocket compatibility issues
# cd pick1websim && source /home/guy/test11/venv/bin/activate && gunicorn -w 1 --bind 0.0.0.0:5000 app:app
cd pick1websim && source /home/guy/test11/venv/bin/activate && python3 app.py

# If we get here, Flask exited
cleanup
