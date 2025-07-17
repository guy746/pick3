#!/bin/bash
# Simple startup script for Pick1WebSim system using Supervisor
# Usage: ./start_pick1.sh

echo "=== Pick1WebSim Supervisor Startup ==="
echo "$(date): Starting Pick1 system..."

# Check if supervisor is running
if ! sudo supervisorctl status > /dev/null 2>&1; then
    echo "ERROR: Supervisor service is not running"
    echo "Please start supervisor first: sudo systemctl start supervisor"
    exit 1
fi

# Stop all Pick1 services first (clean restart)
echo "Stopping existing Pick1 services..."
sudo supervisorctl stop pick1:* > /dev/null 2>&1

# Start all Pick1 services
echo "Starting Pick1 services via Supervisor..."
sudo supervisorctl start pick1:*

# Wait a moment for services to initialize
sleep 3

# Check status
echo ""
echo "=== Service Status ==="
sudo supervisorctl status pick1:*

# Check Redis connectivity
echo ""
echo "=== Redis Status ==="
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Connected"
else
    echo "❌ Redis: Connection failed"
fi

# Show web interface URL
echo ""
echo "=== Access Points ==="
echo "🌐 Web Interface: http://localhost:5000"
echo "📊 System Status: sudo supervisorctl status pick1:*"
echo "📋 Service Logs: sudo supervisorctl tail pick1:SERVICE_NAME"
echo ""
echo "✅ Pick1WebSim startup complete!"