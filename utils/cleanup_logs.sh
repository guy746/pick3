#!/bin/bash
# Pick1 Log Cleanup Script

echo "Pick1 Log Cleanup Tool"
echo "====================="

# Show current log sizes
echo "Current log sizes:"
du -sh /var/log/supervisor/pick1_* | sort -hr

echo
read -p "Do you want to truncate all Pick1 logs? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Truncating logs..."
    sudo truncate -s 0 /var/log/supervisor/pick1_*.log
    echo "All Pick1 logs have been truncated."
else
    echo "No action taken."
fi

echo
echo "Log cleanup complete."