#!/bin/bash
# Vision System Setup Script
# Sets up environment variables and dependencies for OAK-D Pro camera mode

echo "=== Vision System Setup ==="

# Check current mode
echo "Current vision mode (from config):"
grep "mode:" vision_config.yaml | head -1

echo ""
echo "Setup Options:"
echo "1. Configure for Simulation Mode"
echo "2. Configure for OAK-D Pro Camera Mode"
echo "3. Install Camera Dependencies"
echo "4. Test Current Configuration"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "Configuring for Simulation Mode..."
        sed -i 's/mode: ".*"/mode: "simulation"/' vision_config.yaml
        echo "✓ Set mode to simulation in vision_config.yaml"
        echo "✓ Ready to run: python vision_agent_enhanced.py simulation"
        ;;
        
    2)
        echo "Configuring for OAK-D Pro Camera Mode..."
        
        # Update config file
        sed -i 's/mode: ".*"/mode: "oak_d_pro"/' vision_config.yaml
        echo "✓ Set mode to oak_d_pro in vision_config.yaml"
        
        # Prompt for Roboflow credentials
        echo ""
        echo "Enter your Roboflow credentials:"
        read -p "API Key: " api_key
        read -p "Workspace Name: " workspace
        read -p "Project Name: " project
        
        # Create environment setup
        cat > .env << EOF
# Roboflow Configuration
export ROBOFLOW_API_KEY="$api_key"
export ROBOFLOW_WORKSPACE="$workspace"
export ROBOFLOW_PROJECT="$project"
EOF
        
        echo ""
        echo "✓ Created .env file with Roboflow credentials"
        echo "✓ To activate: source .env"
        echo "✓ Then run: python vision_agent_enhanced.py oak_d_pro"
        ;;
        
    3)
        echo "Installing Camera Dependencies..."
        echo "This will install DepthAI and Roboflow packages..."
        
        # Uncomment camera dependencies in requirements.txt
        sed -i 's/# depthai==/depthai==/' requirements.txt
        sed -i 's/# roboflow==/roboflow==/' requirements.txt
        
        # Install dependencies
        pip install -r requirements.txt
        
        echo "✓ Camera dependencies installed"
        ;;
        
    4)
        echo "Testing Current Configuration..."
        
        # Check config file
        echo "Current mode: $(grep 'mode:' vision_config.yaml | head -1 | cut -d'"' -f2)"
        
        # Check environment variables
        echo "Environment variables:"
        echo "  ROBOFLOW_API_KEY: ${ROBOFLOW_API_KEY:-'Not set'}"
        echo "  ROBOFLOW_WORKSPACE: ${ROBOFLOW_WORKSPACE:-'Not set'}"
        echo "  ROBOFLOW_PROJECT: ${ROBOFLOW_PROJECT:-'Not set'}"
        
        # Check dependencies
        echo "Dependencies:"
        python -c "import cv2; print('  OpenCV: ✓')" 2>/dev/null || echo "  OpenCV: ✗"
        python -c "import numpy; print('  NumPy: ✓')" 2>/dev/null || echo "  NumPy: ✗"
        python -c "import depthai; print('  DepthAI: ✓')" 2>/dev/null || echo "  DepthAI: ✗ (for camera mode)"
        python -c "import roboflow; print('  Roboflow: ✓')" 2>/dev/null || echo "  Roboflow: ✗ (for camera mode)"
        ;;
        
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "Setup complete!"