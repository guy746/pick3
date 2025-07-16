#!/usr/bin/env python3
"""
Test script for GRBL Interface Agent
Demonstrates FoxAlien 3S GRBL controller simulation
"""

import time
import threading
from grbl_interface_agent import GrblInterfaceAgent

def test_grbl_interface():
    """Test GRBL interface functionality"""
    print("=== GRBL Interface Test ===")
    
    # Create GRBL interface
    grbl = GrblInterfaceAgent()
    
    # Start GRBL in separate thread
    grbl_thread = threading.Thread(target=grbl.run, daemon=True)
    grbl_thread.start()
    
    # Get serial interface
    serial = grbl.get_serial_interface()
    
    # Wait for startup
    print("Waiting for GRBL startup...")
    time.sleep(1)
    
    # Read startup message
    while serial.in_waiting():
        response = serial.readline().decode().strip()
        print(f"GRBL: {response}")
    
    print("\n=== Testing Commands ===")
    
    # Test commands
    commands = [
        "$",        # Help
        "$I",       # Version info
        "$$",       # Settings
        "$G",       # Parser state
        "$#",       # Parameters
        "G28",      # Home
        "G1 X10 Y10 F1000",  # Move
        "M3 S1000", # Spindle on
        "G1 X0 Y0",  # Return
        "M5",       # Spindle off
        "G4 P500",  # Dwell
    ]
    
    for cmd in commands:
        print(f"\nSending: {cmd}")
        serial.write(cmd + '\n')
        
        # Read responses
        timeout = time.time() + 2.0
        while time.time() < timeout:
            if serial.in_waiting():
                response = serial.readline().decode().strip()
                if response:
                    print(f"  Response: {response}")
                    if response == 'ok' or response.startswith('error:'):
                        break
            time.sleep(0.01)
    
    print("\n=== Testing Status Reports ===")
    
    # Test status reports
    for i in range(5):
        print(f"\nStatus request {i+1}:")
        serial.write('?')
        
        # Wait for status
        timeout = time.time() + 0.5
        while time.time() < timeout:
            if serial.in_waiting():
                response = serial.readline().decode().strip()
                if response.startswith('<') and response.endswith('>'):
                    print(f"  Status: {response}")
                    break
            time.sleep(0.01)
        
        time.sleep(0.2)
    
    print("\n=== Testing Real-time Commands ===")
    
    # Start a long move
    print("\nStarting long move...")
    serial.write('G1 X100 Y100 F500\n')
    time.sleep(0.1)
    
    # Send status requests during move
    for i in range(3):
        time.sleep(0.5)
        serial.write('?')
        
        # Read status
        timeout = time.time() + 0.5
        while time.time() < timeout:
            if serial.in_waiting():
                response = serial.readline().decode().strip()
                if response.startswith('<'):
                    print(f"  Moving: {response}")
                    break
                elif response == 'ok':
                    print("  Move complete")
                    break
            time.sleep(0.01)
    
    # Test feed hold
    print("\nTesting feed hold...")
    serial.write('G1 X0 Y0 F500\n')
    time.sleep(0.1)
    
    # Hold
    serial.write('!')
    time.sleep(0.5)
    serial.write('?')
    
    timeout = time.time() + 0.5
    while time.time() < timeout:
        if serial.in_waiting():
            response = serial.readline().decode().strip()
            if response.startswith('<'):
                print(f"  Held: {response}")
                break
        time.sleep(0.01)
    
    # Resume
    time.sleep(0.5)
    serial.write('~')
    
    # Wait for completion
    time.sleep(2)
    while serial.in_waiting():
        response = serial.readline().decode().strip()
        if response:
            print(f"  Resume: {response}")
    
    print("\n=== Test Complete ===")

def test_error_handling():
    """Test GRBL error handling"""
    print("\n=== Error Handling Test ===")
    
    grbl = GrblInterfaceAgent()
    grbl_thread = threading.Thread(target=grbl.run, daemon=True)
    grbl_thread.start()
    
    serial = grbl.get_serial_interface()
    time.sleep(1)
    
    # Clear startup
    while serial.in_waiting():
        serial.readline()
    
    # Test invalid commands
    invalid_commands = [
        "INVALID",      # Invalid command
        "G999",         # Invalid G-code
        "X1000000",     # Out of bounds
        "G1 X-1000",    # Negative limit
    ]
    
    for cmd in invalid_commands:
        print(f"\nTesting invalid: {cmd}")
        serial.write(cmd + '\n')
        
        timeout = time.time() + 1.0
        while time.time() < timeout:
            if serial.in_waiting():
                response = serial.readline().decode().strip()
                if response:
                    print(f"  Response: {response}")
                    break
            time.sleep(0.01)

def interactive_test():
    """Interactive GRBL test"""
    print("\n=== Interactive GRBL Test ===")
    print("Enter GRBL commands (or 'quit' to exit)")
    
    grbl = GrblInterfaceAgent()
    grbl_thread = threading.Thread(target=grbl.run, daemon=True)
    grbl_thread.start()
    
    serial = grbl.get_serial_interface()
    time.sleep(1)
    
    # Clear startup
    while serial.in_waiting():
        response = serial.readline().decode().strip()
        print(f"GRBL: {response}")
    
    try:
        while True:
            command = input("\nGRBL> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
                
            if not command:
                continue
                
            # Send command
            serial.write(command + '\n')
            
            # Read responses
            timeout = time.time() + 2.0
            responses = []
            
            while time.time() < timeout:
                if serial.in_waiting():
                    response = serial.readline().decode().strip()
                    if response:
                        responses.append(response)
                        print(f"  {response}")
                        
                        # Stop on ok/error for most commands
                        if response in ['ok'] or response.startswith('error:'):
                            if not command == '?':  # Status requests don't send ok
                                break
                        elif command == '?' and response.startswith('<'):
                            break
                            
                time.sleep(0.01)
            
            if not responses:
                print("  (no response)")
                
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    # Run basic test
    test_grbl_interface()
    
    # Test error handling
    test_error_handling()
    
    # Interactive mode
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\nTest completed")