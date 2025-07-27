#!/usr/bin/env python3
"""
Demo script for RBY1 robot teleoperation using MediaPipe pose estimation.

This script demonstrates the complete teleoperation pipeline:
1. MediaPipe captures human pose and extracts SEW coordinates
2. SEW mimic controller converts human poses to robot joint angles using IK
3. Joint torque control applies the computed angles to the robot

Usage:
    python demo_rby1_teleop.py [options]

Requirements:
    - MediaPipe: pip install mediapipe
    - OpenCV: pip install opencv-python
    - Pinocchio: pip install pin
    - MuJoCo: pip install mujoco
    - Webcam or camera device

Controls:
    - 'T' key in MuJoCo viewer: Toggle teleoperation
    - 'R' key: Reset robot to home position
    - 'H' key: Move to home position
    - 'Q' key: Quit
    - Raise both arms to shoulder height: Start pose tracking in MediaPipe window
    - 'q' key in MediaPipe window: Quit
"""

import argparse
import sys
import os
import time

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robosuite.projects.rby1_teleop.rby1_teleop import main


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RBY1 robot teleoperation using MediaPipe pose estimation")
    
    parser.add_argument("--camera-id", type=int, default=0, 
                       help="Camera device ID for pose estimation (default: 0)")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode for pose estimation and controller")
    parser.add_argument("--mirror-actions", action="store_true", default=True,
                       help="Mirror actions (right robot arm follows left human arm)")
    parser.add_argument("--kp", type=float, default=150.0,
                       help="Proportional gain for joint control (default: 150.0)")
    parser.add_argument("--kd", type=float, default=None,
                       help="Derivative gain for joint control (default: auto-computed)")
    
    return parser.parse_args()


def print_instructions():
    """Print detailed usage instructions."""
    print("=" * 80)
    print("RBY1 Robot Teleoperation Demo")
    print("=" * 80)
    print("This demo uses MediaPipe to track your arm movements and control the RBY1 robot.")
    print()
    print("SETUP INSTRUCTIONS:")
    print("1. Position yourself in front of the camera")
    print("2. Make sure your full upper body is visible in the MediaPipe window")
    print("3. The robot will start in home position")
    print()
    print("CONTROL INSTRUCTIONS:")
    print("1. Press 'T' in the MuJoCo viewer to enable teleoperation")
    print("2. In the MediaPipe window, raise both arms to shoulder height")
    print("3. Move your arms to control the robot - the robot will mimic your movements")
    print("4. Lower your arms to pause control")
    print()
    print("KEYBOARD CONTROLS (MuJoCo Viewer):")
    print("  T - Toggle teleoperation ON/OFF")
    print("  R - Reset robot to home position") 
    print("  H - Move robot to home position")
    print("  Q - Quit demo")
    print("  Mouse - Rotate/pan camera")
    print("  Scroll - Zoom")
    print()
    print("KEYBOARD CONTROLS (MediaPipe Window):")
    print("  q - Quit MediaPipe (will also quit demo)")
    print("  r - Reset pose tracking")
    print()
    print("NOTES:")
    print("- The robot uses geometric inverse kinematics to match your arm poses")
    print("- If IK is not available, the robot will engage for stable position holding")
    print("- Joint angles are controlled using inverse dynamics with mass matrix compensation")
    print("- The system works in real-time with pose estimation at ~30 FPS")
    print("- If IK fails for a pose, the robot will hold its current position")
    print("=" * 80)


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import mediapipe
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        import pinocchio
    except ImportError:
        missing_deps.append("pin")
    
    try:
        import mujoco
    except ImportError:
        missing_deps.append("mujoco")
    
    if missing_deps:
        print("ERROR: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print()
        print("Please install missing dependencies:")
        print(f"  pip install {' '.join(missing_deps)}")
        return False
    
    return True


def check_model_files():
    """Check if required model files exist."""
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    xml_file = current_dir / "rby1a" / "mujoco" / "model_act.xml"
    urdf_file = current_dir / "rbyxhand_v2" / "model_modified.urdf"
    
    missing_files = []
    
    if not xml_file.exists():
        missing_files.append(str(xml_file))
    
    if not urdf_file.exists():
        missing_files.append(str(urdf_file))
    
    if missing_files:
        print("ERROR: Missing required model files:")
        for file in missing_files:
            print(f"  - {file}")
        print()
        print("Please ensure the RBY1 model files are in the correct location.")
        return False
    
    return True


if __name__ == "__main__":
    
    # Parse arguments
    args = parse_arguments()
    
    # Print instructions
    print_instructions()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    print("Checking model files...")
    if not check_model_files():
        sys.exit(1)
    
    print("All checks passed!")
    print()
    
    # Wait for user to be ready
    try:
        input("Press Enter when you're ready to start the demo (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        sys.exit(0)
    
    print()
    print("Starting RBY1 teleoperation demo...")
    print("Please wait while the system initializes...")
    
    try:
        # Run the main demo
        main()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Demo completed.")
