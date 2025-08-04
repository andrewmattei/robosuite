"""
RBY1 robot teleoperation using MediaPipe pose estimation.
Integrates MediaPipe device and SEW mimic controller for human pose teleoperation.
Author: Chuizheng Kong
Date created: 2025-07-27
"""

from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import traceback

# Add the current directory to Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mediapipe_teleop_device import MediaPipeTeleopDevice
from sew_mimic_rby1 import SEWMimicRBY1

_HERE = Path(__file__).parent
_XML = _HERE / "rby1a" / "mujoco" / "model.xml"   # parallel jaw gripper version
# Joint names for reference (not all used in this simple version)
# joint_names = [
#     # Base joints
#     "left_wheel", "right_wheel",
#     # Arm joints  
#     "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
#     "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
#     "gripper_finger_r1", "gripper_finger_r2",
#     "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6", 
#     "gripper_finger_l1", "gripper_finger_l2",
#     "head_0", "head_1",
# ]

# _XML = _HERE / "rbyxhand_v2" / "model.xml"  # RBY1 with RBYX hand version
# joint_names = [
#     # Base joints
#     "left_wheel", "right_wheel",
#     # Arm joints  
#     "torso_0", "torso_1", "torso_2", "torso_3", "torso_4",
#     "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
#     "left_arm_0", "left_arm_1", "left_arm_2",| "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6", 
#     "head_0", "head_1",
# ]


class TeleopKeyCallback:
    """Key callback for teleoperation control"""
    
    def __init__(self):
        self.reset_requested = False
        self.teleop_enabled = False
        self.home_requested = False
        
    def __call__(self, key: int) -> None:
        if key == ord('r') or key == ord('R'):
            self.reset_requested = True
            print("Reset requested")
        elif key == ord('q') or key == ord('Q'):
            print("Quit requested")
            return False
        elif key == ord('t') or key == ord('T'):
            self.teleop_enabled = not self.teleop_enabled
            status = "ENABLED" if self.teleop_enabled else "DISABLED"
            print(f"Teleoperation {status}")
        elif key == ord('h') or key == ord('H'):
            self.home_requested = True
            print("Home position requested")
        elif key == 32:  # Space bar
            print("Space pressed")
        else:
            print(f"Key pressed: {key}")

def main():
    """Main function to load RBY1 robot and start teleoperation"""
    
    # Check if XML file exists
    if not _XML.exists():
        print(f"Error: XML file not found at {_XML}")
        print("Please ensure the RBY1 model files are in the correct location.")
        return
    
    try:
        # Load the MuJoCo model
        print(f"Loading model from: {_XML}")
        model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        data = mujoco.MjData(model)
        
        print(f"Model loaded successfully!")
        print(f"Number of joints: {model.nq}")
        print(f"Number of actuators: {model.nu}")
        print(f"Number of bodies: {model.nbody}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return
    
    # Initialize teleoperation components
    print("Initializing teleoperation system...")
    
    try:
        # Initialize MediaPipe device
        device = MediaPipeTeleopDevice(camera_id=0, debug=True, mirror_actions=True)
        device.start_control()
        
        # Initialize SEW mimic controller
        controller = SEWMimicRBY1(model, data, debug=False)
        
        print("Teleoperation system initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing teleoperation system: {e}")
        traceback.print_exc()
        print("Falling back to simple visualization mode...")
        device = None
        controller = None
    
    # Create key callback
    key_callback = TeleopKeyCallback()
    
    # Launch passive viewer
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=True,
        key_callback=key_callback,
    ) as viewer:
        
        # Configure viewer settings
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        
        # Set a good default camera view
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]
        
        print("Viewer launched. Controls:")
        print("  R - Reset robot pose")
        print("  Q - Quit")
        print("  T - Toggle teleoperation (if available)")
        print("  H - Move to home position")
        print("  Mouse - Rotate/pan camera")
        print("  Scroll - Zoom")
        
        if device is not None:
            print("\nTeleoperation Instructions:")
            print("1. Make sure you're visible in the MediaPipe camera window")
            print("2. Press 'T' to enable teleoperation")
            print("3. Raise both arms to shoulder height to start control")
            print("4. Move your arms to control the robot")
        
        # Initialize robot to home position
        mujoco.mj_resetData(model, data)
        
        # Set all joint positions to zero (neutral pose)
        data.qpos[:] = 0
        
        # Forward kinematics to update positions
        mujoco.mj_forward(model, data)
        
        # Main simulation loop
        timestep = model.opt.timestep
        last_print_time = time.time()
        
        while viewer.is_running():
            
            # Handle key callbacks
            if key_callback.reset_requested:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = 0
                mujoco.mj_forward(model, data)
                key_callback.reset_requested = False
                print("Robot reset to home position")
                
                # Reset controller if available
                if controller is not None:
                    controller.reset_to_home_position()
            
            if key_callback.home_requested:
                if controller is not None:
                    controller.reset_to_home_position()
                key_callback.home_requested = False
            
            # Handle teleoperation
            # if (device is not None and controller is not None and 
            #     key_callback.teleop_enabled):
            if (device is not None and controller is not None):
                
                # Check for device quit/reset signals
                if device.should_quit():
                    print("MediaPipe device quit signal received")
                    break
                
                # Get human pose data
                try:
                    state = device.get_controller_state()
                    sew_left = state['sew_poses']['left']
                    sew_right = state['sew_poses']['right']
                    wrist_left = state['wrist_poses']['left'] 
                    wrist_right = state['wrist_poses']['right']
                    engaged = state['engaged']
                    
                    # Always update controller, passing the engagement flag from device
                    controller.update_control(sew_left, sew_right, wrist_left, wrist_right, engaged)
                    
                    if engaged:
                        # Print status occasionally when engaged
                        current_time = time.time()
                        if current_time - last_print_time > 2.0:  # Every 2 seconds
                            if controller.is_engaged():
                                q_right, q_left, q_torso = controller.get_current_joint_angles()
                                right_str = f"[{q_right[0]:.2f},{q_right[1]:.2f},{q_right[2]:.2f}]"
                                left_str = f"[{q_left[0]:.2f},{q_left[1]:.2f},{q_left[2]:.2f}]"
                                print(f"Teleoperation ENGAGED with IK - Right arm: {right_str}, Left arm: {left_str}")
                            else:
                                print("Teleoperation DISENGAGED - No valid SEW data")
                            last_print_time = current_time
                    else:
                        # Print disengaged status occasionally
                        current_time = time.time()
                        if current_time - last_print_time > 5.0:  # Every 5 seconds when disengaged
                            print("Waiting for human engagement (raise both arms to shoulder height)")
                            last_print_time = current_time
                        
                except Exception as e:
                    print(f"Error in teleoperation loop: {e}")
                    traceback.print_exc()
                    # On error, disengage
                    controller.update_control(None, None, None, None, False)
            else:
                # No teleoperation - set all actuator commands to zero or hold position
                if controller is not None:
                    # Apply control to hold current position (includes torso gravity compensation)
                    torques_right, torques_left, torques_torso = controller.compute_control_torques()
                    controller.apply_torques(torques_right, torques_left, torques_torso)
                else:
                    # No controller - just set zero torques
                    data.ctrl[:] = 0
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Sleep to maintain real-time rate
            time.sleep(timestep)
    
    # Cleanup
    print("Shutting down...")
    if device is not None:
        device.stop()
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
