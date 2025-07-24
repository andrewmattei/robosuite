"""
Simple RBY1 robot visualization without IK or complex motion.
Just loads and displays the robot in its initial pose.
"""

from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import time

_HERE = Path(__file__).parent
_XML = _HERE / "rby1a" / "mujoco" / "model_act.xml"

# Joint names for reference (not all used in this simple version)
joint_names = [
    # Base joints
    "left_wheel", "right_wheel",
    # Arm joints  
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "gripper_finger_r1", "gripper_finger_r2",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6", 
    "gripper_finger_l1", "gripper_finger_l2",
    "head_0", "head_1",
]

class SimpleKeyCallback:
    """Simple key callback for basic viewer control"""
    
    def __init__(self):
        self.reset_requested = False
        
    def __call__(self, key: int) -> None:
        if key == ord('r') or key == ord('R'):
            self.reset_requested = True
            print("Reset requested")
        elif key == ord('q') or key == ord('Q'):
            print("Quit requested")
            return False
        elif key == 32:  # Space bar
            print("Space pressed")
        else:
            print(f"Key pressed: {key}")

def main():
    """Main function to load and display the RBY1 robot"""
    
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
        return
    
    # Create key callback
    key_callback = SimpleKeyCallback()
    
    # Launch passive viewer
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=key_callback,
    ) as viewer:
        
        # Configure viewer settings
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        
        # Set a good default camera view
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0, 0, 1.0]
        
        print("Viewer launched. Controls:")
        print("  R - Reset robot pose")
        print("  Q - Quit")
        print("  Mouse - Rotate/pan camera")
        print("  Scroll - Zoom")
        
        # Initialize robot to home position
        mujoco.mj_resetData(model, data)
        
        # Set all joint positions to zero (neutral pose)
        data.qpos[:] = 0
        
        # Forward kinematics to update positions
        mujoco.mj_forward(model, data)
        
        # Main simulation loop
        timestep = model.opt.timestep
        while viewer.is_running():
            
            # Handle reset request
            if key_callback.reset_requested:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = 0
                mujoco.mj_forward(model, data)
                key_callback.reset_requested = False
                print("Robot reset to home position")
            
            # No control input - robot stays in place
            # Set all actuator commands to zero
            data.ctrl[:] = 0
            
            # Step the simulation (though nothing should move)
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Sleep to maintain real-time rate
            time.sleep(timestep)

if __name__ == "__main__":
    main()
