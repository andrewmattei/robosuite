"""
Minimal RBY1 robot loader - just displays the robot model.
No simulation, no IK, no external dependencies.
"""

from pathlib import Path
import mujoco
import mujoco.viewer

_HERE = Path(__file__).parent
_XML = _HERE / "rby1a" / "mujoco" / "model_act.xml"

def main():
    """Load and display the RBY1 robot model"""
    
    # Check if model file exists
    if not _XML.exists():
        print(f"Error: Model file not found at {_XML}")
        print("Expected structure:")
        print("  rby1_teleop/")
        print("    ├── rby1a/")
        print("    │   └── mujoco/")
        print("    │       └── model_act.xml")
        print("    └── rby1_minimal.py")
        return
    
    try:
        # Load MuJoCo model
        print(f"Loading RBY1 model from: {_XML}")
        model = mujoco.MjModel.from_xml_path(str(_XML))
        data = mujoco.MjData(model)
        
        print("Model loaded successfully!")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.nq}") 
        print(f"  Actuators: {model.nu}")
        
        # Reset to initial state
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Launch viewer
    print("\nLaunching viewer...")
    print("Controls:")
    print("  Mouse: Rotate/pan camera")  
    print("  Scroll: Zoom in/out")
    print("  ESC: Close viewer")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera position for good view of robot
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0, 1.2]
        
        # Keep viewer open until user closes it
        while viewer.is_running():
            # Just update the viewer, no simulation
            viewer.sync()

if __name__ == "__main__":
    main()
