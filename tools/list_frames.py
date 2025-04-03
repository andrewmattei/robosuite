import pinocchio as pin

def print_all_frames(urdf_path):
    # Load the model
    model = pin.buildModelFromUrdf(urdf_path)
    
    print("\nAll frames in the model:")
    print("-" * 50)
    for frame in model.frames:
        print(f"Frame name: {frame.name}")
        print(f"Frame parent joint: {model.names[frame.parent]}")
        print(f"Frame ID: {frame.parentJoint}")
        print("-" * 50)

if __name__ == "__main__":
    urdf_path = "robosuite/models/assets/robots/dual_kinova3/leonardo.urdf"
    print_all_frames(urdf_path)
