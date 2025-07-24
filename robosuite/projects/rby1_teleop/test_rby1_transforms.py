#!/usr/bin/env python3
"""
Test script for RBY1 frame transforms function
"""

import numpy as np
from geometric_kinematics_rby1 import load_rby1_model, get_frame_transforms_from_pinocchio

def test_rby1_transforms():
    """Test the frame transforms function for different parts of RBY1"""
    
    print("Loading RBY1 model...")
    try:
        model, data = load_rby1_model()
        print(f"Model loaded successfully: {model.nq} DOF, {model.njoints} joints")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test different parts
    parts_to_test = ["right_arm", "left_arm", "torso", "base", "head"]
    
    for part_name in parts_to_test:
        print(f"\n{'='*50}")
        print(f"Testing part: {part_name}")
        print(f"{'='*50}")
        
        try:
            transforms = get_frame_transforms_from_pinocchio(model, part_name)
            
            print(f"Found {len(transforms['R'])} transformations")
            print(f"Joint names: {transforms['joint_names']}")
            
            for i, (R, p, h, name) in enumerate(zip(transforms['R'], transforms['p'], 
                                                   transforms['h'], transforms['joint_names'])):
                print(f"\nTransform {i}: {name}")
                print(f"  Rotation R_{i}_{i+1}:")
                print(f"    {R}")
                print(f"  Translation p_{i}_{i+1}: {p}")
                print(f"  Joint axis h_{i}: {h}")
                
                # Verify rotation matrix properties
                det_R = np.linalg.det(R)
                print(f"  det(R) = {det_R:.6f} (should be ~1.0)")
                
        except Exception as e:
            print(f"Error testing {part_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_rby1_transforms()
