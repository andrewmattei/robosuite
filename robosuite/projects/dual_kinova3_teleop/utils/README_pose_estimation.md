# MediaPipe Pose Estimation Scripts

This directory contains scripts for real-time human pose estimation using Google's MediaPipe library, with integration capabilities for RoboSuite robot control.

## Scripts Overview

### 1. `realtime_pose_estimation.py`
A standalone script for real-time pose estimation using a webcam.

**Features:**
- Real-time pose detection and tracking
- Landmark visualization with connections
- Performance metrics (FPS)
- Key pose angle calculations
- Screenshot and video recording capabilities
- Configurable detection parameters

**Usage:**
```bash
python realtime_pose_estimation.py [OPTIONS]
```

**Options:**
- `--camera_id`: Camera device ID (default: 0)
- `--confidence`: Minimum detection confidence (default: 0.5)
- `--complexity`: Model complexity 0-2 (default: 1)
- `--smooth`: Enable landmark smoothing (default: True)
- `--save_output`: Save output video
- `--output_path`: Output video file path
- `--width`: Video width (default: 640)
- `--height`: Video height (default: 480)

**Controls:**
- `q`: Quit
- `s`: Save screenshot
- `r`: Reset pose tracking
- `a`: Toggle angle display

### 2. `pose_to_robot_control.py`
Demonstrates integration between pose estimation and RoboSuite robot control.

**Features:**
- Maps human poses to robot actions
- Calibratable reference pose system
- Real-time robot control via pose gestures
- Visual feedback of control commands
- Support for various RoboSuite environments

**Usage:**
```bash
python pose_to_robot_control.py [OPTIONS]
```

**Options:**
- `--env`: RoboSuite environment name (default: 'Lift')
- `--robot`: Robot name (default: 'Panda')
- `--camera_id`: Camera device ID (default: 0)
- `--control_freq`: Control frequency Hz (default: 20)
- `--simulation_only`: Run without camera input

**Controls:**
- `q`: Quit
- `c`: Calibrate reference pose
- `r`: Reset environment
- `s`: Save screenshot

## Installation

### Prerequisites
Make sure you have Python 3.7+ installed.

### Install Dependencies
```bash
# Install MediaPipe pose estimation dependencies
pip install -r requirements-pose-estimation.txt

# For RoboSuite integration (if not already installed)
pip install robosuite
```

### Alternative Installation
```bash
# Individual package installation
pip install mediapipe>=0.10.0
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
```

## Getting Started

### Basic Pose Estimation
1. Connect a webcam to your computer
2. Run the basic pose estimation script:
   ```bash
   python realtime_pose_estimation.py
   ```
3. Position yourself in front of the camera
4. Use the controls to interact with the application

### Robot Control Integration
1. Ensure RoboSuite is properly installed
2. Run the pose-to-robot control script:
   ```bash
   python pose_to_robot_control.py --env Lift --robot Panda
   ```
3. Calibrate your reference pose by pressing 'c' and holding still
4. Use your body movements to control the robot

## Pose Mapping Details

### Body-to-Robot Mapping
The pose-to-robot control system maps human body movements as follows:

- **Right Arm → Robot End-Effector Position**
  - Right wrist X/Y/Z movements control robot TCP position
  - Movements are relative to calibrated reference pose

- **Left Arm → Robot Orientation**
  - Left wrist movements influence robot end-effector orientation
  - Simplified roll/pitch control

- **Hand Distance → Gripper Control**
  - Distance between hands controls gripper open/close
  - Threshold-based activation

### Calibration Process
1. Stand in a neutral, comfortable pose
2. Press 'c' to start calibration
3. Hold still for 2 seconds while the system records your reference pose
4. After calibration, movements are tracked relative to this pose

## Customization

### Adjusting Pose Sensitivity
In `pose_to_robot_control.py`, modify these parameters:

```python
# In PoseToRobotController.__init__()
self.arm_scale = 2.0          # Arm movement sensitivity
self.position_scale = 0.5     # Position movement sensitivity
self.gripper_threshold = 150.0 # Gripper activation threshold
```

### Adding New Pose Mappings
To add custom pose-to-action mappings:

1. Extend the `pose_to_action()` method
2. Add new landmark calculations
3. Map to additional robot action dimensions

### Environment Configuration
The scripts support any RoboSuite environment. Popular options:
- `Lift`: Pick and place task
- `Stack`: Block stacking
- `NutAssembly`: Precision assembly
- `Door`: Door opening task
- `Wipe`: Surface cleaning

## Performance Tips

### Improving Detection Accuracy
- Ensure good lighting conditions
- Minimize background clutter
- Wear contrasting clothing
- Position camera at appropriate height

### Optimizing Performance
- Reduce model complexity for faster processing
- Lower camera resolution if needed
- Adjust confidence thresholds based on environment
- Enable GPU acceleration if available

## Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# List available cameras
ls /dev/video*

# Try different camera IDs
python realtime_pose_estimation.py --camera_id 1
```

**Poor pose detection:**
- Increase lighting
- Check camera positioning
- Lower confidence threshold
- Ensure full body visibility

**RoboSuite errors:**
- Verify RoboSuite installation
- Check environment/robot name spelling
- Ensure proper dependencies

**Performance issues:**
- Reduce model complexity to 0
- Lower camera resolution
- Close other applications

### Error Messages

**"Could not open camera":**
- Check camera permissions
- Try different camera ID
- Verify camera is not in use by other applications

**"RoboSuite not available":**
- Install RoboSuite: `pip install robosuite`
- Check Python environment

## Technical Details

### MediaPipe Model
- Uses MediaPipe Pose solution
- 33 pose landmarks detected
- Real-time processing capable
- Configurable complexity levels

### Coordinate Systems
- MediaPipe: Normalized coordinates [0,1]
- Robot: Action space [-1,1]
- Camera: Pixel coordinates

### Performance Characteristics
- **Latency**: ~10-50ms depending on hardware
- **FPS**: 15-30 depending on settings
- **Accuracy**: High for visible landmarks
- **Robustness**: Good under normal lighting

## Future Enhancements

Potential improvements and extensions:
- Hand gesture recognition for fine control
- Multi-person pose tracking
- Pose sequence recognition for complex commands
- Integration with other robotic frameworks
- Machine learning for personalized control mappings
- VR/AR visualization overlays

## References

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [RoboSuite Documentation](https://robosuite.ai/)
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## License

These scripts are provided under the same license as the parent RoboSuite project.
