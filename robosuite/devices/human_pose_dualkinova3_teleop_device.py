import threading
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os
from typing import Optional, Dict, Tuple

# Stronger Pose Estimation Model
# wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

import robosuite.utils.transform_utils as T
from robosuite.devices import Device


class HumanPoseDualKinova3Teleop(Device):
    """
    Device for teleoperation using human pose estimation via MediaPipe.
    Extracts SEW (Shoulder, Elbow, Wrist) coordinates from human pose and 
    sends them to the SEW Mimic controller for robot arm control.
    """
    
    def __init__(self, env=None, debug=False, camera_id=0, mirror_actions=True):
        super().__init__(env)
        
        # Check robot models and see if there are multiple arms
        self.robot_interface = env
        self.env_sim = self.env.env.sim
        self.robot_models = []
        self.bimanual = False

        for robot in self.robot_interface.robots:
            self.robot_models.append(robot.robot_model.name)
            if robot.robot_model.arm_type == 'bimanual':
                self.bimanual = True
        print("Robot models:", self.robot_models)

        # Setup MediaPipe pose estimation
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True
        )
        
        # Camera setup
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        # Threading and state management
        self.controller_state_lock = threading.Lock()
        self.controller_state = None
        self._reset_state = 0
        self._quit_state = False  # New: separate quit signal from reset
        self.stop_event = threading.Event()

        # Arm configuration
        self.mirror_actions = mirror_actions
        if self.mirror_actions:
            self._arm2side = {
                "right": "left",  # Right robot arm follows left human arm
                "left": "right",  # Left robot arm follows right human arm
            }
        else:
            self._arm2side = {
                "left": "left",   # Left robot arm follows left human arm
                "right": "right", # Right robot arm follows right human arm
            }

        # SEW pose tracking
        self.human_sew_poses = {
            "left": {"S": None, "E": None, "W": None},
            "right": {"S": None, "E": None, "W": None}
        }
        
        self.debug = debug
        self.engaged = False
        
        # Gripper control (simplified - could be extended with hand gestures)
        self.grasp_states = [[0.0] * len(self.all_robot_arms[i]) for i in range(self.num_robots)]
        
        self._display_controls()
        self._reset_internal_state()
        
        # Start pose estimation thread
        self.pose_thread = threading.Thread(target=self._pose_estimation_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()

    @staticmethod
    def _display_controls():
        """Method to pretty print controls."""
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Raise arms to shoulder height", "Start pose tracking")
        print_command("Lower arms", "Stop pose tracking")
        print_command("'q' key in camera window", "Quit")
        print("")

    def _reset_internal_state(self):
        """Reset internal state variables."""
        super()._reset_internal_state()
        self.grasp_states = [[0.0] * len(self.all_robot_arms[i]) for i in range(self.num_robots)]
        
        # Reset pose tracking
        self.human_sew_poses = {
            "left": {"S": None, "E": None, "W": None},
            "right": {"S": None, "E": None, "W": None}
        }
        self.engaged = False

    def start_control(self):
        """Method that should be called externally before controller can start receiving commands."""
        self._reset_internal_state()
        self._reset_state = 0
        self.engaged = True

    def _pose_estimation_loop(self):
        """Main loop for pose estimation running in separate thread."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                continue
                
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                
                # Perform pose detection
                results = self.pose.process(rgb_frame)
                
                # Convert back to BGR for display
                rgb_frame.flags.writeable = True
                display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Process pose landmarks using body-centric coordinates
                if results.pose_landmarks and results.pose_world_landmarks:
                    self._process_pose_landmarks(results, display_frame)
                    
                    # Draw pose landmarks
                    self.mp_drawing.draw_landmarks(
                        display_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Add SEW coordinate info
                    self._add_sew_info_to_frame(display_frame)
                
                # Add status info
                engaged_text = "ENGAGED" if self.engaged else "DISENGAGED"
                cv2.putText(display_frame, f"Control: {engaged_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.engaged else (0, 0, 255), 2)
                
                # Resize the display frame to 200% (2x) its original size
                height, width = display_frame.shape[:2]
                magnify = 2.0
                new_width = int(width * magnify)
                new_height = int(height * magnify)
                display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Display frame
                cv2.imshow('Human Pose Teleoperation', display_frame)
                
                # Check if window was closed
                if self._is_window_closed():
                    if self.debug:
                        print("OpenCV window closed, quitting...")
                    self._quit_state = True
                    self._reset_state = 1
                    break
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    if self.debug:
                        print("'q' key pressed, quitting...")
                    self._quit_state = True
                    self._reset_state = 1
                    break
                    
            except Exception as e:
                if self.debug:
                    print(f"Error in pose estimation loop: {e}")
                
            # time.sleep(0.01)  # ~100 Hz

    def _process_pose_landmarks(self, pose_results, frame):
        """Process MediaPipe pose landmarks using body-centric coordinate system."""
        try:
            # Get body-centric coordinates
            body_centric_coords = self._get_body_centric_coordinates(pose_results)
            
            # Check visibility using normalized landmarks for engagement logic
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Check visibility
            left_sew_valid = all(lm.visibility > 0.75 for lm in [left_shoulder, left_elbow, left_wrist])
            right_sew_valid = all(lm.visibility > 0.75 for lm in [right_shoulder, right_elbow, right_wrist])
            
            # Use timeout to prevent hanging
            if self.controller_state_lock.acquire(timeout=0.05):
                try:
                    if body_centric_coords and left_sew_valid:
                        # Use body-centric coordinates directly for left arm
                        left_sew = body_centric_coords['left']
                        # Validate coordinates are finite and reasonable
                        if (all(np.isfinite(left_sew['S'])) and all(np.isfinite(left_sew['E'])) and all(np.isfinite(left_sew['W'])) and
                            np.linalg.norm(left_sew['S']) < 10 and np.linalg.norm(left_sew['E']) < 10 and np.linalg.norm(left_sew['W']) < 10):
                            self.human_sew_poses["left"] = {
                                "S": left_sew['S'], 
                                "E": left_sew['E'], 
                                "W": left_sew['W']
                            }
                        else:
                            self.human_sew_poses["left"] = {"S": None, "E": None, "W": None}
                    else:
                        self.human_sew_poses["left"] = {"S": None, "E": None, "W": None}
                    
                    if body_centric_coords and right_sew_valid:
                        # Use body-centric coordinates directly for right arm
                        right_sew = body_centric_coords['right']
                        # Validate coordinates are finite and reasonable
                        if (all(np.isfinite(right_sew['S'])) and all(np.isfinite(right_sew['E'])) and all(np.isfinite(right_sew['W'])) and
                            np.linalg.norm(right_sew['S']) < 10 and np.linalg.norm(right_sew['E']) < 10 and np.linalg.norm(right_sew['W']) < 10):
                            self.human_sew_poses["right"] = {
                                "S": right_sew['S'], 
                                "E": right_sew['E'], 
                                "W": right_sew['W']
                            }
                        else:
                            self.human_sew_poses["right"] = {"S": None, "E": None, "W": None}
                    else:
                        self.human_sew_poses["right"] = {"S": None, "E": None, "W": None}
                    
                    # Engagement logic: engage if both arms are visible and in reasonable pose
                    both_arms_visible = left_sew_valid and right_sew_valid and body_centric_coords
                    
                    self.engaged = both_arms_visible
                finally:
                    self.controller_state_lock.release()
            else:
                # Timeout occurred - skip this frame
                if self.debug:
                    print("Warning: Pose processing lock timeout")
                        
        except Exception as e:
            if self.debug:
                print(f"Error processing pose landmarks: {e}")

    def _get_body_centric_coordinates(self, pose_results):
        """
        Convert MediaPipe world landmarks to a body-centric coordinate system.
        Same implementation as in realtime_pose_estimation.py
        
        Args:
            pose_results: MediaPipe pose detection results
            
        Returns:
            Dictionary containing SEW coordinates in body-centric frame
        """
        if not pose_results.pose_world_landmarks:
            return None
            
        landmarks_3d = pose_results.pose_world_landmarks.landmark
        
        # Get key body landmarks
        left_shoulder = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate body center (midpoint between shoulders and hips)
        shoulder_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2
        ])
        
        hip_center = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2
        ])
        
        # Use shoulder center as origin for upper body tracking
        body_origin = shoulder_center
        
        # Create body-centric coordinate frame
        # Y-axis: right to left (shoulder line)
        y_axis = np.array([
            left_shoulder.x - right_shoulder.x,
            left_shoulder.y - right_shoulder.y,
            left_shoulder.z - right_shoulder.z
        ])
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)  # normalize
        
        # Z-axis: up direction (shoulder to hip, inverted)
        torso_vector = hip_center - shoulder_center
        z_axis = -torso_vector / (np.linalg.norm(torso_vector) + 1e-8)  # up is positive Z

        # X-axis: forward direction (cross product)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        # Create transformation matrix from world to body-centric frame
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        def transform_to_body_frame(landmark):
            """Transform a landmark to body-centric coordinates."""
            world_pos = np.array([landmark.x, landmark.y, landmark.z])
            # Translate to body origin
            translated = world_pos - body_origin
            # Rotate to body frame
            body_pos = rotation_matrix.T @ translated
            return body_pos
        
        # Extract SEW coordinates in body-centric frame
        sew_coordinates = {}
        
        for side in ['LEFT', 'RIGHT']:
            side_key = side.lower()
            
            # Get landmarks
            shoulder = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_SHOULDER')]
            elbow = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_ELBOW')]
            wrist = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_WRIST')]
            
            # Transform to body-centric coordinates
            S_body = transform_to_body_frame(shoulder)
            E_body = transform_to_body_frame(elbow)
            W_body = transform_to_body_frame(wrist)
            
            sew_coordinates[side_key] = {
                'S': S_body,
                'E': E_body, 
                'W': W_body
            }
        
        # Add body frame info
        sew_coordinates['body_frame'] = {
            'origin': body_origin,
            'x_axis': x_axis,  # forward
            'y_axis': y_axis,  # right to left  
            'z_axis': z_axis   # down to up (hip to shoulder)
        }
        
        return sew_coordinates

    def _add_sew_info_to_frame(self, frame):
        """Add SEW coordinate information to the display frame."""
        y_offset = 90
        cv2.putText(frame, "Body-Centric SEW Coordinates (meters):", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        y_offset += 20
        
        for side in ["left", "right"]:
            sew = self.human_sew_poses[side]
            if all(pose is not None for pose in sew.values()):
                cv2.putText(frame, f"{side.upper()} ARM:", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 15
                
                for joint, pos in sew.items():
                    if pos is not None:
                        text = f"  {joint}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
                        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.35, (0, 255, 0), 1, cv2.LINE_AA)
                        y_offset += 12
                y_offset += 5

    def get_controller_state(self):
        """Get current controller state with SEW poses."""
        # Use timeout to prevent hanging
        if self.controller_state_lock.acquire(timeout=0.1):
            try:
                if not self.engaged:
                    return None
                    
                # Create controller state dict
                controller_state = {}
                
                for arm_side in ["left", "right"]:
                    sew = self.human_sew_poses[arm_side]
                    
                    # Check if all SEW poses are available
                    if all(pose is not None for pose in sew.values()):
                        # Package SEW coordinates for the controller
                        sew_action = np.concatenate([sew["S"], sew["E"], sew["W"]])
                        controller_state[f"{arm_side}_sew"] = sew_action
                        controller_state[f"{arm_side}_valid"] = True
                    else:
                        # Send invalid/empty action to hold current pose
                        controller_state[f"{arm_side}_sew"] = np.full(9, np.nan)
                        controller_state[f"{arm_side}_valid"] = False
                    
                    # Simple gripper control (could be enhanced with hand gesture recognition)
                    controller_state[f"{arm_side}_grasp"] = 0.0  # Keep gripper neutral
                    
                    # No reset triggered
                    controller_state[f"{arm_side}_reset"] = False
                
                return controller_state
            finally:
                self.controller_state_lock.release()
        else:
            # Timeout occurred, return None
            if self.debug:
                print("Warning: Controller state lock timeout")
            return None

    def input2action(self, mirror_actions=False):
        """
        Converts pose input into valid action sequence for env.step().
        
        Args:
            mirror_actions (bool): Whether to mirror actions for different viewpoint
            
        Returns:
            Optional[Dict]: Dictionary of actions for env.step() or None if reset
        """
        robot = self.env.robots[self.active_robot]
        
        state = self.get_controller_state()
        
        ac_dict = {}
        
        # Process each robot arm
        for arm in robot.arms:
            human_side = self._arm2side[arm]
            
            if state is not None:
                # We have a valid state (engaged)
                sew_valid = state[f"{human_side}_valid"]
                sew_action = state[f"{human_side}_sew"]
                grasp = state[f"{human_side}_grasp"]
                reset = state[f"{human_side}_reset"]
                
                # If reset is triggered, return None
                if reset:
                    return None
            else:
                # Not engaged - use NaN to signal "hold current pose"
                sew_valid = False
                sew_action = np.full(9, np.nan)
                grasp = 0.0
                reset = False
            
            # Get controller and verify it's SEW_MIMIC
            controller = robot.part_controllers[arm]
            if controller.name != "SEW_MIMIC":
                print(f"Warning: Expected SEW_MIMIC controller for arm {arm}, got {controller.name}")
                # Fall back to hold pose (NaN for SEW controller)
                sew_action = np.full(9, np.nan)
            
            # Create action dict entries
            ac_dict[f"{arm}_sew"] = sew_action
            ac_dict[f"{arm}_gripper"] = np.array([grasp])
            
            # For compatibility with existing action structure
            ac_dict[f"{arm}_abs"] = sew_action  # SEW positions as absolute coordinates
            ac_dict[f"{arm}_delta"] = np.zeros(9)  # No delta for SEW controller

        # Clip actions to safe ranges (but preserve NaN values)
        for (k, v) in ac_dict.items():
            if "abs" not in k and "sew" not in k:
                ac_dict[k] = np.clip(v, -1, 1)
            elif "sew" in k:
                # Only clip finite values, preserve NaN
                if np.any(np.isfinite(v)):
                    ac_dict[k] = np.clip(v, -2.0, 2.0)
                # If all NaN, leave as is (signals "hold current pose")

        return ac_dict

    def stop(self):
        """Stop the pose estimation and cleanup resources."""
        self.stop_event.set()
        if self.pose_thread.is_alive():
            self.pose_thread.join(timeout=1.0)
        
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        print("Human pose teleoperation stopped.")

    def __del__(self):
        """Cleanup on object destruction."""
        self.stop()

    def _is_window_closed(self):
        """Check if the OpenCV window is closed."""
        try:
            # Try to get window property - this will return -1 if window is closed
            return cv2.getWindowProperty('Human Pose Teleoperation', cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            # Exception means window doesn't exist
            return True
        except Exception:
            # Any other exception, assume window is closed
            return True

    def should_quit(self):
        """Check if the user wants to quit completely (not just reset)."""
        return self._quit_state
