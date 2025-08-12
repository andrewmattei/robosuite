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
import Angle_Transfer

# Stronger Pose Estimation Model
# wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

import robosuite.utils.transform_utils as T
from robosuite.devices import Device

class MediaPipeTeleop(Device):
    """
    Device for teleoperation using human pose estimation via MediaPipe.
    Extracts SEW (Shoulder, Elbow, Wrist) coordinates from human pose and 
    sends them to the SEW Mimic controller for robot arm control.

    Author: Chuizheng Kong
    Date created: 2025-07-25
    """
    
    def __init__(self, env=None, cap=None, debug=False, mirror_actions=True):
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
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True
        )
        
        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # # Camera setup
        # self.camera_id = camera_id
        # self.cap = cv2.VideoCapture(camera_id)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # if not self.cap.isOpened():
        #     raise RuntimeError(f"Could not open camera {camera_id}")

        # Use the provided camera capture object
        self.cap = cap
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

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
        
        # Hand pose tracking
        self.human_hand_poses = {
            "left": {"landmarks": None, "confidence": 0.0},
            "right": {"landmarks": None, "confidence": 0.0}
        }

        self.human_head_angles = {
                'theta': None,
                'phi': None
        }

        # Transfering Data to Arduino
        self.serial_port = '/dev/cu.usbmodem1101'
        self.baud = 9600
        self.transfer = None
        
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
        self.human_hand_poses = {
            "left": {"landmarks": None, "confidence": 0.0},
            "right": {"landmarks": None, "confidence": 0.0}
        }

        self.human_head_angles = {
                'theta': None,
                'phi': None
        }

        self.engaged = False

    def start_control(self):
        """Method that should be called externally before controller can start receiving commands."""
        self._reset_internal_state()
        self._reset_state = 0
        self.engaged = True

    def _pose_estimation_loop(self):
        """Main loop for pose estimation running in separate thread."""
        # try:
        #     self.transfer = Angle_Transfer.Angle_Transfer(self.serial_port, baud=9600)

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
                    
                    # Perform hand detection
                    hand_results = self.hands.process(rgb_frame)
                    
                    # Convert back to BGR for display
                    rgb_frame.flags.writeable = True
                    display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    
                    # Process pose landmarks using body-centric coordinates
                    if results.pose_landmarks and results.pose_world_landmarks:
                        # Debug print to confirm landmarks are detected
                        if self.debug:
                            print("Pose landmarks detected!")

                        self._process_pose_landmarks(results, display_frame)
                        angles = self._process_head_landmarks(results, display_frame)

                        # if angles is not None:
                        #     self.transfer.send_face(angles[0], angles[1])
                        
                        # ççççDraw pose landmarks
                        self.mp_drawing.draw_landmarks(
                            display_frame,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                            # Explicitly provide image dimensions for correct projection
                            image_dimensions=(display_frame.shape[1], display_frame.shape[0])
                        )
                        
                        # Add SEW coordinate info
                        self._add_sew_info_to_frame(display_frame)
                        # self._add_head_info_to_frame(display_frame)
                    
                    # Process hand landmarks if detected
                    if hand_results and hand_results.multi_hand_landmarks:
                        # Draw hand landmarks
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                display_frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                        
                        # Process hand landmarks if pose is also available
                        if results.pose_landmarks and results.pose_world_landmarks:
                            self._process_hand_landmarks(hand_results, results, display_frame)
                            
                            # Add hand coordinate info - get current hand poses from state
                            if self.controller_state_lock.acquire(timeout=0.01):
                                try:
                                    current_hand_poses = self.human_hand_poses.copy()
                                    self._add_hand_info_to_frame(display_frame, current_hand_poses)
                                finally:
                                    self.controller_state_lock.release()
                            else:
                                # If can't get lock, just show "processing" message
                                cv2.putText(display_frame, "Hand processing...", (10, 200), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                    
                    # Add status info
                    engaged_text = "ENGAGED" if self.engaged else "DISENGAGED"
                    cv2.putText(display_frame, f"Control: {engaged_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.engaged else (0, 0, 255), 2)
                    
                    # Resize the display frame to 200% (2x) its original size
                    height, width = display_frame.shape[:2]
                    magnify = 1.0
                    new_width = int(width * magnify)
                    new_height = int(height * magnify)
                    display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                    # Display frame with error handling
                    try:
                        cv2.imshow('Human Pose Teleoperation', display_frame)
                        
                        # Handle keyboard input - this is critical for keeping display responsive
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            if self.debug:
                                print("'q' key pressed, quitting...")
                            self._quit_state = True
                            self._reset_state = 1
                            break
                            
                        # Check if window was closed
                        if self._is_window_closed():
                            if self.debug:
                                print("OpenCV window closed, quitting...")
                            self._quit_state = True
                            self._reset_state = 1
                            break
                            
                    except Exception as cv_error:
                        if self.debug:
                            print(f"OpenCV display error: {cv_error}")
                        # Continue to prevent total freeze
                        continue
                        
                except Exception as e:
                    if self.debug:
                        print(f"Error in pose estimation loop: {e}")
                    # Continue even if there's an error to keep display responsive
                    
                # Small delay to prevent overwhelming the display and allow proper frame processing
                time.sleep(0.016)  # ~60 Hz, enough for smooth display without freezing
        # except (IOError, KeyboardInterrupt) as e:
        #     print(f"An error occurred: {e}")
        # finally:
        #     if self.transfer:
        #         self.transfer.close()
            

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

    def _process_hand_landmarks(self, hand_results, pose_results, frame):
        """Process MediaPipe hand landmarks using hand-centric coordinate system."""
        try:
            # Get body-centric coordinates first
            body_centric_coords = self._get_body_centric_coordinates(pose_results)
            if not body_centric_coords:
                return
            
            # Get hand-centric coordinates
            hand_centric_coords = self._get_hand_centric_coordinates(hand_results, body_centric_coords)
            
            # Use timeout to prevent hanging
            if self.controller_state_lock.acquire(timeout=0.05):
                try:
                    if hand_centric_coords:
                        for hand_label in ['left', 'right']:
                            if hand_label in hand_centric_coords:
                                hand_info = hand_centric_coords[hand_label]
                                self.human_hand_poses[hand_label] = {
                                    "landmarks": hand_info['landmarks'],
                                    "confidence": hand_info['confidence']
                                }
                            else:
                                self.human_hand_poses[hand_label] = {
                                    "landmarks": None,
                                    "confidence": 0.0
                                }
                    else:
                        # Reset hand poses if no valid coordinates
                        for hand_label in ['left', 'right']:
                            self.human_hand_poses[hand_label] = {
                                "landmarks": None,
                                "confidence": 0.0
                            }
                finally:
                    self.controller_state_lock.release()
            else:
                if self.debug:
                    print("Warning: Hand processing lock timeout")
                        
        except Exception as e:
            if self.debug:
                print(f"Error processing hand landmarks: {e}")

    def _get_hand_centric_coordinates(self, hand_results, body_centric_coords):
        """
        Align hand world landmarks to pose world landmarks and convert to body-centric coordinates.
        Same implementation as in realtime_pose_estimation.py
        
        Args:
            hand_results: MediaPipe hand detection results
            body_centric_coords: Body-centric coordinate system from _get_body_centric_coordinates()
            
        Returns:
            Dictionary containing aligned hand landmarks in body-centric frame
        """
        if (not hand_results or not hand_results.multi_hand_world_landmarks or 
            not body_centric_coords):
            return None
            
        hand_frames = {}
        
        # Get body frame transformation matrix for converting to body-centric coordinates
        body_frame = body_centric_coords['body_frame']
        body_origin = body_frame['origin']
        body_rotation_matrix = np.column_stack([
            body_frame['x_axis'], 
            body_frame['y_axis'], 
            body_frame['z_axis']
        ])
        
        for hand_idx, (hand_landmarks, hand_world_landmarks, handedness) in enumerate(
            zip(hand_results.multi_hand_landmarks, hand_results.multi_hand_world_landmarks, hand_results.multi_handedness)):
            
            # Flip MediaPipe hand labels to match body pose perspective
            mediapipe_label = handedness.classification[0].label.lower()
            actual_hand_label = 'right' if mediapipe_label == 'left' else 'left'
            
            # Get the corresponding pose wrist for alignment
            if actual_hand_label == 'left':
                pose_wrist_body = body_centric_coords['left']['W']
            else:
                pose_wrist_body = body_centric_coords['right']['W']
            
            # Get hand wrist position
            hand_wrist = hand_world_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Calculate translation offset to align hand wrist with pose wrist
            hand_wrist_world = np.array([hand_wrist.x, hand_wrist.y, hand_wrist.z])
            
            # Align all hand landmarks to pose world frame and convert to body frame
            aligned_hand_landmarks = {}
            landmark_names = [
                'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
                'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
                'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
            ]
            
            for landmark_name in landmark_names:
                landmark_id = getattr(self.mp_hands.HandLandmark, landmark_name)
                hand_landmark = hand_world_landmarks.landmark[landmark_id]
                
                # Get hand landmark in world coordinates
                hand_landmark_world = np.array([
                    hand_landmark.x,
                    hand_landmark.y, 
                    hand_landmark.z
                ])
                
                # Align hand landmark to pose world frame: 
                # Step 1: Translate hand landmark relative to hand wrist
                relative_to_hand_wrist = hand_landmark_world - hand_wrist_world
                
                # Step 2: Convert to body frame and add pose wrist position
                relative_body = body_rotation_matrix.T @ relative_to_hand_wrist
                aligned_body_pos = relative_body + pose_wrist_body
                
                aligned_hand_landmarks[landmark_name.lower()] = aligned_body_pos
            
            # Store aligned hand information
            hand_frames[actual_hand_label] = {
                'landmarks': aligned_hand_landmarks,  # all landmarks aligned and in body-centric coordinates
                'confidence': handedness.classification[0].score
            }
        
        return hand_frames

    def _compute_wrist_rotation_from_hand(self, hand_side, hand_landmarks):
        """
        Compute wrist rotation matrix from hand landmarks.
        
        Args:
            hand_side (str): 'left' or 'right'
            hand_landmarks (dict): Hand landmarks in body-centric coordinates
            
        Returns:
            np.array: 3x3 rotation matrix representing wrist orientation in body frame, or None if computation fails
        """
        try:
            # Get key landmarks for defining wrist coordinate frame
            wrist = hand_landmarks.get('wrist')
            index_mcp = hand_landmarks.get('index_finger_mcp')
            pinky_mcp = hand_landmarks.get('pinky_mcp')
            ring_mcp = hand_landmarks.get('ring_finger_mcp')
            middle_mcp = hand_landmarks.get('middle_finger_mcp')

            if wrist is None or index_mcp is None or pinky_mcp is None:
                return None
                
            # Define hand coordinate frame with wrist as origin
            # X-axis: from wrist towards palm center (average of finger MCPs)
            if middle_mcp is not None and ring_mcp is not None:
                palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4
            elif middle_mcp is not None:
                palm_center = (index_mcp + middle_mcp + pinky_mcp) / 3
            else:
                palm_center = (index_mcp + pinky_mcp) / 2

            x_axis = (palm_center - wrist)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

            # Collect all available MCP points relative to wrist
            mcp_points = []
            if index_mcp is not None:
                mcp_points.append(index_mcp - wrist)
            if middle_mcp is not None:
                mcp_points.append(middle_mcp - wrist)
            if ring_mcp is not None:
                mcp_points.append(ring_mcp - wrist)
            if pinky_mcp is not None:
                mcp_points.append(pinky_mcp - wrist)
            
            if len(mcp_points) < 2:
                return None  # Need at least 2 MCP points for reliable computation
            
            # Use least squares to find Y-axis that's perpendicular to the palm plane
            # Stack MCP vectors into matrix A
            A = np.array(mcp_points)  # shape: (n_points, 3)
            
            # Find the normal to the plane containing MCP points using SVD
            # The normal is the last column of V (smallest singular value)
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            palm_normal = Vt[-1, :]  # Last row of V^T = last column of V
            palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
            
            # Determine handedness-consistent Y-axis direction
            # Use the palm normal, but ensure correct orientation based on hand side
            wrist_to_index = index_mcp - wrist
            wrist_to_pinky = pinky_mcp - wrist
            
            # Cross product gives a reference direction
            reference_y = np.cross(wrist_to_index, wrist_to_pinky)
            reference_y = reference_y / (np.linalg.norm(reference_y) + 1e-8)
            
            # Choose palm normal direction that aligns with handedness
            if np.dot(palm_normal, reference_y) < 0:
                palm_normal = -palm_normal
            
            y_axis = palm_normal

            # Z-axis: Z = X × Y (ensuring right-handed coordinate system)
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

            # Create rotation matrix
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            # Validate that it's a proper rotation matrix
            if np.abs(np.linalg.det(rotation_matrix) - 1.0) > 0.1:
                if self.debug:
                    print(f"Warning: Invalid rotation matrix computed for {hand_side} hand")
                return None
                
            return rotation_matrix
            
        except Exception as e:
            if self.debug:
                print(f"Error computing wrist rotation for {hand_side} hand: {e}")
            return None

    def _compute_finger_positions_wrist_frame(self, hand_side, hand_data, sew_poses):
        """
        Compute finger joint positions in wrist frame coordinates.
        
        Args:
            hand_side (str): 'left' or 'right'
            hand_data (dict): Hand data containing landmarks and confidence
            sew_poses (dict): SEW poses with 'S', 'E', 'W' positions in body frame
            
        Returns:
            dict: Dictionary with finger joint positions in wrist frame, or None if invalid
        """
        try:
            # Check if hand data is valid
            if (hand_data["landmarks"] is None or 
                hand_data["confidence"] < 0.5 or
                sew_poses["W"] is None):
                return None
            
            landmarks = hand_data["landmarks"]
            wrist_pos_body = sew_poses["W"]  # Wrist position in body frame
            
            # Get wrist rotation matrix to transform from body frame to wrist frame
            wrist_rotation_matrix = self._compute_wrist_rotation_from_hand(hand_side, landmarks)
            
            if wrist_rotation_matrix is None:
                # If we can't compute wrist rotation, fall back to translation only
                if self.debug:
                    print(f"Warning: Could not compute wrist rotation for {hand_side} hand, using translation only")
                wrist_rotation_matrix = np.eye(3)  # Identity matrix
            
            # Inverse of wrist rotation matrix to transform from body frame to wrist frame
            body_to_wrist_rotation = wrist_rotation_matrix.T
            
            # Define the finger joints we want to track (matching Psyonic hand structure)
            finger_joints = {
                'thumb': ['thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip'],
                'index': ['index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip'],
                'middle': ['middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip'],
                'ring': ['ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip'],
                'pinky': ['pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
            }
            
            finger_positions_wrist_frame = {}
            
            for finger_name, joint_names in finger_joints.items():
                finger_positions_wrist_frame[finger_name] = {}
                
                for joint_name in joint_names:
                    # Get joint position in body frame
                    joint_pos_body = landmarks.get(joint_name)
                    
                    if joint_pos_body is not None:
                        # Step 1: Translate from body frame to wrist origin
                        joint_pos_translated = joint_pos_body - wrist_pos_body
                        
                        # Step 2: Rotate from body frame orientation to wrist frame orientation
                        joint_pos_wrist = body_to_wrist_rotation @ joint_pos_translated
                        
                        finger_positions_wrist_frame[finger_name][joint_name] = joint_pos_wrist
                    else:
                        # Joint not detected, set to None
                        finger_positions_wrist_frame[finger_name][joint_name] = None
            
            return finger_positions_wrist_frame
            
        except Exception as e:
            if self.debug:
                print(f"Error computing finger positions for {hand_side} hand: {e}")
            return None

    def _compute_grasp_from_hand(self, hand_side, hand_data):
        """
        Compute gripper control value based on hand gesture (thumb-index finger distance).
        
        Args:
            hand_side (str): 'left' or 'right'
            hand_data (dict): Hand data containing landmarks and confidence
            
        Returns:
            float: Gripper value (0.0 = open, 1.0 = closed)
        """
        try:
            # Check if hand data is valid
            if (hand_data["landmarks"] is None or 
                hand_data["confidence"] < 0.5):
                if self.debug:
                    print(f"{hand_side} hand: No valid hand data - keeping gripper open")
                return 0.0  # Default to open gripper if no valid hand data
            
            landmarks = hand_data["landmarks"]
            
            # Get thumb tip and index finger tip positions
            thumb_tip = landmarks.get('thumb_tip')
            index_tip = landmarks.get('index_finger_tip')
            middle_tip = landmarks.get('middle_finger_tip')
            ring_tip = landmarks.get('ring_finger_tip')
            pinky_tip = landmarks.get('pinky_finger_tip')

            if thumb_tip is None or index_tip is None:
                if self.debug:
                    print(f"{hand_side} hand: Missing thumb or index finger tip - keeping gripper open")
                return 0.0  # Default to open gripper if tips are missing
            # Calculate distance between thumb and index finger tips
            index_distance = np.linalg.norm(thumb_tip - index_tip) if index_tip is not None else 1.0
            middle_distance = np.linalg.norm(thumb_tip - middle_tip) if middle_tip is not None else 1.0
            ring_distance = np.linalg.norm(thumb_tip - ring_tip) if ring_tip is not None else 1.0
            pinky_distance = np.linalg.norm(thumb_tip - pinky_tip) if pinky_tip is not None else 1.0
            distance = np.mean([index_distance, middle_distance, ring_distance, pinky_distance])
            # print(f"{hand_side} hand: Finger distance = {distance:.4f}m")

            # Grasp threshold - close gripper when distance < 0.04 meters (4 cm)
            grasp_threshold = 0.31
            
            if distance < grasp_threshold:
                grasp_value = 1.0  # Close gripper
                if self.debug:
                    print(f"{hand_side} hand: Thumb-Index distance {distance:.4f}m < {grasp_threshold}m - GRASPING")
            else:
                grasp_value = 0.0  # Open gripper
                if self.debug:
                    print(f"{hand_side} hand: Thumb-Index distance {distance:.4f}m >= {grasp_threshold}m - OPEN")
            
            return grasp_value
            
        except Exception as e:
            if self.debug:
                print(f"Error computing grasp for {hand_side} hand: {e}")
            return 0.0  # Default to open gripper on error
        


    def _process_head_landmarks(self, pose_results, frame):
        """Process MediaPipe head landmarks using head-centric coordinate system."""
        try:
            # Get body-centric coordinates first
            body_centric_coords = self._get_body_centric_coordinates(pose_results)
            if not body_centric_coords:
                return
            
            # Use timeout to prevent hanging
            if self.controller_state_lock.acquire(timeout=0.05):
                rx, ry, rz = self._compute_head_rotation(pose_results, body_centric_coords)
                self.controller_state_lock.release()
                return np.array(ry, rz)
            else:
                if self.debug:
                    print("Warning: Head processing lock timeout")
                    return None
                        
        except Exception as e:
            if self.debug:
                print(f"Error processing hand landmarks: {e}")

    
    def _compute_head_rotation(self, head_results, body_centric_coords):
        """
        Compute head rotation as Euler angles (roll, pitch, yaw) from pose landmarks.

        Args:
            head_results: MediaPipe pose detection results.
            body_centric_coords: Body-centric coordinate system.

        Returns:
            Tuple[float, float, float]: Roll, pitch, and yaw angles in radians, or (None, None, None).
        """
        if not head_results or not head_results.pose_world_landmarks or not body_centric_coords:
            return None, None, None

        landmarks_3d = head_results.pose_world_landmarks.landmark

        # Get landmarks for defining head coordinate system
        left_eye = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_EYE]
        left_ear = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_EAR]
        mouth_left = landmarks_3d[self.mp_pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = landmarks_3d[self.mp_pose.PoseLandmark.MOUTH_RIGHT]

        # Transform landmarks to body-centric frame
        body_frame = body_centric_coords['body_frame']
        body_origin = body_frame['origin']
        body_rotation_matrix = np.column_stack([
            body_frame['x_axis'], 
            body_frame['y_axis'], 
            body_frame['z_axis']
            ])

        def transform_to_body_frame(landmark):
            world_pos = np.array([landmark.x, landmark.y, landmark.z])
            translated = world_pos - body_origin
            return body_rotation_matrix.T @ translated

        left_eye_body = transform_to_body_frame(left_eye)
        right_eye_body = transform_to_body_frame(right_eye)
        left_ear_body = transform_to_body_frame(left_ear)
        right_ear_body = transform_to_body_frame(right_ear)
        mouth_left_body = transform_to_body_frame(mouth_left)
        mouth_right_body = transform_to_body_frame(mouth_right)

        # Define head axes
        # Y-axis (stable): from right ear to left ear
        head_y_axis = (left_ear_body - right_ear_body)
        head_y_axis /= (np.linalg.norm(head_y_axis) + 1e-8)

        # Z-axis: from mouth center to eye center
        eye_center = (left_eye_body + right_eye_body) / 2
        mouth_center = (mouth_left_body + mouth_right_body) / 2
        head_z_axis = (eye_center - mouth_center)
        head_z_axis /= (np.linalg.norm(head_z_axis) + 1e-8)

        # X-axis: forward direction (cross product)
        head_x_axis = np.cross(head_y_axis, head_z_axis)
        head_x_axis /= (np.linalg.norm(head_x_axis) + 1e-8)

        # Create rotation matrix
        pre_R_BH = np.column_stack([head_x_axis, head_y_axis, head_z_axis])
        
        # Use SVD to get the closest orthogonal matrix (a proper rotation matrix)
        U_R, _, V_R_T = np.linalg.svd(pre_R_BH)
        R_BH = U_R @ V_R_T

        # Convert rotation matrix to Euler angles
        roll, pitch, yaw = self.rotation_matrix_to_euler(R_BH)

        roll = int(roll)
        yaw = int(yaw)
        pitch = int(pitch)


        print(f"rotation about y: {pitch}, rotation about z: {yaw}")
        return roll, pitch, yaw
    

    
    def rotation_matrix_to_euler(self, rotation_matrix):
        """
        Convert a rotation matrix to (ZYX) Euler angles.

        Args:
        - rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

        Returns:
        - roll (float): Rotation around the x-axis (in radians).
        - pitch (float): Rotation around the y-axis (in radians).
        - yaw (float): Rotation around the z-axis (in radians).
        """
        # Extract the Euler angles from the rotation matrix
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi
        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2)) * 180 / np.pi
        # pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)) * 180 / np.pi
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi

        return roll, pitch, yaw

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

    def _add_hand_info_to_frame(self, frame, hand_poses):
        """Add hand landmark information to frame display."""
        y_offset = 250  # Start below pose info
        
        # Add header in magenta to match realtime_pose_estimation.py
        cv2.putText(frame, "Hand-Centric Coordinates:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        y_offset += 25
        
        if hand_poses and any(hand_poses[hand]['landmarks'] is not None for hand in ['left', 'right']):
            for hand_label in ['left', 'right']:
                hand_data = hand_poses.get(hand_label, {})
                landmarks = hand_data.get('landmarks')
                confidence = hand_data.get('confidence', 0.0)
                
                if landmarks is not None and confidence > 0:
                    # Hand label and confidence in cyan to match realtime_pose_estimation.py
                    hand_text = f"{hand_label.upper()} HAND (conf: {confidence:.2f}):"
                    frame = cv2.putText(frame, hand_text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    y_offset += 20
                    
                    # Key landmarks header in light blue
                    cv2.putText(frame, "  Hand-frame landmarks:", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1, cv2.LINE_AA)
                    y_offset += 15
                    
                    # Key landmarks in body-centric coordinates using magenta
                    if isinstance(landmarks, dict):
                        # Show thumb, index, middle tips in wrist frame
                        sew = self.human_sew_poses[hand_label]
                        finger_positions = self._compute_finger_positions_wrist_frame(hand_label, hand_data, sew)
                        if finger_positions:
                            tip_keys = [
                                ('thumb', 'thumb_tip'),
                                ('index', 'index_finger_tip'),
                                ('middle', 'middle_finger_tip')
                            ]
                            for finger, tip_key in tip_keys:
                                tip_pos = finger_positions.get(finger, {}).get(tip_key)
                                if tip_pos is not None:
                                    display_name = f"{finger.title()} Tip (wrist frame)"
                                    coord_text = f"    {display_name}: ({tip_pos[0]:+.3f}, {tip_pos[1]:+.3f}, {tip_pos[2]:+.3f})"
                                    frame = cv2.putText(frame, coord_text, (10, y_offset), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
                                    y_offset += 12
                        # Add grasp state information (still use body-centric for gesture)
                        thumb_tip = landmarks.get('thumb_tip')
                        index_tip = landmarks.get('index_finger_tip')
                        if thumb_tip is not None and index_tip is not None:
                            distance = np.linalg.norm(thumb_tip - index_tip)
                            grasp_threshold = 0.01
                            grasp_state = "GRASPING" if distance < grasp_threshold else "OPEN"
                            grasp_color = (0, 255, 0) if distance < grasp_threshold else (0, 165, 255)  # Green for grasp, orange for open
                            grasp_text = f"    Thumb-Index Dist: {distance:.4f}m - {grasp_state}"
                            frame = cv2.putText(frame, grasp_text, (10, y_offset), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, grasp_color, 1, cv2.LINE_AA)
                            y_offset += 12
                    y_offset += 10  # Extra space between hands
                else:
                    # Show hand not detected in gray
                    hand_text = f"{hand_label.upper()} HAND: Not detected"
                    frame = cv2.putText(frame, hand_text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
                    y_offset += 25
        else:
            frame = cv2.putText(frame, "No hands detected", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
        
        return frame
    


    # def _add_head_info_to_frame(self, frame):
    #     """Add SEW coordinate information to the display frame."""
    #     y_offset = 90
    #     cv2.putText(frame, "Body-Centric SEW Coordinates (meters):", (10, y_offset),
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    #     y_offset += 20
        
    #     for side in ["left", "right"]:
    #         sew = self.human_sew_poses[side]
    #         if all(pose is not None for pose in sew.values()):
    #             cv2.putText(frame, f"{side.upper()} ARM:", (10, y_offset),
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    #             y_offset += 15
                
    #             for joint, pos in sew.items():
    #                 if pos is not None:
    #                     text = f"  {joint}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
    #                     cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
    #                                0.35, (0, 255, 0), 1, cv2.LINE_AA)
    #                     y_offset += 12
    #             y_offset += 5


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
                    hand_data = self.human_hand_poses[arm_side]
                    
                    # Check if all SEW poses are available
                    if all(pose is not None for pose in sew.values()):
                        # Start with basic SEW coordinates
                        sew_action = np.concatenate([sew["S"], sew["E"], sew["W"]])
                        
                        # Always extend to 18 elements for consistent controller interface
                        # Check if hand pose data is available to compute wrist rotation matrix
                        if (hand_data["landmarks"] is not None and 
                            hand_data["confidence"] > 0.5):  # Minimum confidence threshold
                            
                            # Compute wrist rotation matrix from hand landmarks
                            wrist_rotation_matrix = self._compute_wrist_rotation_from_hand(arm_side, hand_data["landmarks"])
                            
                            if wrist_rotation_matrix is not None:
                                # Use computed rotation matrix (flatten row-wise)
                                rotation_flat = wrist_rotation_matrix.flatten()
                            else:
                                # Hand detection failed - use identity matrix
                                rotation_flat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
                        else:
                            # No hand pose or low confidence - use identity matrix  
                            rotation_flat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
                        
                        # Always create 18-element action: SEW (9) + rotation matrix (9)
                        sew_action = np.concatenate([sew_action, rotation_flat])
                        
                        controller_state[f"{arm_side}_sew"] = sew_action
                        controller_state[f"{arm_side}_valid"] = True
                    else:
                        # Send invalid/empty action to hold current pose - use 18 elements with NaN
                        controller_state[f"{arm_side}_sew"] = np.full(18, np.nan)
                        controller_state[f"{arm_side}_valid"] = False
                    
                    # Hand gesture-based gripper control
                    grasp_value = self._compute_grasp_from_hand(arm_side, hand_data)
                    # print(f"{arm_side.capitalize()} hand grasp value: {grasp_value}")
                    controller_state[f"{arm_side}_grasp"] = grasp_value
                    
                    # Add finger joint positions in wrist frame
                    finger_positions = self._compute_finger_positions_wrist_frame(arm_side, hand_data, sew)
                    controller_state[f"{arm_side}_fingers"] = finger_positions
                    
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
                # Not engaged - use NaN to signal "hold current pose" with 18 elements
                sew_valid = False
                sew_action = np.full(18, np.nan)
                grasp = 0.0
                reset = False
            
            # Get controller and verify it's SEW_MIMIC
            controller = robot.part_controllers[arm]
            if controller.name != "SEW_MIMIC":
                print(f"Warning: Expected SEW_MIMIC controller for arm {arm}, got {controller.name}")
                # Fall back to hold pose (NaN for SEW controller) with 18 elements
                sew_action = np.full(18, np.nan)
            
            # Create action dict entries
            ac_dict[f"{arm}_sew"] = sew_action
            ac_dict[f"{arm}_gripper"] = np.array([grasp * 1.6 - 0.8])
            
            # For compatibility with existing action structure
            ac_dict[f"{arm}_abs"] = sew_action  # SEW positions (and rotation) as absolute coordinates
            # Delta should be 18 elements to match SEW action size
            ac_dict[f"{arm}_delta"] = np.zeros(18)

        # Clip actions to safe ranges (but preserve NaN values)
        for (k, v) in ac_dict.items():
            if "abs" not in k and "sew" not in k:
                ac_dict[k] = np.clip(v, -1, 1)
            elif "sew" in k or "abs" in k:
                # All SEW actions are now 18-element (SEW + rotation matrix)
                if np.any(np.isfinite(v)):
                    # SEW positions + rotation matrix - clip appropriately
                    sew_part = np.clip(v[:9], -2.0, 2.0)  # Position bounds
                    rot_part = np.clip(v[9:], -1.0, 1.0)  # Rotation matrix element bounds
                    ac_dict[k] = np.concatenate([sew_part, rot_part])
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


if __name__ == "__main__":
    """
    Standalone realtime demonstration of MediaPipe teleoperation device.
    This allows testing the pose estimation and hand tracking functionality
    without requiring a full robosuite environment.
    """
    import argparse
    
    def print_pose_data(device):
        """Print current pose and hand data in a formatted way."""
        with device.controller_state_lock:
            # Print SEW poses
            print("\n" + "="*60)
            print("HUMAN POSE DATA (Body-Centric Coordinates)")
            print("="*60)
            
            for side in ["left", "right"]:
                sew = device.human_sew_poses[side]
                hand_data = device.human_hand_poses[side]
                
                print(f"\n{side.upper()} ARM:")
                print("-" * 20)
                
                if all(pose is not None for pose in sew.values()):
                    for joint, pos in sew.items():
                        print(f"  {joint}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
                    
                    # Hand information
                    if hand_data["landmarks"] is not None and hand_data["confidence"] > 0.5:
                        print(f"  HAND (confidence: {hand_data['confidence']:.2f}):")
                        landmarks = hand_data["landmarks"]
                        
                        # Key landmarks
                        key_landmarks = ['wrist', 'thumb_tip', 'index_finger_tip', 'middle_finger_tip']
                        for landmark_name in key_landmarks:
                            if landmark_name in landmarks:
                                coord = landmarks[landmark_name]
                                print(f"    {landmark_name}: ({coord[0]:+.3f}, {coord[1]:+.3f}, {coord[2]:+.3f})")
                        
                        # Grasp state
                        thumb_tip = landmarks.get('thumb_tip')
                        index_tip = landmarks.get('index_finger_tip')
                        if thumb_tip is not None and index_tip is not None:
                            distance = np.linalg.norm(thumb_tip - index_tip)
                            grasp_state = "GRASPING" if distance < 0.31 else "OPEN"
                            print(f"    Grasp: {distance:.4f}m - {grasp_state}")
                        
                        # Wrist rotation matrix
                        wrist_rotation = device._compute_wrist_rotation_from_hand(side, landmarks)
                        if wrist_rotation is not None:
                            print(f"    Wrist Rotation Matrix:")
                            for i in range(3):
                                print(f"      [{wrist_rotation[i,0]:+.3f}, {wrist_rotation[i,1]:+.3f}, {wrist_rotation[i,2]:+.3f}]")
                        
                        # Finger positions in wrist frame
                        finger_positions = device._compute_finger_positions_wrist_frame(side, hand_data, sew)
                        if finger_positions:
                            print(f"    Finger tips in wrist frame:")
                            for finger, joints in finger_positions.items():
                                tip_key = f"{finger}_tip" if finger != 'thumb' else 'thumb_tip'
                                if tip_key in joints and joints[tip_key] is not None:
                                    tip_pos = joints[tip_key]
                                    print(f"      {finger}: ({tip_pos[0]:+.3f}, {tip_pos[1]:+.3f}, {tip_pos[2]:+.3f})")
                        head_position = device._
                    else:
                        print("  HAND: Not detected or low confidence")
                else:
                    print("  Not detected or occluded")
            
            print(f"\nStatus: {'ENGAGED' if device.engaged else 'DISENGAGED'}")
            print("="*60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MediaPipe Teleoperation Device Standalone Test')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--print-interval', type=float, default=2.0, help='Print data interval in seconds (default: 2.0)')
    parser.add_argument('--no-print', action='store_true', help='Disable periodic data printing')
    
    args = parser.parse_args()
    
    print("MediaPipe Teleoperation Device - Standalone Mode")
    print("=" * 50)
    print("Controls:")
    print("  - Raise both arms to shoulder height to start tracking")
    print("  - Lower arms to stop tracking")
    print("  - Press 'q' in camera window to quit")
    print("  - Close camera window to quit")
    print("")
    
    # Create a mock environment class for testing
    class MockRobot:
        def __init__(self):
            self.robot_model = MockRobotModel()
            self.arms = ['left', 'right']
            self.part_controllers = {
                'left': MockController(),
                'right': MockController()
            }
    
    class MockRobotModel:
        def __init__(self):
            self.name = "mock_robot"
            self.arm_type = 'bimanual'
    
    class MockController:
        def __init__(self):
            self.name = "SEW_MIMIC"
    
    class MockSim:
        pass
    
    class MockEnvironment:
        def __init__(self):
            self.robots = [MockRobot()]
            self.env = MockInnerEnv()
    
    class MockInnerEnv:
        def __init__(self):
            self.sim = MockSim()
    
    try:
        # Open the camera first to avoid initialization conflicts
        print(f"Attempting to open camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera}")
        print(f"Camera {args.camera} opened successfully!")

        # Initialize the device with the opened camera
        mock_env = MockEnvironment()
        device = MediaPipeTeleop(
            env=mock_env,
            cap=cap,
            debug=args.debug,
            mirror_actions=False
        )
        print("Starting pose estimation...")
        
        # Start the device
        device.start_control()
        
        last_print_time = time.time()
        
        # Main loop
        while True:
            # Check if user wants to quit
            if device.should_quit():
                print("\nQuitting...")
                break
            
            # Periodic data printing
            if not args.no_print and time.time() - last_print_time >= args.print_interval:
                try:
                    print_pose_data(device)
                    last_print_time = time.time()
                except Exception as e:
                    if args.debug:
                        print(f"Error printing pose data: {e}")
            
            # Small delay to prevent high CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        # Clean up
        try:
            device.stop()
        except:
            pass
        print("Cleanup complete.")