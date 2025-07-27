"""
MediaPipe-based teleoperation device for RBY1 robot.
Extracts human SEW (Shoulder, Elbow, Wrist) coordinates and wrist poses from camera stream.
Works standalone without robosuite framework.
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import traceback
from typing import Optional, Dict, Tuple


class MediaPipeTeleopDevice:
    """
    Device for teleoperation using human pose estimation via MediaPipe.
    Extracts SEW (Shoulder, Elbow, Wrist) coordinates and wrist poses from human pose
    for controlling RBY1 robot arms.
    """
    
    def __init__(self, camera_id=0, debug=False, mirror_actions=False):
        """
        Initialize MediaPipe teleoperation device.
        
        Args:
            camera_id (int): Camera device ID
            debug (bool): Enable debug visualization
            mirror_actions (bool): Mirror actions (right robot arm follows left human arm)
        """
        
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
        
        # Camera setup
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
            
        # Test camera by reading a frame
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            raise RuntimeError(f"Failed to read test frame from camera {camera_id}")
        
        print(f"Camera {camera_id} initialized successfully")
        print(f"Frame shape: {test_frame.shape}")
        print(f"Frame dtype: {test_frame.dtype}")
        print(f"Frame min/max values: {test_frame.min()}/{test_frame.max()}")
        
        # Threading and state management
        self.controller_state_lock = threading.Lock()
        self.controller_state = None
        self._reset_state = 0
        self._quit_state = False
        self.stop_event = threading.Event()
        
        # Configuration
        self.mirror_actions = mirror_actions
        self.debug = debug
        self.engaged = False
        
        # SEW pose tracking - store in body-centric coordinates
        self.human_sew_poses = {
            "left": {"S": None, "E": None, "W": None},
            "right": {"S": None, "E": None, "W": None}
        }
        
        # Wrist pose tracking - store 3x3 rotation matrices
        self.human_wrist_poses = {
            "left": None,
            "right": None
        }
        
        # Hand pose tracking for wrist orientation computation
        self.human_hand_poses = {
            "left": {"landmarks": None, "confidence": 0.0},
            "right": {"landmarks": None, "confidence": 0.0}
        }
        
        self._display_controls()
        self._reset_internal_state()
        
        # Start pose estimation thread
        self.pose_thread = threading.Thread(target=self._pose_estimation_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()
    
    @staticmethod
    def _display_controls():
        """Display control instructions."""
        print("=" * 60)
        print("MediaPipe Teleoperation Device Controls:")
        print("- Raise both arms to shoulder height: Start pose tracking")
        print("- Lower arms: Stop pose tracking")
        print("- 'q' key in camera window: Quit")
        print("- 'r' key in camera window: Reset")
        print("=" * 60)
    
    def _reset_internal_state(self):
        """Reset internal state variables."""
        self.human_sew_poses = {
            "left": {"S": None, "E": None, "W": None},
            "right": {"S": None, "E": None, "W": None}
        }
        self.human_wrist_poses = {
            "left": None,
            "right": None
        }
        self.human_hand_poses = {
            "left": {"landmarks": None, "confidence": 0.0},
            "right": {"landmarks": None, "confidence": 0.0}
        }
        self.engaged = False
    
    def start_control(self):
        """Start the control loop."""
        self._reset_internal_state()
        self._reset_state = 0
        self.engaged = True
    
    def _pose_estimation_loop(self):
        """Main loop for pose estimation running in separate thread."""
        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    time.sleep(0.1)
                    continue
                
                if frame is None:
                    print("Camera returned None frame")
                    time.sleep(0.1)
                    continue
                    
                # Debug frame info occasionally
                if self.debug and hasattr(self, '_debug_frame_count'):
                    self._debug_frame_count += 1
                    if self._debug_frame_count % 100 == 0:  # Every 100 frames
                        print(f"Frame {self._debug_frame_count}: shape={frame.shape}, dtype={frame.dtype}, min/max={frame.min()}/{frame.max()}")
                elif self.debug:
                    self._debug_frame_count = 1
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
                
                # Process landmarks
                self._process_pose_landmarks(pose_results, frame)
                self._process_hand_landmarks(hand_results, pose_results, frame)
                
                # Update controller state
                with self.controller_state_lock:
                    self.controller_state = {
                        'sew_poses': self.human_sew_poses.copy(),
                        'wrist_poses': self.human_wrist_poses.copy(),
                        'engaged': self.engaged
                    }
                
                # Display frame with debugging info
                if self.debug:
                    self._add_debug_info_to_frame(frame)
                
                # Check if frame is still valid before displaying
                if frame is not None and frame.size > 0:
                    cv2.imshow('MediaPipe Pose Estimation', frame)
                else:
                    print("Warning: Invalid frame for display")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._quit_state = True
                    break
                elif key == ord('r'):
                    self._reset_state = 1
                
                # Check if window was closed
                if self._is_window_closed():
                    self._quit_state = True
                    break
                    
            except Exception as e:
                print(f"Error in pose estimation loop: {e}")
                traceback.print_exc()
                # Continue even if there's an error to keep the loop running
                time.sleep(0.1)
    
    def _process_pose_landmarks(self, pose_results, frame):
        """Process MediaPipe pose landmarks using body-centric coordinate system."""
        try:
            if pose_results.pose_landmarks and pose_results.pose_world_landmarks:
                # Get body-centric coordinates
                body_centric_coords = self._get_body_centric_coordinates(pose_results)
                
                if body_centric_coords:
                    # Check visibility using normalized landmarks for engagement logic
                    landmarks = pose_results.pose_landmarks.landmark
                    left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                    left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    
                    right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                    right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    
                    # Check visibility (same threshold as reference code)
                    left_sew_valid = all(lm.visibility > 0.75 for lm in [left_shoulder, left_elbow, left_wrist])
                    right_sew_valid = all(lm.visibility > 0.75 for lm in [right_shoulder, right_elbow, right_wrist])
                    
                    # Update SEW poses only if landmarks are visible
                    if left_sew_valid and 'left' in body_centric_coords:
                        self.human_sew_poses['left'] = body_centric_coords['left']
                    else:
                        self.human_sew_poses['left'] = {"S": None, "E": None, "W": None}
                    
                    if right_sew_valid and 'right' in body_centric_coords:
                        self.human_sew_poses['right'] = body_centric_coords['right']
                    else:
                        self.human_sew_poses['right'] = {"S": None, "E": None, "W": None}
                    
                    # Check engagement based on visibility
                    self._check_engagement()
                    
                    # Draw pose landmarks on frame
                    self.mp_drawing.draw_landmarks(
                        frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                else:
                    # No body-centric coordinates - reset poses
                    self.human_sew_poses['left'] = {"S": None, "E": None, "W": None}
                    self.human_sew_poses['right'] = {"S": None, "E": None, "W": None}
                    self._check_engagement()
            else:
                # No pose landmarks - reset poses
                self.human_sew_poses['left'] = {"S": None, "E": None, "W": None}
                self.human_sew_poses['right'] = {"S": None, "E": None, "W": None}
                self._check_engagement()
        except Exception as e:
            print(f"Error processing pose landmarks: {e}")
            traceback.print_exc()
    
    def _get_body_centric_coordinates(self, pose_results):
        """
        Convert MediaPipe world landmarks to a body-centric coordinate system.
        
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
            world_pos = np.array([landmark.x, landmark.y, landmark.z])
            relative_pos = world_pos - body_origin
            return rotation_matrix.T @ relative_pos
        
        # Extract SEW coordinates in body-centric frame
        sew_coordinates = {}
        
        for side in ['LEFT', 'RIGHT']:
            side_key = side.lower()
            
            shoulder_landmark = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_SHOULDER')]
            elbow_landmark = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_ELBOW')]
            wrist_landmark = landmarks_3d[getattr(self.mp_pose.PoseLandmark, f'{side}_WRIST')]
            
            # Transform to body-centric coordinates
            S = transform_to_body_frame(shoulder_landmark)
            E = transform_to_body_frame(elbow_landmark)
            W = transform_to_body_frame(wrist_landmark)
            
            sew_coordinates[side_key] = {
                'S': S,
                'E': E,
                'W': W
            }
        
        # Add body frame info for debugging
        sew_coordinates['body_frame'] = {
            'origin': body_origin,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis
        }
        
        return sew_coordinates
    
    def _process_hand_landmarks(self, hand_results, pose_results, frame):
        """Process MediaPipe hand landmarks to compute wrist orientation."""
        # Clear stale hand data when no hands detected
        if not hand_results or not hand_results.multi_hand_landmarks or not hand_results.multi_handedness:
            self.human_wrist_poses['left'] = None
            self.human_wrist_poses['right'] = None
            self.human_hand_poses['left'] = {'landmarks': None, 'confidence': 0.0}
            self.human_hand_poses['right'] = {'landmarks': None, 'confidence': 0.0}
            return
        try:
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                # Get body-centric coordinates for reference
                body_centric_coords = self._get_body_centric_coordinates(pose_results)
                if not body_centric_coords:
                    return
                
                hand_centric_coords = self._get_hand_centric_coordinates(hand_results, body_centric_coords)
                
                # Compute wrist rotations from hand landmarks
                if hand_centric_coords:
                    for side in ['left', 'right']:
                        if side in hand_centric_coords:
                            wrist_rotation = self._compute_wrist_rotation_from_hand(side, hand_centric_coords[side]['landmarks'])
                            if wrist_rotation is not None:
                                self.human_wrist_poses[side] = wrist_rotation

                            # Store hand landmarks for debugging
                            self.human_hand_poses[side] = {
                                'landmarks': hand_centric_coords[side]['landmarks'],
                                'confidence': hand_centric_coords[side].get('confidence', 0.0)
                            }
                
                # Draw hand landmarks on frame
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
        except Exception as e:
            print(f"Error processing hand landmarks: {e}")
            traceback.print_exc()
    
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
        
        def world_to_body_frame(world_pos):
            """Convert world position to body-centric coordinates."""
            # Translate to body origin
            translated = world_pos - body_origin
            # Rotate to body frame
            body_pos = body_rotation_matrix.T @ translated
            return body_pos
        
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
            # Z-axis: from wrist towards palm center (average of finger MCPs)
            if middle_mcp is not None and ring_mcp is not None:
                palm_center = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4
            elif middle_mcp is not None:
                palm_center = (index_mcp + middle_mcp + pinky_mcp) / 3
            else:
                palm_center = (index_mcp + pinky_mcp) / 2
                
            z_axis = -(palm_center - wrist)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
            
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
            if hand_side == 'left':
                reference_y = np.cross(wrist_to_pinky, wrist_to_index)
            else:  # right hand
                reference_y = np.cross(wrist_to_index, wrist_to_pinky)
            
            reference_y = reference_y / (np.linalg.norm(reference_y) + 1e-8)
            
            # Choose palm normal direction that aligns with handedness
            if np.dot(palm_normal, reference_y) < 0:
                palm_normal = -palm_normal
            
            y_axis = palm_normal
            
            # X-axis: X = Y × Z (ensuring right-handed coordinate system)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
            
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
                traceback.print_exc()
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
                return 0.0  # Default to open gripper
            
            landmarks = hand_data["landmarks"]
            
            # Get thumb tip and index finger tip positions
            thumb_tip = landmarks.get('thumb_tip')
            index_tip = landmarks.get('index_finger_tip')
            
            if thumb_tip is None or index_tip is None:
                return 0.0  # Default to open gripper
            
            # Calculate distance between thumb and index finger tips
            distance = np.linalg.norm(thumb_tip - index_tip)
            
            # Grasp threshold - close gripper when distance < 0.04 meters (4 cm)
            grasp_threshold = 0.04
            
            if distance < grasp_threshold:
                grasp_value = 1.0  # Close gripper
            else:
                grasp_value = 0.0  # Open gripper
            
            return grasp_value
            
        except Exception as e:
            if self.debug:
                print(f"Error computing grasp for {hand_side} hand: {e}")
                traceback.print_exc()
            return 0.0  # Default to open gripper on error
    
    def _check_engagement(self):
        """Check if user is engaged based on arm positions and visibility."""
        # Check if both arms have valid SEW data
        left_sew = self.human_sew_poses["left"]
        right_sew = self.human_sew_poses["right"]
            
        left_engaged = (left_sew["S"] is not None and 
                       left_sew["E"] is not None and 
                       left_sew["W"] is not None)
        right_engaged = (right_sew["S"] is not None and 
                        right_sew["E"] is not None and 
                        right_sew["W"] is not None)

        # Require both arms to be visible and tracked for engagement
        self.engaged = left_engaged and right_engaged
    
    def _add_debug_info_to_frame(self, frame):
        """Add debugging information to the display frame."""
        y_offset = 30
        
        # Engagement status
        status_color = (0, 255, 0) if self.engaged else (0, 0, 255)
        status_text = "ENGAGED" if self.engaged else "DISENGAGED"
        cv2.putText(frame, f"Control: {status_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        y_offset += 30
        
        # Add SEW info to frame
        self._add_sew_info_to_frame(frame)
        
        # Add hand/wrist info to frame  
        self._add_hand_info_to_frame(frame)
    
    def _add_sew_info_to_frame(self, frame):
        """Add SEW coordinate information to the display frame."""
        y_offset = 90
        cv2.putText(frame, "Body-Centric SEW Coordinates (meters):", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        y_offset += 20
        
        for side in ["left", "right"]:
            sew = self.human_sew_poses[side]
            if all(pose is not None for pose in sew.values()):
                s_str = f"S=[{sew['S'][0]:.3f},{sew['S'][1]:.3f},{sew['S'][2]:.3f}]"
                e_str = f"E=[{sew['E'][0]:.3f},{sew['E'][1]:.3f},{sew['E'][2]:.3f}]"
                w_str = f"W=[{sew['W'][0]:.3f},{sew['W'][1]:.3f},{sew['W'][2]:.3f}]"
                cv2.putText(frame, f"{side.upper()}: {s_str}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 15
                cv2.putText(frame, f"      {e_str}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 15
                cv2.putText(frame, f"      {w_str}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 20
            else:
                cv2.putText(frame, f"{side.upper()}: Not detected", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv2.LINE_AA)
                y_offset += 20
    
    def _add_hand_info_to_frame(self, frame):
        """Add hand/wrist information to frame display."""
        y_offset = 250  # Start below pose info
        
        # Add header in magenta to match reference code style
        cv2.putText(frame, "Wrist Orientations:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        y_offset += 25
        
        hands_detected = any(self.human_wrist_poses[hand] is not None for hand in ['left', 'right'])
        
        if hands_detected:
            for hand_label in ['left', 'right']:
                wrist_pose = self.human_wrist_poses[hand_label]
                hand_data = self.human_hand_poses[hand_label]
                
                if wrist_pose is not None:
                    confidence = hand_data.get('confidence', 0.0)
                    cv2.putText(frame, f"{hand_label.upper()}: Available (conf: {confidence:.2f})", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"{hand_label.upper()}: Not detected", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
                y_offset += 20
        else:
            cv2.putText(frame, "No hands detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
    
    def get_controller_state(self):
        """Get current controller state with SEW poses and wrist orientations."""
        with self.controller_state_lock:
            if self.controller_state is not None:
                return self.controller_state.copy()
            else:
                # Return default state if not initialized
                return {
                    'sew_poses': self.human_sew_poses.copy(),
                    'wrist_poses': self.human_wrist_poses.copy(),
                    'engaged': self.engaged
                }
    
    def get_sew_and_wrist_poses(self):
        """Get current SEW poses and wrist orientations."""
        return {
            'sew_poses': self.human_sew_poses.copy(),
            'wrist_poses': self.human_wrist_poses.copy()
        }
    
    def should_quit(self):
        """Check if quit was requested."""
        return self._quit_state
    
    def should_reset(self):
        """Check if reset was requested."""
        if self._reset_state == 1:
            self._reset_state = 0
            return True
        return False
    
    def _is_window_closed(self):
        """Check if OpenCV window was closed."""
        try:
            # Try to get window property - will throw exception if window is closed
            prop = cv2.getWindowProperty('MediaPipe Pose Estimation', cv2.WND_PROP_VISIBLE)
            return prop < 1
        except cv2.error:
            return True
        except Exception as e:
            if self.debug:
                print(f"Error checking window status: {e}")
            return True
    
    def stop(self):
        """Stop the device and cleanup resources."""
        print("Stopping MediaPipe device...")
        self.stop_event.set()
        if hasattr(self, 'pose_thread') and self.pose_thread.is_alive():
            self.pose_thread.join(timeout=2.0)
        
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("MediaPipe device stopped.")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.stop()
        except:
            pass
