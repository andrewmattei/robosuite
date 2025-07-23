"""
Real-time pose and hand estimation using MediaPipe.

This script captures video from a webcam and performs real-time human pose estimation
and optional hand detection using Google's MediaPipe library. It displays the pose 
landmarks and connections on the video feed, with optional hand landmark detection.

Usage:
    python realtime_pose_estimation.py [--camera_id CAMERA_ID] [--confidence CONFIDENCE]

Args:
    --camera_id: Camera device ID (default: 0)
    --confidence: Minimum detection confidence (default: 0.5)
    --complexity: Model complexity (0, 1, or 2) (default: 1)
    --smooth: Enable landmark smoothing (default: True)
    --enable_hands: Enable hand detection (default: False)
    --save_output: Save the output video to file (default: False)
    --output_path: Path to save output video (default: pose_estimation_output.mp4)

Controls:
    - Press 'q' to quit
    - Press 's' to save a screenshot
    - Press 'r' to reset pose tracking
    - Press 'w' to toggle world coordinates display
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os
from typing import Optional, Tuple, List


class PoseAndHandEstimator:
    """Real-time pose and hand estimation using MediaPipe."""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_hand_detection: bool = True):
        """
        Initialize the pose and hand estimator.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks
            enable_hand_detection: Whether to enable hand detection
        """
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks
            )
            
            # Initialize hand detection if enabled
            self.enable_hand_detection = enable_hand_detection
            if enable_hand_detection:
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            else:
                self.hands = None
            
            # Performance tracking
            self.fps_counter = 0
            self.start_time = time.time()
            self.frame_times = []
            
            hand_status = "with hand detection" if enable_hand_detection else "without hand detection"
            print(f"MediaPipe Pose initialized successfully {hand_status}")
            
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            print("This might be due to protobuf version compatibility issues.")
            print("Try running: pip install protobuf==3.20.3")
            raise
        
    def process_frame(self, frame: np.ndarray, show_world_coords: bool = True) -> Tuple[np.ndarray, Optional[object], Optional[object], Optional[object]]:
        """
        Process a single frame and detect poses and hands.
        
        Args:
            frame: Input BGR frame
            show_world_coords: Whether to display world coordinates
            
        Returns:
            Tuple of (annotated_frame, pose_landmarks, pose_results, hand_results)
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Perform pose detection
            pose_results = self.pose.process(rgb_frame)
            
            # Perform hand detection if enabled
            hand_results = None
            if self.enable_hand_detection and self.hands:
                hand_results = self.hands.process(rgb_frame)
            
            # Convert back to BGR for OpenCV
            rgb_frame.flags.writeable = True
            annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Get body-centric coordinates and display them if enabled
                if show_world_coords:
                    body_centric_coords = self.get_body_centric_coordinates(pose_results)
                    if body_centric_coords:
                        self.display_body_centric_info(annotated_frame, body_centric_coords)
            
            # Draw hand landmarks if enabled and detected
            if self.enable_hand_detection and hand_results and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Get hand-centric coordinates and display them if world coords are enabled
                if show_world_coords and pose_results.pose_world_landmarks:
                    body_centric_coords = self.get_body_centric_coordinates(pose_results)
                    if body_centric_coords:
                        hand_centric_coords = self.get_hand_centric_coordinates(hand_results, body_centric_coords)
                        if hand_centric_coords:
                            self.display_hand_centric_info(annotated_frame, hand_centric_coords)
            
            return annotated_frame, pose_results.pose_landmarks, pose_results, hand_results
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Return the original frame if processing fails
            return frame, None, None, None
    
    def update_fps(self) -> float:
        """Update and return current FPS."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 30 frames for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        else:
            fps = 0
            
        return fps
    
    def cleanup(self):
        """Cleanup resources."""
        self.pose.close()
        if self.enable_hand_detection and self.hands:
            self.hands.close()
    
    def get_body_centric_coordinates(self, pose_results):
        """
        Convert MediaPipe world landmarks to a body-centric coordinate system.
        MediaPipe world has origin at the center of hip, x-axis pointing left, y-axis pointing down, z-axis pointing backward, in facing camera perspective.
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

    def get_hand_centric_coordinates(self, hand_results, body_centric_coords):
        """
        Convert MediaPipe hand landmarks to hand-centric coordinate systems.
        Note that hand has the same world coordinate system as body pose.
        Except origin is at wrist, axes are the same as pose world landmarks.
        Args:
            hand_results: MediaPipe hand detection results
            body_centric_coords: Body-centric coordinate system from get_body_centric_coordinates()
            
        Returns:
            Dictionary containing hand-centric coordinate frames and landmark positions
        """
        if not hand_results or not hand_results.multi_hand_world_landmarks or not body_centric_coords:
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
        
        def wrist_world_to_body_frame(landmark, hand_wrist, pose_wrist_body):
            """Convert world landmark to body-centric coordinates."""
            wrist_world_pos = np.array([landmark.x, landmark.y, landmark.z])
            translated = wrist_world_pos - hand_wrist
            body_pos = body_rotation_matrix.T @ translated + pose_wrist_body
            return body_pos
        
        for hand_idx, (hand_landmarks, hand_world_landmarks, handedness) in enumerate(
            zip(hand_results.multi_hand_landmarks, hand_results.multi_hand_world_landmarks, hand_results.multi_handedness)):
            
            # Flip MediaPipe hand labels to match body pose perspective
            mediapipe_label = handedness.classification[0].label.lower()
            actual_hand_label = 'right' if mediapipe_label == 'left' else 'left'
            
            # Get key landmarks in world coordinates (use world landmarks for 3D coordinates)
            wrist = hand_world_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            index_mcp = hand_world_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_world_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
            
            # Convert to body-centric coordinates
            hand_wrist = np.array([wrist.x, wrist.y, wrist.z])
            pose_wrist_body = body_centric_coords[actual_hand_label]['W']
            wrist_body = wrist_world_to_body_frame(wrist, hand_wrist, pose_wrist_body)
            index_mcp_body = wrist_world_to_body_frame(index_mcp, hand_wrist, pose_wrist_body)
            pinky_mcp_body = wrist_world_to_body_frame(pinky_mcp, hand_wrist, pose_wrist_body)

            # Define hand frame origin at wrist
            hand_origin = wrist_body
            
            # Z-axis points toward midpoint between INDEX_FINGER_MCP and PINKY_MCP
            mcp_midpoint = (index_mcp_body + pinky_mcp_body) / 2
            z_axis = mcp_midpoint - hand_origin
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)  # normalize
            
            # Define Y-axis based on hand laterality
            wrist_to_index = index_mcp_body - hand_origin
            wrist_to_pinky = pinky_mcp_body - hand_origin
            
            if actual_hand_label == 'left':
                # Left hand: y = (WRIST->PINKY) × (WRIST->INDEX)
                y_axis = np.cross(wrist_to_pinky, wrist_to_index)
            else:  # right hand
                # Right hand: y = (WRIST->INDEX) × (WRIST->PINKY)
                y_axis = np.cross(wrist_to_index, wrist_to_pinky)
            
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)  # normalize
            
            # X-axis: x = y × z
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)  # normalize
            
            # Create homogeneous transformation matrix (4x4)
            # This represents the hand frame in body-centric coordinates
            hand_transform = np.eye(4)
            hand_transform[:3, 0] = x_axis  # X column
            hand_transform[:3, 1] = y_axis  # Y column
            hand_transform[:3, 2] = z_axis  # Z column
            hand_transform[:3, 3] = hand_origin  # Translation

            if actual_hand_label == 'left':
                print(hand_transform)
            
            # Create rotation matrix for transforming points to hand frame
            hand_rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            def body_to_hand_frame(body_pos):
                """Transform a body-centric position to hand-centric coordinates."""
                translated = body_pos - hand_origin
                hand_pos = hand_rotation_matrix.T @ translated
                return hand_pos
            
            # Transform all hand landmarks to hand-centric coordinates
            hand_landmarks_in_hand_frame = {}
            landmark_names = [
                'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
                'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
                'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
            ]
            
            for landmark_name in landmark_names:
                landmark_id = getattr(self.mp_hands.HandLandmark, landmark_name)
                landmark = hand_world_landmarks.landmark[landmark_id]  # Use world landmarks
                landmark_body = wrist_world_to_body_frame(landmark, hand_wrist, pose_wrist_body)
                landmark_hand = body_to_hand_frame(landmark_body)
                hand_landmarks_in_hand_frame[landmark_name.lower()] = landmark_hand
            
            # Store hand frame information
            hand_frames[actual_hand_label] = {
                'transform_matrix': hand_transform,  # 4x4 homogeneous matrix in body frame
                'origin': hand_origin,  # hand origin in body-centric coordinates
                'x_axis': x_axis,  # hand frame axes in body-centric coordinates
                'y_axis': y_axis,
                'z_axis': z_axis,
                'landmarks': hand_landmarks_in_hand_frame,  # all landmarks in hand-centric coordinates
                'confidence': handedness.classification[0].score
            }
        
        return hand_frames

    def display_body_centric_info(self, frame: np.ndarray, sew_coords: dict) -> None:
        """Display body-centric SEW coordinates on frame."""
        if not sew_coords:
            return
            
        y_offset = 150
        cv2.putText(frame, "Body-Centric Coordinates (meters):", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        y_offset += 25
        
        for side in ['left', 'right']:
            if side in sew_coords:
                cv2.putText(frame, f"{side.upper()} ARM:", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 20
                
                sew = sew_coords[side]
                for joint, pos in sew.items():
                    text = f"  {joint}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, (0, 255, 0), 1, cv2.LINE_AA)
                    y_offset += 15
                y_offset += 5
    
    def display_hand_centric_info(self, frame: np.ndarray, hand_coords: dict) -> None:
        """Display hand-centric coordinate information on frame."""
        if not hand_coords:
            return
            
        # Position hand info to the right side of the frame to avoid overlap
        x_offset = frame.shape[1] - 300  # Right side of frame
        y_offset = 30
        
        cv2.putText(frame, "Hand-Centric Coordinates:", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        y_offset += 25
        
        for hand_label in ['left', 'right']:
            if hand_label in hand_coords:
                hand_info = hand_coords[hand_label]
                
                cv2.putText(frame, f"{hand_label.upper()} HAND (conf: {hand_info['confidence']:.2f}):", 
                           (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 20
                
                # Show hand frame origin in body coordinates
                origin = hand_info['origin']
                cv2.putText(frame, f"  Origin: ({origin[0]:+.3f}, {origin[1]:+.3f}, {origin[2]:+.3f})", 
                           (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
                y_offset += 15
                
                # Show key landmarks in hand-centric coordinates
                key_landmarks = ['wrist', 'thumb_tip', 'index_finger_tip', 'middle_finger_tip', 
                               'ring_finger_tip', 'pinky_tip']
                
                cv2.putText(frame, "  Hand-frame landmarks:", (x_offset, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1, cv2.LINE_AA)
                y_offset += 15
                
                for landmark_name in key_landmarks:
                    if landmark_name in hand_info['landmarks']:
                        pos = hand_info['landmarks'][landmark_name]
                        display_name = landmark_name.replace('_', ' ').title()
                        cv2.putText(frame, f"    {display_name}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})", 
                                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
                        y_offset += 12
                
                y_offset += 10  # Space between hands
    
def main():
    """Main function to run the pose and hand estimation."""
    parser = argparse.ArgumentParser(description='Real-time pose and hand estimation using MediaPipe')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum detection confidence')
    parser.add_argument('--complexity', type=int, default=1, choices=[0, 1, 2], help='Model complexity')
    parser.add_argument('--smooth', type=bool, default=True, help='Enable landmark smoothing')
    parser.add_argument('--enable_hands', action='store_true', help='Enable hand detection')
    parser.add_argument('--save_output', action='store_true', help='Save output video')
    parser.add_argument('--output_path', type=str, default='pose_estimation_output.mp4', help='Output video path')
    parser.add_argument('--width', type=int, default=640, help='Video width')
    parser.add_argument('--height', type=int, default=480, help='Video height')
    
    args = parser.parse_args()
    
    # Initialize pose and hand estimator
    pose_estimator = PoseAndHandEstimator(
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
        model_complexity=args.complexity,
        smooth_landmarks=args.smooth,
        enable_hand_detection=True
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize video writer if saving output
    video_writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        video_writer = cv2.VideoWriter(args.output_path, fourcc, fps, (args.width, args.height))
    
    hand_status = "enabled" if args.enable_hands else "disabled"
    print(f"Starting pose estimation with hand detection {hand_status}...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'r' - Reset tracking")
    print("  'w' - Toggle world coordinates display")
    if args.enable_hands:
        print("  Hand detection is enabled - fingertip positions will be shown")
    
    screenshot_counter = 0
    show_world_coords = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Process frame
            annotated_frame, landmarks, pose_results, hand_results = pose_estimator.process_frame(frame, show_world_coords)
            
            # Calculate and display FPS
            fps = pose_estimator.update_fps()
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, annotated_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Save frame if recording
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow('MediaPipe Pose Estimation', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"pose_screenshot_{screenshot_counter}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
                screenshot_counter += 1
            elif key == ord('r'):
                print("Resetting pose tracking...")
                pose_estimator.cleanup()
                pose_estimator = PoseAndHandEstimator(
                    min_detection_confidence=args.confidence,
                    min_tracking_confidence=args.confidence,
                    model_complexity=args.complexity,
                    smooth_landmarks=args.smooth,
                    enable_hand_detection=args.enable_hands
                )
            elif key == ord('w'):
                show_world_coords = not show_world_coords
                print(f"World coordinates display: {'ON' if show_world_coords else 'OFF'}")
                
    except KeyboardInterrupt:
        print("\nStopping pose estimation...")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        pose_estimator.cleanup()
        print("Cleanup completed.")


if __name__ == "__main__":
    main()
