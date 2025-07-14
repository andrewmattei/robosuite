"""
Real-time pose estimation using MediaPipe.

This script captures video from a webcam and performs real-time human pose estimation
using Google's MediaPipe library. It displays the pose landmarks and connections
on the video feed.

Usage:
    python realtime_pose_estimation.py [--camera_id CAMERA_ID] [--confidence CONFIDENCE]

Args:
    --camera_id: Camera device ID (default: 0)
    --confidence: Minimum detection confidence (default: 0.5)
    --complexity: Model complexity (0, 1, or 2) (default: 1)
    --smooth: Enable landmark smoothing (default: True)
    --save_output: Save the output video to file (default: False)
    --output_path: Path to save output video (default: pose_estimation_output.mp4)

Controls:
    - Press 'q' to quit
    - Press 's' to save a screenshot
    - Press 'r' to reset pose tracking
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import os
from typing import Optional, Tuple, List


class PoseEstimator:
    """Real-time pose estimation using MediaPipe."""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True):
        """
        Initialize the pose estimator.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks
        """
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks
            )
            
            # Performance tracking
            self.fps_counter = 0
            self.start_time = time.time()
            self.frame_times = []
            
            print("MediaPipe Pose initialized successfully")
            
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            print("This might be due to protobuf version compatibility issues.")
            print("Try running: pip install protobuf==3.20.3")
            raise
        
    def process_frame(self, frame: np.ndarray, show_world_coords: bool = True) -> Tuple[np.ndarray, Optional[object], Optional[object]]:
        """
        Process a single frame and detect poses.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (annotated_frame, pose_landmarks, pose_results)
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Perform pose detection
            results = self.pose.process(rgb_frame)
            
            # Convert back to BGR for OpenCV
            rgb_frame.flags.writeable = True
            annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Add landmark coordinates as text
                self._add_landmark_info(annotated_frame, results.pose_landmarks)
                
                # Get body-centric coordinates and display them if enabled
                if show_world_coords:
                    body_centric_coords = self.get_body_centric_coordinates(results)
                    if body_centric_coords:
                        self.display_body_centric_info(annotated_frame, body_centric_coords)
            
            return annotated_frame, results.pose_landmarks, results
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Return the original frame if processing fails
            return frame, None, None
    
    def _add_landmark_info(self, frame: np.ndarray, landmarks) -> None:
        """Add landmark information to the frame."""
        # Display key landmarks with 3D coordinates
        key_landmarks = {
            'L_Shoulder (S)': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'L_Elbow (E)': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'L_Wrist (W)': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'R_Shoulder (S)': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'R_Elbow (E)': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'R_Wrist (W)': self.mp_pose.PoseLandmark.RIGHT_WRIST,
        }
        
        y_offset = 30
        for name, landmark_id in key_landmarks.items():
            landmark = landmarks.landmark[landmark_id]
            if landmark.visibility > 0.5:  # Only show visible landmarks
                # Display 3D coordinates (normalized)
                x, y, z = landmark.x, landmark.y, landmark.z
                text = f"{name}: ({x:.3f}, {y:.3f}, {z:.3f})"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.45, (0, 255, 0), 1, cv2.LINE_AA)
                y_offset += 20
    
    def calculate_pose_angles(self, landmarks) -> dict:
        """
        Calculate key pose angles.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary of calculated angles
        """
        if not landmarks:
            return {}
        
        def calculate_angle(a, b, c):
            """Calculate angle between three points."""
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
                
            return angle
        
        angles = {}
        
        try:
            # Left arm angle (shoulder-elbow-wrist)
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            angles['left_arm'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Right arm angle
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            angles['right_arm'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Left leg angle (hip-knee-ankle)
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            angles['left_leg'] = calculate_angle(left_hip, left_knee, left_ankle)
            
            # Right leg angle
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            angles['right_leg'] = calculate_angle(right_hip, right_knee, right_ankle)
            
        except Exception as e:
            print(f"Error calculating angles: {e}")
            
        return angles
    
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
    
    def get_body_centric_coordinates(self, pose_results):
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
    
def main():
    """Main function to run the pose estimation."""
    parser = argparse.ArgumentParser(description='Real-time pose estimation using MediaPipe')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum detection confidence')
    parser.add_argument('--complexity', type=int, default=1, choices=[0, 1, 2], help='Model complexity')
    parser.add_argument('--smooth', type=bool, default=True, help='Enable landmark smoothing')
    parser.add_argument('--save_output', action='store_true', help='Save output video')
    parser.add_argument('--output_path', type=str, default='pose_estimation_output.mp4', help='Output video path')
    parser.add_argument('--width', type=int, default=640, help='Video width')
    parser.add_argument('--height', type=int, default=480, help='Video height')
    
    args = parser.parse_args()
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
        model_complexity=args.complexity,
        smooth_landmarks=args.smooth
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
    
    print("Starting pose estimation...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'r' - Reset tracking")
    print("  'a' - Toggle angle display")
    print("  'w' - Toggle world coordinates display")
    
    screenshot_counter = 0
    show_angles = False
    show_world_coords = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Process frame
            annotated_frame, landmarks, pose_results = pose_estimator.process_frame(frame, show_world_coords)
            
            # Calculate and display FPS
            fps = pose_estimator.update_fps()
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, annotated_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Calculate and display angles if enabled
            if show_angles and landmarks:
                angles = pose_estimator.calculate_pose_angles(landmarks)
                y_offset = annotated_frame.shape[0] - 100
                for name, angle in angles.items():
                    text = f"{name}: {angle:.1f}°"
                    cv2.putText(annotated_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    y_offset += 20
            
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
                pose_estimator = PoseEstimator(
                    min_detection_confidence=args.confidence,
                    min_tracking_confidence=args.confidence,
                    model_complexity=args.complexity,
                    smooth_landmarks=args.smooth
                )
            elif key == ord('a'):
                show_angles = not show_angles
                print(f"Angle display: {'ON' if show_angles else 'OFF'}")
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
