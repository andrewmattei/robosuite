"""
SEW Mimic Controller for RBY1 robot.
Uses human SEW poses and wrist orientations to compute joint angles via IK,
then applies joint torque control in MuJoCo.

Features:
- Geometric IK for arm pose tracking
- Gravity compensation for all torso joints (torso_0 to torso_5)
- Joint-space PD control for arm movements
- Position holding when not engaged

Author: Chuizheng Kong
Date created: 2025-07-27
"""

import numpy as np
import mujoco
from typing import Optional, Dict, Tuple
import pinocchio as pin
import os

# Import the IK function and geometric subproblems
from geometric_kinematics_rby1 import IK_3R_R_3R_SEW, load_rby1_model, get_frame_transforms_from_pinocchio
import robosuite.projects.shared_scripts.geometric_subproblems as gsp


class SEWMimicRBY1:
    """
    SEW Mimic Controller for RBY1 robot using geometric inverse kinematics.
    Converts human SEW poses to robot joint angles and applies joint torque control.
    """
    
    def __init__(self, mujoco_model, mujoco_data, urdf_path=None, 
                 kp=150.0, kd=None, debug=False):
        """
        Initialize SEW Mimic controller for RBY1.
        
        Args:
            mujoco_model: MuJoCo model
            mujoco_data: MuJoCo data
            urdf_path: Path to RBY1 URDF file (optional, uses default if None)
            kp: Proportional gain for joint control
            kd: Derivative gain for joint control (auto-computed if None)
            debug: Enable debug output
        """
        self.model = mujoco_model
        self.data = mujoco_data
        self.debug = debug
        
        # Initialize engagement state
        self.engaged = False
        
        # Enable torso gravity compensation by default
        self.torso_gravity_compensation_enabled = True
        
        # Load Pinocchio model for IK computations
        if urdf_path is None:
            urdf_path = os.path.join(os.path.dirname(__file__), 'rbyxhand_v2', 'model_modified.urdf')
        
        self.pin_model, self.pin_data = load_rby1_model(urdf_path)
        
        # Get model transforms for both arms
        self.right_arm_transforms = get_frame_transforms_from_pinocchio(self.pin_model, "right_arm")
        self.left_arm_transforms = get_frame_transforms_from_pinocchio(self.pin_model, "left_arm")
        
        # Get joint indices in MuJoCo model
        self._setup_joint_indices()
        
        # Control gains
        self.kp = kp
        self.kd = 2 * np.sqrt(kp) if kd is None else kd
        
        # Torso control gains (more conservative to prevent unwanted movement)
        self.Kp_torso = 100.0  # Lower gain for torso stability
        self.Kd_torso = 20.0   # Damping for torso
        
        # Solution indices for consistent IK solution selection
        self.right_arm_sol_ids = None
        self.left_arm_sol_ids = None
        
        # Previous joint configurations for solution selection
        self.q_prev_right = None
        self.q_prev_left = None
        
        # Goal joint angles
        self.q_goal_right = np.zeros(7)
        self.q_goal_left = np.zeros(7)
        
        # Torso goal angles (hold initial position)
        self.q_goal_torso = None
        
        # Initialize with current joint positions
        self._update_current_positions()
        
        # Initialize solution indices for consistent IK solution selection
        self._initialize_solution_indices()
        
        print("SEW Mimic RBY1 controller initialized")
        print(f"Torso gravity compensation: {'ENABLED' if self.torso_gravity_compensation_enabled else 'DISABLED'}")
        print("Status: IK system loaded - ready for teleoperation")
        if self.debug:
            print(f"Right arm joints: {self.right_arm_joint_names}")
            print(f"Left arm joints: {self.left_arm_joint_names}")
            print(f"Torso joints: {self.torso_joint_names}")
    
    def _setup_joint_indices(self):
        """Setup joint indices and names for both arms."""
        # Right arm joint names in MuJoCo
        self.right_arm_joint_names = [
            "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
            "right_arm_4", "right_arm_5", "right_arm_6"
        ]
        
        # Left arm joint names in MuJoCo  
        self.left_arm_joint_names = [
            "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
            "left_arm_4", "left_arm_5", "left_arm_6"
        ]
        
        # Torso joint names in MuJoCo
        self.torso_joint_names = [
            "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5"
        ]
        
        # Get joint qpos addresses using the correct approach
        self.right_arm_qpos_addrs = []
        self.left_arm_qpos_addrs = []
        self.torso_qpos_addrs = []
        
        # Build joint name to ID mapping like in binding_utils.py
        joint_name2id = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_name2id[joint_name] = i
        
        # Get qpos addresses for right arm joints
        for joint_name in self.right_arm_joint_names:
            if joint_name in joint_name2id:
                joint_id = joint_name2id[joint_name]
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.right_arm_qpos_addrs.append(qpos_addr)
            else:
                print(f"Warning: Could not find right arm joint {joint_name}")
        
        # Get qpos addresses for left arm joints
        for joint_name in self.left_arm_joint_names:
            if joint_name in joint_name2id:
                joint_id = joint_name2id[joint_name]
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.left_arm_qpos_addrs.append(qpos_addr)
            else:
                print(f"Warning: Could not find left arm joint {joint_name}")
        
        # Get qpos addresses for torso joints
        for joint_name in self.torso_joint_names:
            if joint_name in joint_name2id:
                joint_id = joint_name2id[joint_name]
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.torso_qpos_addrs.append(qpos_addr)
            else:
                print(f"Warning: Could not find torso joint {joint_name}")
        
        # For qvel, we need the DOF addresses
        self.right_arm_qvel_addrs = []
        self.left_arm_qvel_addrs = []
        self.torso_qvel_addrs = []
        
        # Get qvel addresses for right arm joints
        for joint_name in self.right_arm_joint_names:
            if joint_name in joint_name2id:
                joint_id = joint_name2id[joint_name]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                self.right_arm_qvel_addrs.append(qvel_addr)
            else:
                print(f"Warning: Could not find right arm joint {joint_name}")
        
        # Get qvel addresses for left arm joints
        for joint_name in self.left_arm_joint_names:
            if joint_name in joint_name2id:
                joint_id = joint_name2id[joint_name]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                self.left_arm_qvel_addrs.append(qvel_addr)
            else:
                print(f"Warning: Could not find left arm joint {joint_name}")
        
        # Get qvel addresses for torso joints
        for joint_name in self.torso_joint_names:
            if joint_name in joint_name2id:
                joint_id = joint_name2id[joint_name]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                self.torso_qvel_addrs.append(qvel_addr)
            else:
                print(f"Warning: Could not find torso joint {joint_name}")
        
        print(f"Found {len(self.right_arm_qpos_addrs)} right arm joints")
        print(f"Found {len(self.left_arm_qpos_addrs)} left arm joints")
        print(f"Found {len(self.torso_qpos_addrs)} torso joints")
        print(f"Right arm qpos addresses: {self.right_arm_qpos_addrs}")
        print(f"Left arm qpos addresses: {self.left_arm_qpos_addrs}")
        print(f"Torso qpos addresses: {self.torso_qpos_addrs}")
        print(f"Right arm qvel addresses: {self.right_arm_qvel_addrs}")
        print(f"Left arm qvel addresses: {self.left_arm_qvel_addrs}")
        print(f"Torso qvel addresses: {self.torso_qvel_addrs}")
        
        # Keep the old joint_ids for compatibility with torque application
        self.right_arm_joint_ids = [joint_name2id[name] for name in self.right_arm_joint_names if name in joint_name2id]
        self.left_arm_joint_ids = [joint_name2id[name] for name in self.left_arm_joint_names if name in joint_name2id]
        self.torso_joint_ids = [joint_name2id[name] for name in self.torso_joint_names if name in joint_name2id]
    
    def _update_current_positions(self):
        """Update current joint positions from MuJoCo data."""
        # Get current joint positions using qpos addresses
        self.q_current_right = np.array([self.data.qpos[addr] for addr in self.right_arm_qpos_addrs])
        self.q_current_left = np.array([self.data.qpos[addr] for addr in self.left_arm_qpos_addrs])
        self.q_current_torso = np.array([self.data.qpos[addr] for addr in self.torso_qpos_addrs])
        
        # Get current joint velocities using qvel addresses
        self.qd_current_right = np.array([self.data.qvel[addr] for addr in self.right_arm_qvel_addrs])
        self.qd_current_left = np.array([self.data.qvel[addr] for addr in self.left_arm_qvel_addrs])
        self.qd_current_torso = np.array([self.data.qvel[addr] for addr in self.torso_qvel_addrs])
        
        # Initialize previous positions if not set
        if self.q_prev_right is None:
            self.q_prev_right = self.q_current_right.copy()
            self.q_goal_right = self.q_current_right.copy()
            
        if self.q_prev_left is None:
            self.q_prev_left = self.q_current_left.copy()
            self.q_goal_left = self.q_current_left.copy()
            
        # Initialize torso goal to hold current position
        if self.q_goal_torso is None:
            self.q_goal_torso = self.q_current_torso.copy()
    
    def _initialize_solution_indices(self):
        """
        Initialize solution indices using the robot's actual initial joint configuration.
        This determines which solution branch to consistently select for IK.
        Uses the robot's current SEW positions at initialization to set up indexing.
        """
        try:
            # Initialize solution indices for both arms
            for arm_side in ['right', 'left']:
                if arm_side == 'right':
                    q_init = self.q_current_right.copy()
                    model_transforms = self.right_arm_transforms
                else:
                    q_init = self.q_current_left.copy()
                    model_transforms = self.left_arm_transforms
                
                # Get robot SEW positions at the initial configuration using forward kinematics
                # For now, we'll use a simple approximation - this could be improved with proper FK
                # Create dummy SEW positions based on current joint angles
                S_init = np.array([0.0, -0.2 if arm_side == 'right' else 0.2, 0.0])  # Shoulder position
                E_init = np.array([-0.05, -0.2 if arm_side == 'right' else 0.2, -0.25])  # Elbow position
                W_init = np.array([0.0, -0.2 if arm_side == 'right' else 0.2, -0.5])  # Wrist position

                R_0_7_init = gsp.rot_numerical(-np.array([0.0, 1.0, 0.0]), np.pi/10)

                # Get all possible solutions for the initial SEW configuration
                Q_solutions, is_LS_vec, human_vectors, sol_ids_used = IK_3R_R_3R_SEW(
                    S_init, E_init, W_init, model_transforms, sol_ids=None, R_0_7=R_0_7_init
                )
                
                if Q_solutions is not None and Q_solutions.shape[1] > 0:
                    # Select the solution closest to the initial joint configuration
                    best_distance = float('inf')
                    best_sol_idx = 0
                    for i in range(Q_solutions.shape[1]):
                        distance = np.linalg.norm(Q_solutions[:, i] - q_init)
                        if distance < best_distance:
                            best_distance = distance
                            best_sol_idx = i
                    
                    # Store the solution indices for the selected solution
                    if arm_side == 'right':
                        self.right_arm_sol_ids = {
                            'q12_idx': sol_ids_used['q12_idx'][best_sol_idx],
                            'q56_idx': sol_ids_used['q56_idx'][best_sol_idx],
                        }
                    else:
                        self.left_arm_sol_ids = {
                            'q12_idx': sol_ids_used['q12_idx'][best_sol_idx],
                            'q56_idx': sol_ids_used['q56_idx'][best_sol_idx],
                        }
                    
                    if self.debug:
                        print(f"SEW Controller: Initialized {arm_side} arm solution indices from q_init={q_init.round(3)}: {self.right_arm_sol_ids if arm_side == 'right' else self.left_arm_sol_ids}")
                else:
                    if self.debug:
                        print(f"SEW Controller: Warning - no IK solutions found for {arm_side} arm initial configuration")
                    
        except Exception as e:
            if self.debug:
                print(f"SEW Controller: Solution indices initialization failed: {e}")
            # Fall back to no specific indexing
            self.right_arm_sol_ids = None
            self.left_arm_sol_ids = None
    
    def set_sew_goals(self, sew_left, sew_right, wrist_left=None, wrist_right=None, engaged=None):
        """
        Set SEW goals for both arms and compute corresponding joint angles.
        
        Args:
            sew_left: Dict with 'S', 'E', 'W' keys containing 3D positions for left arm
            sew_right: Dict with 'S', 'E', 'W' keys containing 3D positions for right arm  
            wrist_left: Optional 3x3 rotation matrix for left wrist orientation
            wrist_right: Optional 3x3 rotation matrix for right wrist orientation
            engaged: Optional bool to override engagement state
        """
        # Check if valid SEW data is provided
        valid_sew_right = (sew_right is not None and 
                          sew_right.get('S') is not None and 
                          sew_right.get('E') is not None and 
                          sew_right.get('W') is not None)
        
        valid_sew_left = (sew_left is not None and
                         sew_left.get('S') is not None and
                         sew_left.get('E') is not None and
                         sew_left.get('W') is not None)
        
        # Use engagement override if provided, otherwise check for valid SEW data
        if engaged is not None:
            self.engaged = engaged and (valid_sew_right or valid_sew_left)
        else:
            self.engaged = valid_sew_right or valid_sew_left
        
        if not self.engaged:
            if self.debug:
                print("Controller DISENGAGED: No valid SEW data or not engaged")
            return
        
        # Update positions and solve IK for valid arms
        self._update_current_positions()
        
        # Process arms with IK
        if valid_sew_right:
            self._solve_ik_for_arm('right', sew_right, wrist_right)
        
        if valid_sew_left:
            self._solve_ik_for_arm('left', sew_left, wrist_left)
    
    def _solve_ik_for_arm(self, arm_side, sew_pose, wrist_rotation=None):
        """
        Solve inverse kinematics for one arm using the full geometric IK.
        
        Args:
            arm_side: 'left' or 'right'
            sew_pose: Dict with 'S', 'E', 'W' keys
            wrist_rotation: Optional 3x3 rotation matrix
        """
        try:
            # Add debugger breakpoint - uncomment to debug
            # import pdb; pdb.set_trace()
            
            # Get SEW positions
            S_human = sew_pose['S']
            E_human = sew_pose['E'] 
            W_human = sew_pose['W']
            
            # Get model transforms for this arm
            if arm_side == 'right':
                model_transforms = self.right_arm_transforms
                sol_ids = self.right_arm_sol_ids
                q_prev = self.q_prev_right
            else:
                model_transforms = self.left_arm_transforms
                sol_ids = self.left_arm_sol_ids 
                q_prev = self.q_prev_left
            
            # Prepare wrist rotation in robot frame
            R_0_7 = wrist_rotation
            
            # Call IK solver (reset previous solution IDs to avoid list-int comparison errors)
            Q, is_LS_vec, human_vectors, sol_ids_used = IK_3R_R_3R_SEW(
                S_human, E_human, W_human,
                model_transforms,
                sol_ids=sol_ids,
                R_0_7=R_0_7
            )
            
            if Q.shape[1] > 0:
                q_target = Q[:, 0]
                
                # Update goal and previous configuration
                if arm_side == 'right':
                    self.q_goal_right = q_target
                    self.q_prev_right = q_target
                else:
                    self.q_goal_left = q_target
                    self.q_prev_left = q_target
                
                if self.debug:
                    target_str = f"[{q_target[0]:.3f},{q_target[1]:.3f},{q_target[2]:.3f}...]"
                    print(f"IK solved for {arm_side} arm: {target_str}")
            else:
                if self.debug:
                    print(f"IK failed for {arm_side} arm - no valid solutions found")
        
        except Exception as e:
            if self.debug:
                print(f"Error solving IK for {arm_side} arm: {e}")
                import traceback
                traceback.print_exc()  # Print full stack trace for debugging
            # Don't update goals if IK fails
                
    
    def compute_control_torques(self):
        """
        Compute joint torques for both arms using inverse dynamics with mass matrix compensation.
        Uses the same approach as the robosuite SEW mimic controller.
        
        Returns:
            Tuple of (right_arm_torques, left_arm_torques, torso_torques)
        """
        self._update_current_positions()
        
        # Compute mass matrix and compensation terms for both arms and torso
        right_mass_matrix = self._get_arm_mass_matrix('right')
        left_mass_matrix = self._get_arm_mass_matrix('left')
        torso_mass_matrix = self._get_torso_mass_matrix()
        
        right_compensation = self._get_arm_torque_compensation('right')
        left_compensation = self._get_arm_torque_compensation('left')
        torso_compensation = self._get_torso_torque_compensation()
        
        # Always apply gravity compensation for torso joints (if enabled)
        torso_compensation = self._get_torso_gravity_compensation() if self.torso_gravity_compensation_enabled else np.zeros(6)
        
        # If not engaged, use low-gain PD to hold current position
        if not self.engaged:
            # Use lower gains when disengaged but still maintain proper control
            hold_kp = 50.0  # Reduced but still effective
            hold_kd = 10.0
            
            # Compute desired accelerations for holding current position
            q_error_right = np.zeros(7)  # No position error when holding  
            qd_error_right = -self.qd_current_right  # Damp velocities
            qdd_desired_right = hold_kp * q_error_right + hold_kd * qd_error_right
            
            q_error_left = np.zeros(7)
            qd_error_left = -self.qd_current_left
            qdd_desired_left = hold_kp * q_error_left + hold_kd * qd_error_left
            
        else:
            # Normal PD control when engaged - compute desired accelerations
            q_error_right = self.q_goal_right - self.q_current_right
            # Wrap joint position error to [-pi, pi]
            q_error_right = (q_error_right + np.pi) % (2 * np.pi) - np.pi
            qd_error_right = -self.qd_current_right  # Desired velocity is 0
            qdd_desired_right = self.kp * q_error_right + self.kd * qd_error_right
            
            q_error_left = self.q_goal_left - self.q_current_left
            # Wrap joint position error to [-pi, pi]  
            q_error_left = (q_error_left + np.pi) % (2 * np.pi) - np.pi
            qd_error_left = -self.qd_current_left
            qdd_desired_left = self.kp * q_error_left + self.kd * qd_error_left
        
        # Compute torques using inverse dynamics: tau = M * qdd + compensation
        torques_right = np.dot(right_mass_matrix, qdd_desired_right) + right_compensation
        torques_left = np.dot(left_mass_matrix, qdd_desired_left) + left_compensation

        # Torso PD control to hold initial position and prevent drift
        q_error_torso = self.q_goal_torso - self.q_current_torso
        # Wrap joint position error to [-pi, pi] for torso joints
        q_error_torso = (q_error_torso + np.pi) % (2 * np.pi) - np.pi
        qd_error_torso = -self.qd_current_torso  # Desired velocity is 0
        qdd_desired_torso = self.Kp_torso * q_error_torso + self.Kd_torso * qd_error_torso
        
        torques_torso = np.dot(torso_mass_matrix, qdd_desired_torso) + torso_compensation

        # torques_right = right_compensation
        # torques_left = left_compensation
        
        # # Clamp torques to reasonable limits
        # max_torque = 100.0  # Increased limit for mass matrix control
        # torques_right = np.clip(torques_right, -max_torque, max_torque)
        # torques_left = np.clip(torques_left, -max_torque, max_torque)
        
        if self.debug and np.random.random() < 0.01:  # Print occasionally
            if self.engaged:
                status = "ENGAGED with IK"
                q_err_right_str = f"[{q_error_right[0]:.3f},{q_error_right[1]:.3f},{q_error_right[2]:.3f}]"
                q_err_left_str = f"[{q_error_left[0]:.3f},{q_error_left[1]:.3f},{q_error_left[2]:.3f}]"
                torq_right_str = f"[{torques_right[0]:.1f},{torques_right[1]:.1f},{torques_right[2]:.1f}]"
                torq_left_str = f"[{torques_left[0]:.1f},{torques_left[1]:.1f},{torques_left[2]:.1f}]"
                print(f"Status: {status} - Right: q_err={q_err_right_str}, τ={torq_right_str} | Left: q_err={q_err_left_str}, τ={torq_left_str}")
            else:
                status = "DISENGAGED (holding position)"
                torq_right_str = f"[{torques_right[0]:.1f},{torques_right[1]:.1f},{torques_right[2]:.1f}]"
                torq_left_str = f"[{torques_left[0]:.1f},{torques_left[1]:.1f},{torques_left[2]:.1f}]"
                print(f"Status: {status} - Right: τ={torq_right_str} | Left: τ={torq_left_str}")
        
        return torques_right, torques_left, torques_torso
    
    def _get_arm_mass_matrix(self, arm_side):
        """
        Get the mass matrix for one arm from MuJoCo.
        
        Args:
            arm_side: 'left' or 'right'
            
        Returns:
            np.array: 7x7 mass matrix for the arm
        """
        # Get qvel addresses for this arm
        if arm_side == 'right':
            qvel_addrs = self.right_arm_qvel_addrs
        else:
            qvel_addrs = self.left_arm_qvel_addrs
        
        # Get full mass matrix from MuJoCo
        full_mass_matrix = np.ndarray(shape=(self.model.nv, self.model.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.model, full_mass_matrix, self.data.qM)
        full_mass_matrix = np.reshape(full_mass_matrix, (len(self.data.qvel), len(self.data.qvel)))
        
        
        # Extract submatrix for this arm using qvel addresses
        arm_mass_matrix = full_mass_matrix[qvel_addrs, :][:, qvel_addrs]
        
        # Add small regularization to avoid singularities
        # arm_mass_matrix += np.eye(7) * 1e-6
        
        return arm_mass_matrix
    
    def _get_arm_torque_compensation(self, arm_side):
        """
        Get bias forces (gravity + Coriolis) for one arm from MuJoCo.
        
        Args:
            arm_side: 'left' or 'right'
            
        Returns:
            np.array: 7-element bias force vector for the arm
        """
        # Get qvel addresses for this arm
        if arm_side == 'right':
            qvel_addrs = self.right_arm_qvel_addrs
        else:
            qvel_addrs = self.left_arm_qvel_addrs
        
        # Get bias forces from MuJoCo (includes gravity and Coriolis/centrifugal forces)
        bias_forces = np.zeros(7)
        for i, qvel_addr in enumerate(qvel_addrs[:7]):  # Only first 7 joints
            if qvel_addr < self.model.nv:
                bias_forces[i] = self.data.qfrc_bias[qvel_addr]
        
        return bias_forces
    
    def _get_torso_mass_matrix(self):
        """
        Get the mass matrix for the torso joints from MuJoCo.
        
        Returns:
            np.array: 6x6 mass matrix for the torso
        """
        # Get qvel addresses for torso
        qvel_addrs = self.torso_qvel_addrs
        
        # Get full mass matrix from MuJoCo
        full_mass_matrix = np.ndarray(shape=(self.model.nv, self.model.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(self.model, full_mass_matrix, self.data.qM)
        full_mass_matrix = np.reshape(full_mass_matrix, (len(self.data.qvel), len(self.data.qvel)))
        
        # Extract submatrix for torso using qvel addresses
        torso_mass_matrix = full_mass_matrix[qvel_addrs, :][:, qvel_addrs]
        
        # Add small regularization to avoid singularities
        torso_mass_matrix += np.eye(6) * 1e-6
        
        return torso_mass_matrix
    
    def _get_torso_torque_compensation(self):
        """
        Get bias forces (gravity + Coriolis) for torso joints from MuJoCo.
        
        Returns:
            np.array: 6-element bias force vector for the torso
        """
        # Get bias forces from MuJoCo (includes gravity and Coriolis/centrifugal forces)
        bias_forces = np.zeros(6)
        for i, qvel_addr in enumerate(self.torso_qvel_addrs[:6]):  # Only first 6 joints
            if qvel_addr < self.model.nv:
                bias_forces[i] = self.data.qfrc_bias[qvel_addr]
        
        return bias_forces
    
    def _get_torso_gravity_compensation(self):
        """
        Get gravity compensation torques for torso joints from MuJoCo.
        
        Returns:
            np.array: 6-element gravity compensation vector for torso joints
        """
        # Get bias forces from MuJoCo (includes gravity and Coriolis/centrifugal forces)
        gravity_compensation = np.zeros(6)
        for i, qvel_addr in enumerate(self.torso_qvel_addrs[:6]):  # Only first 6 joints
            if qvel_addr < self.model.nv:
                gravity_compensation[i] = self.data.qfrc_bias[qvel_addr]
        
        return gravity_compensation
    
    def apply_torques(self, torques_right, torques_left, torques_torso=None):
        """
        Apply computed torques to MuJoCo actuators.
        
        Args:
            torques_right: Array of 7 torques for right arm
            torques_left: Array of 7 torques for left arm
            torques_torso: Array of 6 torques for torso joints (optional)
        """
        # Apply right arm torques
        for i, joint_id in enumerate(self.right_arm_joint_ids):
            if i < len(torques_right):
                # Find actuator that controls this joint
                for actuator_id in range(self.model.nu):
                    if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                        self.data.ctrl[actuator_id] = torques_right[i]
                        break
        
        # Apply left arm torques
        for i, joint_id in enumerate(self.left_arm_joint_ids):
            if i < len(torques_left):
                # Find actuator that controls this joint
                for actuator_id in range(self.model.nu):
                    if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                        self.data.ctrl[actuator_id] = torques_left[i]
                        break
        
        # Apply torso torques (gravity compensation)
        if torques_torso is not None:
            for i, joint_id in enumerate(self.torso_joint_ids):
                if i < len(torques_torso):
                    # Find actuator that controls this joint
                    for actuator_id in range(self.model.nu):
                        if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                            self.data.ctrl[actuator_id] = torques_torso[i]
                            break
    
    def update_control(self, sew_left, sew_right, wrist_left=None, wrist_right=None, engaged=None):
        """
        Complete control update: set goals, compute torques, and apply them.
        
        Args:
            sew_left: Left arm SEW pose dict or None
            sew_right: Right arm SEW pose dict or None
            wrist_left: Left wrist rotation matrix or None
            wrist_right: Right wrist rotation matrix or None
            engaged: Optional bool to override engagement state
        """
        # Set new goals with engagement override
        self.set_sew_goals(sew_left, sew_right, wrist_left, wrist_right, engaged)
        
        # Compute control torques (now includes torso)
        torques_right, torques_left, torques_torso = self.compute_control_torques()
        
        # Apply torques (including torso gravity compensation)
        self.apply_torques(torques_right, torques_left, torques_torso)
    
    def reset_to_home_position(self):
        """Reset both arms to home position."""
        # Set goals to zero (home position)
        self.q_goal_right = np.zeros(7)
        self.q_goal_left = np.zeros(7)
        
        # Reset solution tracking
        self.right_arm_sol_ids = None
        self.left_arm_sol_ids = None
        self.q_prev_right = None
        self.q_prev_left = None
        
        # Reset engagement
        self.engaged = False
        
        print("Reset to home position - Controller DISENGAGED")
    
    def is_engaged(self):
        """
        Check if controller is currently engaged.
        
        Returns:
            bool: True if engaged and actively controlling, False if disengaged
        """
        return self.engaged
    
    def get_current_joint_angles(self):
        """
        Get current joint angles for both arms and torso.
        
        Returns:
            Tuple of (right_arm_angles, left_arm_angles, torso_angles)
        """
        self._update_current_positions()
        return self.q_current_right.copy(), self.q_current_left.copy(), self.q_current_torso.copy()
    
    def get_goal_joint_angles(self):
        """
        Get goal joint angles for both arms.
        
        Returns:
            Tuple of (right_arm_goals, left_arm_goals)
        """
        return self.q_goal_right.copy(), self.q_goal_left.copy()
    
    def set_torso_gravity_compensation(self, enabled):
        """
        Enable or disable gravity compensation for torso joints.
        
        Args:
            enabled: bool, True to enable gravity compensation, False to disable
        """
        self.torso_gravity_compensation_enabled = enabled
        if self.debug:
            status = "ENABLED" if enabled else "DISABLED"
            print(f"Torso gravity compensation {status}")
    
    def is_torso_gravity_compensation_enabled(self):
        """
        Check if torso gravity compensation is enabled.
        
        Returns:
            bool: True if enabled, False if disabled
        """
        return self.torso_gravity_compensation_enabled
