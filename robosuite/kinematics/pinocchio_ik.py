import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve

def compute_ik(urdf_path, ee_frame, target_pose, q0, max_iter=100, tol=1e-4):
    """
    Compute inverse kinematics using Pinocchio.
    
    Args:
        urdf_path (str): Path to the robot's URDF.
        ee_frame (str): Name of the end-effector frame.
        target_pose (np.ndarray): Desired 4x4 homogeneous transformation.
        target_pose (np.ndarray): Desired 3x1 position.
        q0 (np.ndarray): Initial configuration.
        max_iter (int): Maximum iterations.
        tol (float): Tolerance.
        
    Returns:
        np.ndarray: Joint configuration achieving target_pose.
    """
    # model = pin.buildModelFromUrdf(urdf_path)
    # data = model.createData()
    # # Use the default Pinocchio solver (e.g., Levenberg-Marquardt) as a placeholder
    # q = q0.copy()
    # q_pin = standard_to_pinocchio(model, q)
    # for _ in range(max_iter):
    #     pin.forwardKinematics(model, data, q_pin)
    #     pin.updateFramePlacements(model, data)
    #     current_pose = data.oMf[model.getFrameId(ee_frame)]
    #     error_mat = pin.log(current_pose.inverse() * pin.SE3(target_pose[:3,:3], target_pose[:3,3]))
    #     err = np.linalg.norm(error_mat)
    #     if err < tol:
    #         break
    #     # Compute Jacobian
    #     J = pin.computeFrameJacobian(model, data, q_pin, model.getFrameId(ee_frame))
    #     dq = np.linalg.lstsq(J, error_mat, rcond=None)[0]
    #     q += dq
    # return q

    oMdes = pin.SE3(target_pose[:3,:3], target_pose[:3,3])

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    # Use the default Pinocchio solver (e.g., Levenberg-Marquardt) as a placeholder
    tool_frame_id = model.getFrameId(ee_frame)

    q = q0.copy()
    q_pin = standard_to_pinocchio(model, q)
    # q_pin      = pin.neutral(model)
    eps    = 1e-4
    IT_MAX = 1000
    DT     = 1e-1
    damp   = 1e-12
    
    i=0
    while True:
        pin.forwardKinematics(model,data,q_pin)
        pin.updateFramePlacement(model, data, tool_frame_id)
        # TODO debug tool_frame_id
        # dMi_1 = oMdes.actInv(data.oMi[tool_frame_id])
        # dMi_2 = data.oMi[tool_frame_id].actInv(oMdes)
        dMi_2 = len(data.oMi)
        dMi = dMi_2
        err = pin.log(dMi).vector
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeJointJacobian(model,data,q_pin ,tool_frame_id)
        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q_pin = pin.integrate(model,q_pin,v*DT)
        if not i % 10:
            print('%d: error = %s' % (i, err.T))
        i += 1
    
    if success:
        print("Convergence achieved!")
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
    
    print('\nresult: %s' % q.flatten().tolist())
    print('\nfinal error: %s' % err.T)

def standard_to_pinocchio(self, q: np.ndarray) -> np.ndarray:
    """Convert standard joint angles (rad) to Pinocchio joint angles"""
    q_pin = np.zeros(self.nq)
    for i, j in enumerate(self.joints[1:]):
        if j.nq == 1:
            q_pin[j.idx_q] = q[j.idx_v]
        else:
            # cos(theta), sin(theta)
            q_pin[j.idx_q:j.idx_q+2] = np.array([np.cos(q[j.idx_v]), np.sin(q[j.idx_v])])
    return q_pin

def pinocchio_to_standard(self, q_pin: np.ndarray) -> np.ndarray:
    """Convert Pinocchio joint angles to standard joint angles (rad)"""
    q = np.zeros(self.model.nv)
    for i, j in enumerate(self.model.joints[1:]):
        if j.nq == 1:
            q[j.idx_v] = q_pin[j.idx_q]
        else:
            q_back = np.arctan2(q_pin[j.idx_q+1], q_pin[j.idx_q])
            q[j.idx_v] = q_back + 2*np.pi if q_back < 0 else q_back
    return q
