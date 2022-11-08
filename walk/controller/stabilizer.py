# coding:utf-8

import numpy as np
import pinocchio as pin

######################################################
class BodyStabilizer:
    def __init__(self, dt, qb_ini, com, body_pose, gain=0.0, limit=0.0):
        self._root_pos_delta = qb_ini[:3] - com
        self._root_quat_des = pin.Quaternion(qb_ini[3:].copy())
        body_pos, body_rot = body_pose.translation.copy(), body_pose.rotation.copy()
        self._body_offset = body_pos - qb_ini[:3]
        self._body_integral = np.zeros_like(body_pos)
        self._body_rot_des = body_rot.copy()
        self._gain = gain * dt
        self._limit = limit

    def __call__(self, com_ref, root_pos, com):
        root_pos_ref = com_ref + self._root_pos_delta
        root_quat_ref = self._root_quat_des
        root_ref = pin.SE3ToXYZQUAT(pin.SE3(root_quat_ref, root_pos_ref))
        delta = root_pos - com
        self._body_integral = np.clip(self._body_integral + self._gain * (delta - self._root_pos_delta), -self._limit, self._limit)
        body_pos_ref = root_pos_ref + self._body_offset + self._body_integral
        body_ref = pin.SE3(self._body_rot_des, body_pos_ref)
        return root_ref, body_ref
