# coding:utf-8

import os
import numpy as np
import pinocchio as pin

from controller.mdik import QPIK, MDIK

######################################################
class FloatingKinematics:
    def __init__(self, urdf, ees, grav, dt,
                zmp="q", ik="mdik", time_limit=0.5,
                priority={"leg": 1.0, "arm": 1.0, "body": 1.0},
                verbose=True,
                ):
        self.robot = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
        # set limitation for flyer
        # self.robot.upperPositionLimit[:7] = 0.0
        # self.robot.lowerPositionLimit[:7] = 0.0
        self.robot.velocityLimit[:6] = 0.0
        self.data = self.robot.createData()
        self.ids_ee = {key: self.robot.getJointId(name) for key, name in ees.items()}
        self.grav = grav
        self.dt = dt
        self.mass = pin.computeTotalMass(self.robot)
        self.zmp_mode = zmp
        self.priority = {key: priority["leg"] if "leg" in key else priority["arm"] if "arm" in key else priority["body"] if "body" in key else 0.0 for key, name in ees.items()}
        self.ik_solver = QPIK(self.dt, self.robot.createData(), time_limit=time_limit) if "qp" in ik else MDIK(self.dt, self.robot.createData(), time_limit=time_limit)
        #
        if verbose:
            print(self.robot, self.ids_ee)
        self.reset()

    def reset(self):
        # for computing metrics
        self._dq_old = None
        self._vcom_old = None
        self._dcm_old = None

######################################################
    def fk(self, q_, dq=None, ddq=None, w_=None):
        self.q_ = q_
        if dq is not None:
            if ddq is None:
                ddq = np.zeros_like(dq) if self._dq_old is None else (dq - self._dq_old) / self.dt
            pin.forwardKinematics(self.robot, self.data, q_, dq, ddq)
            self._dq_old = dq
        else:
            pin.forwardKinematics(self.robot, self.data, q_)
        # pose of ees and root
        self.update_poses()
        # com
        self.update_com()
        # dcm
        self.update_dcm()
        # zmp
        self.update_zmp(w_)

    def update_com(self):
        pin.centerOfMass(self.robot, self.data)
        self._vcom_old = self.data.vcom[0] if self._vcom_old is None else self.vcom
        self.com, self.vcom = self.data.com[0], self.data.vcom[0]
        self.acom = self.data.acom[0] if self.zmp_mode == "q" else (self.vcom - self._vcom_old) / self.dt
        return self.com, self.vcom, self.acom

    def update_dcm(self):
        self.dcm = self.com + self.vcom * np.sqrt(self.com[-1] / self.grav)
        self.vdcm = np.zeros_like(self.dcm) if self._dcm_old is None else (self.dcm - self._dcm_old) / self.dt
        self._dcm_old = self.dcm
        return self.dcm, self.vdcm

    def update_poses(self):
        self.poses = {key: self.data.oMi[id] for key, id in self.ids_ee.items()}
        self.poses["root"] = self.data.oMi[1]
        return self.poses

    def update_zmp(self, w_=None):
        if w_ is None:
            self.zmp = self.com - self.com[-1] / (self.acom[-1] + self.grav) * self.acom
        else:
            pass
        # for simplicity, walk on flat
        self.zmp[-1] = min([se3.translation[-1] for key, se3 in self.poses.items() if key != "root"])
        return self.zmp

######################################################
    def ik(self, qb, ts=None, **prs):
        q_ = self.q_.copy()
        q_[:7] = qb
        return self.ik_solver(self.robot, self.ids_ee, self.priority, q_, **prs, ts=ts)
