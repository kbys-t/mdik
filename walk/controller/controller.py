# coding:utf-8

import time
import numpy as np
import pinocchio as pin

from controller.kinematics import FloatingKinematics
from controller.generator import GaitGenerator
from controller.stabilizer import BodyStabilizer

######################################################
class WalkController:
    def __init__(self, urdf, ees, names, grav, dt,
        qb_ini=None, qj_ini=None,
        kin_params={}, gen_params={}, stb_params={},
        # footsteps_default=[{"step": np.array([0.15, 0.0, 0.0]), "ssp": 0.9, "dsp": 0.1}] + [{"step": np.array([0.3, 0.0, 0.0]), "ssp": 0.9, "dsp": 0.1}]*48,
        footsteps_default=([{"step": np.array([0.25, 0.0, 0.0]), "ssp": 0.95, "dsp": 0.05},
                            {"step": np.array([0.25, -0.25, 0.0]), "ssp": 0.95, "dsp": 0.05},
                            {"step": np.array([0.0, -0.25, 0.0]), "ssp": 0.95, "dsp": 0.05},
                            {"step": np.array([-0.25, 0.0, 0.0]), "ssp": 0.95, "dsp": 0.05},
                            {"step": np.array([-0.25, 0.25, 0.0]), "ssp": 0.95, "dsp": 0.05},
                            {"step": np.array([0.0, 0.25, 0.0]), "ssp": 0.95, "dsp": 0.05},
                            ]*5)[:-1],
        verbose=True,
        ):
        # initialize floating kinematics w/ robot model
        self.kin = FloatingKinematics(urdf, ees, grav, dt, **kin_params, verbose=verbose)
        self._map_e2m = np.array([self.kin.robot.getJointId(name)-2 for name in names])
        self._map_m2e = np.argsort(self._map_e2m)
        if qb_ini is None:
            qb_ini = np.zeros(7)
            qb_ini[2] = 1.0
        if qj_ini is None:
            qj_ini = pin.neutral(self.kin.robot)
        self.kin.fk(np.concatenate([qb_ini, qj_ini[self._map_e2m]]))
        # initialize gait generator based on dcm + vhip
        leg_pos = {key: se3.translation.copy() for key, se3 in self.kin.poses.items() if "leg" in key}
        self._leg_rot = {key: se3.rotation.copy() for key, se3 in self.kin.poses.items() if "leg" in key}
        self.gen = GaitGenerator(grav, dt, self.kin.com.copy(), leg_pos, **gen_params)
        # initialize simple body stabilizer
        self.stab = BodyStabilizer(dt, qb_ini, self.kin.com, self.kin.poses["body"], **stb_params)
        # store default footstep
        self._footsteps_default = footsteps_default

    def __call__(self, qb, dqb, qj, dqj, contact):
        ts = time.time()
        # compute forward kinematics + metrics
        q_ = np.concatenate([qb, qj[self._map_e2m]])
        dq = np.concatenate([dqb, dqj[self._map_e2m]])
        self.kin.fk(q_, dq)
        # compute reference positions
        com_ref, leg_ref = self.gen.step(contact)
        leg_ref = {key: pin.SE3(self._leg_rot[key], pos) for key, pos in leg_ref.items()}
        root_ref, body_ref = self.stab(com_ref, qb[:3], self.kin.com)
        # compute command
        q_ref = self.kin.ik(root_ref, **leg_ref, body=body_ref, ts=ts)
        return root_ref, q_ref[7:][self._map_m2e]

    def start(self, first=None, footsteps=None):
        verbose = True
        if first is None:
            first = list(self.gen.leg_pos.keys())[0]
        if footsteps is None:
            footsteps = self._footsteps_default
            verbose = False
        self.gen.start(first, footsteps, verbose)

    def trajectory(self):
        return self.kin.com, self.kin.dcm, self.kin.zmp

    def fail(self):
        return self.kin.com[-1] < 0.5
