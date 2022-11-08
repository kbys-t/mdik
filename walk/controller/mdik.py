# coding:utf-8

import time
import numpy as np
import pinocchio as pin

import qpsolvers
from numba import njit

######################################################
@njit
def _diff_mirror(g_, step_size, dqul, qn):
    eg = np.exp(step_size * g_)
    tmp = qn * (1.0 - eg)
    diff = dqul * (tmp * (1.0 - qn)) / (tmp + eg)
    diff[~np.isfinite(diff)] = 0.0
    return diff

@njit
def _interp(q1, q2, ratio):
    lambd = 1.0 / (1.0 + ratio)
    return lambd * q1 + (1.0 - lambd) * q2

######################################################
class Accelerator:
    # see: https://papers.nips.cc/paper/2015/hash/f60bb6bb4c96d4df93c51bd69dcc15a0-Abstract.html
    def __init__(self, accel=5.0, gamma=2.0, step_size=1.0, smooth_reset=0.5):
        self.active = accel >= 3.0
        if self.active:
            self.accel_inv = 1.0 / accel
            self.step_size = step_size / gamma
            self.smooth_reset = (smooth_reset, 1.0 - smooth_reset)
            self.z_ = None
            self.ratio = 0.0

    def reset(self, z_, ql, qu):
        if self.active:
            self.z_ = z_ if self.z_ is None or self.smooth_reset[0] == 0.0 else (self.smooth_reset[0] * self.z_ + self.smooth_reset[1] * z_).clip(ql, qu)
            self.ql = ql
            self.qu = qu
            self.ratio *= self.smooth_reset[0]

    def __call__(self, robot, q_, g_):
        if self.active:
            self.ratio += self.accel_inv
            self.z_ = pin.integrate(robot, self.z_, - self.ratio * self.step_size * g_).clip(self.ql, self.qu)
            return _interp(self.z_, q_, self.ratio)
        else:
            return q_

######################################################
class QPIK:
    def __init__(self, dt, data,
        weight=[1.0]*6, time_limit=0.1,
        damper=1e-3,
        ):
        self.dt = dt
        self.data = data
        self.weight = np.array(weight)
        self.weight *= len(self.weight) / self.weight.sum()
        self.weight = self.weight.reshape(-1, 1)
        self.time_limit = time_limit * dt
        #
        self.damper = damper

    def __call__(self, robot, ees, priority, q0, ts=None, **prs):
        if ts is None:
            ts = time.time()
        pin.forwardKinematics(robot, self.data, q0)
        # make limitation
        qm = pin.integrate(robot, q0, - robot.velocityLimit * self.dt)
        qp = pin.integrate(robot, q0,   robot.velocityLimit * self.dt)
        ql = np.maximum(robot.lowerPositionLimit, np.minimum(qm, qp))
        qu = np.minimum(robot.upperPositionLimit, np.maximum(qm, qp))
        # formulation
        errs = []
        Js = []
        WJs = []
        for key, pr in prs.items():
            id = ees[key]
            wee = priority[key]
            if wee:
                errs.append(pin.log(self.data.oMi[id].actInv(pr)).vector)
                Js.append(pin.computeJointJacobian(robot, self.data, q0, id))
                WJs.append(wee * self.weight * Js[-1])
        errs = np.concatenate(errs, axis=0)
        Js = np.concatenate(Js, axis=0)
        WJs = np.concatenate(WJs, axis=0)
        C_ = Js.T.dot(WJs) + self.damper * np.eye(Js.shape[1])
        c_ = - (errs.reshape(1, -1).dot(WJs)).flatten()
        lb = pin.difference(robot, q0, ql)
        ub = pin.difference(robot, q0, qu)
        # solve
        time_limit = self.time_limit - (time.time() - ts)
        if time_limit > 0.0:
            dq = qpsolvers.solve_qp(C_, c_, None, None, None, None, lb, ub, solver="osqp", time_limit=time_limit)
        else:
            dq = None
        if dq is not None:
            q_ = pin.integrate(robot, q0, dq.clip(lb, ub))
        else:
            q_ = q0
        return q_

######################################################
class MDIK:
    def __init__(self, dt, data,
        weight=[1.0]*6, time_limit=0.1,
        margin=1e-2, step_size=1.0,
        accel=5.0, gamma=2.0, smooth=0.5,
        ):
        self.dt = dt
        self.data = data
        self.weight = np.array(weight)
        self.weight = self.weight.reshape(-1, 1)
        self.time_limit = time_limit * dt
        #
        self.margin = margin
        self.step_size = step_size * 2.0 * np.log((1.0 - margin) / margin)
        #
        self.accelerator = Accelerator(accel, gamma, step_size, smooth)

    def __call__(self, robot, ees, priority, q0, ts=None, **prs):
        if ts is None:
            ts = time.time()
        pin.forwardKinematics(robot, self.data, q0)
        q_ = q0.copy()
        # make limitation
        qm = pin.integrate(robot, q0, - robot.velocityLimit * self.dt)
        qp = pin.integrate(robot, q0,   robot.velocityLimit * self.dt)
        ql = np.maximum(robot.lowerPositionLimit, np.minimum(qm, qp))
        qu = np.minimum(robot.upperPositionLimit, np.maximum(qm, qp))
        dqul = pin.difference(robot, ql, qu)
        mask = (dqul == 0.0)
        #
        self.accelerator.reset(q0, ql, qu)
        #
        err0s = []
        Js = []
        JTws = []
        for key, pr in prs.items():
            id = ees[key]
            wee = priority[key]
            if wee:
                err0s.append(pin.log(self.data.oMi[id].actInv(pr)).vector)
                Js.append(pin.computeJointJacobian(robot, self.data, q0, id))
                JTws.append((wee * self.weight * Js[-1]).T)
        err0s = np.concatenate(err0s, axis=0)
        Js = np.concatenate(Js, axis=0)
        JTws = np.concatenate(JTws, axis=1)
        # iteration until time up or convergence
        errs = err0s
        while time.time() - ts < self.time_limit:
            # compute gradient
            g_ = - JTws.dot(errs)
            # to satisfy software limitations
            q_, qn = self._normalize(robot, q_, ql, dqul, mask)
            # update under constraint
            q_ = self._solve(robot, q_, g_, dqul, qn)
            # replace by accelerator
            q_ = self.accelerator(robot, q_, g_)
            # update (estimated) error
            errs = err0s - Js.dot(pin.difference(robot, q0, q_))
        return q_



    def _normalize(self, robot, q_, ql, dqul, mask):
        q_normalize = (pin.difference(robot, ql, q_) / dqul).clip(self.margin, 1.0 - self.margin)
        q_normalize[mask] = 0.0
        q_margin = pin.integrate(robot, ql, q_normalize * dqul)
        return q_margin, q_normalize

    def _solve(self, robot, q_, g_, dqul, qn):
        return pin.integrate(robot, q_, _diff_mirror(g_, self.step_size, dqul, qn))
