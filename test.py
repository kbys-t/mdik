# coding:utf-8

import os
import time
from tqdm import tqdm
from tqdm.contrib import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pinocchio as pin
from pinocchio.visualize.panda3d_visualizer import Panda3dVisualizer
from panda3d_viewer import ViewerClosedError
from example_robot_data import robots_loader

from config import *
np.seterr(all="warn" if is_debug else "ignore")

######################################################
def normalize(model, q_, ql, dqul, margin):
    q_normalize = (pin.difference(model, ql, q_) / dqul).clip(margin, 1.0 - margin)
    q_normalize[dqul == 0.0] = 0.0
    q_margin = pin.integrate(model, ql, q_normalize * dqul)
    return q_margin, q_normalize

######################################################
# init robot and viewer
robot = robots_loader.load(name)
robot.setVisualizer(Panda3dVisualizer())
robot.initViewer(load_model=True)
robot.model.velocityLimit[robot.model.velocityLimit > 1e+16] = 0.0

if is_debug:
    print(robot.model.lowerPositionLimit, robot.model.upperPositionLimit, robot.model.velocityLimit)
    print(robot.model)
    # print(robot.__dict__.keys())
    # import inspect
    # for m in inspect.getmembers(robot, inspect.ismethod):
    #     print(m)

robot.viewer.append_group("world")
robot.viewer.append_capsule("world", "destination", radius=0.05, length=0.1)

######################################################
# main process
with tqdm(itertools.product(range(n_resume+1, n_trial+1), gradients, constraints)) as pbar:
    for trial, gradient, constraint in pbar:
        # set random seed as trial number
        np.random.seed(trial)

        # make save directory and lists for storing data
        sdir = "./result/" + task + "/" + name + "/" + gradient + "/" + constraint + "/" + str(trial) + "/"
        os.makedirs(sdir, exist_ok=True)
        images = []
        times = []
        successes = []
        trajs = []
        errors = []
        qs = []

        # init robot
        q0 = pin.neutral(robot.model)
        if name == "talos":
            q0[2] = 1.0
        q_ = q0.copy()

        # display once as reset
        robot.viewer.set_material("world", "destination", color_rgba=(0.1, 0.1, 0.8, 1))
        robot.viewer.move_nodes("world", {"destination": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))})
        robot.display(q_)
        oM = robot.placement(q_, id_ee, False)

        # init target pose
        if "regulation" in task:
            rpy = np.random.uniform(low_rot, high_rot, 3)
            trans = np.random.uniform(low_trans, high_trans, 3)
        elif "tracking" in task:
            rpy = 0.5 * (low_rot + high_rot)
            trans = 0.5 * (low_trans + high_trans)
            freq = np.random.uniform(-freq_max, freq_max, 6)

        oMdes = pin.SE3(pin.rpy.rpyToMatrix(rpy), trans)
        tar = pin.SE3ToXYZQUATtuple(oMdes)
        if is_debug:
            print("neutral", pin.SE3ToXYZQUATtuple(oM))
            print("diff", pin.SE3ToXYZQUATtuple(oM.actInv(oMdes)))
            print("target", tar)

        # visualize initial situation
        robot.viewer.move_nodes("world", {"destination": (tar[:3], tar[-1:] + tar[3:6])})
        robot.display(q_)

        # test
        for tstep, telapsed in enumerate(np.arange(0.0, tmax * (1.0 + tperiod), dt)):
            if telapsed > tmax:
                ratio = 0.5 * (1.0 + np.sin(2.0 * np.pi * (telapsed - tmax) * freq))
                rpy = (1.0 - ratio[3:]) * low_rot + ratio[3:] * high_rot
                trans = (1.0 - ratio[:3]) * low_trans + ratio[:3] * high_trans
                oMdes = pin.SE3(pin.rpy.rpyToMatrix(rpy), trans)
                tar = pin.SE3ToXYZQUATtuple(oMdes)
                robot.viewer.set_material("world", "destination", color_rgba=(0.1, 0.1, 0.8, 1))
                robot.viewer.move_nodes("world", {"destination": (tar[:3], tar[-1:] + tar[3:6])})
                robot.display(q0)
            success = False
            # make limitation
            qm = pin.integrate(robot.model, q0, - robot.model.velocityLimit * dt)
            qp = pin.integrate(robot.model, q0,   robot.model.velocityLimit * dt)
            ql = np.maximum(robot.model.lowerPositionLimit, np.minimum(qm, qp))
            qu = np.minimum(robot.model.upperPositionLimit, np.maximum(qm, qp))
            dqul = pin.difference(robot.model, ql, qu)
            # iteration until time up or convergence
            ts = time.time()
            for iter in range(max_iter):
                # to satisfy hard-soft limitations
                q_, qn = normalize(robot.model, q_, ql, dqul, margin)
                # compute error (true in placement is for forwardkinematics flag)
                dM = robot.placement(q_, id_ee, True).actInv(oMdes)
                err = pin.log(dM).vector
                loss = 0.5 * (err * weight * err).sum()
                # check termination
                if iter > 0 and loss < eps_err:
                    success = True
                    break
                elif is_timelimit and time.time() - ts >= dt:
                    break
                else:
                    # compute gradient
                    J_ = robot.computeJointJacobian(q_, id_ee)
                    JTw = J_.T * weight.reshape(1, -1)
                    if "LM" in gradient:
                        g_ = - JTw.dot(np.linalg.solve(J_.dot(JTw) + (damp + loss) * np.eye(6), err))
                    elif "JT" in gradient:
                        g_ = - JTw.dot(err)
                    # constraint
                    if "projected" in constraint:
                        q_ = pin.integrate(robot.model, q_, - alpha * g_).clip(ql, qu)
                    else:
                        # special case with sigmoid function
                        if "parameterized" in constraint:
                            eg = np.exp(gain * g_ * dqul * gain * qn * (1.0 - qn))
                        elif "mirror" in constraint:
                            eg = np.exp(gain * g_)
                        diff = (1.0 - qn) * (1.0 - eg)/ (qn + (1.0 - qn) * eg)
                        diff[eg == float("inf")] = -1.0
                        q_ = pin.integrate(robot.model, q_, dqul * qn * diff)
            # save results
            times.append([time.time() - ts, iter])
            if is_record:
                if "regulation" in task:
                    successes.append(int(success))
                elif "tracking" in task:
                    trajs.append(pin.SE3ToXYZQUATtuple(robot.placement(q_, id_ee, True)) + tar)
                errors.append(err.tolist())
                qs.append(q_.tolist())
            if tstep % frame_skip == 0:
                if success:
                    robot.viewer.set_material("world", "destination", color_rgba=(0.8, 0.1, 0.1, 1))
                robot.display(q_)
                if is_record and trial % trial_skip == 1:
                    images.append(robot.viewer.get_screenshot()[:, :, [2, 1, 0]])
            q0 = q_.copy()
        # save results
        if "regulation" in task:
            # success or not at the end
            res = 100.0 * success
        elif "tracking" in task:
            # mean of error norm during tracking
            if is_record:
                res = np.linalg.norm(np.array(errors)[int(tmax / dt):], axis=1).mean()
            else:
                res = np.linalg.norm(err)
        pbar.set_postfix({"trial": trial, "method": gradient + "-" + constraint, "result": res})
        if is_debug:
            print("result of {}\n\ttarget: {}\n\tfinal error: {}\n\tresult: {}".format(sdir, pin.SE3ToXYZQUATtuple(oMdes), err.tolist(), res))
            print("\n\tjoints: {}".format(q_.flatten().tolist()))
        if len(images):
            fig = plt.figure()
            fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            fig.gca().set_axis_off()
            images = [[plt.imshow(img, animated=True)] for img in images]
            ani = animation.ArtistAnimation(fig, images, interval=dt * frame_skip * 1000, blit=True)
            ani.save(sdir + "video.mp4")
            del ani
            del images
            plt.clf()
            plt.close()
        if len(times):
            times = pd.DataFrame(times, columns=["time", "iter"])
            times.to_csv(sdir + "time.csv", index=False)
            del times
        if len(successes):
            successes = pd.DataFrame(successes, columns=["success"])
            successes.to_csv(sdir + "success.csv", index=False)
            del successes
        if len(trajs):
            trajs = pd.DataFrame(trajs, columns=["tx", "ty", "tz", "qx", "qy", "qz", "qw"]*2)
            trajs.to_csv(sdir + "traj.csv", index=False)
            del trajs
        if len(errors):
            errors = pd.DataFrame(errors, columns=["tx", "ty", "tz", "rx", "ry", "rz"])
            errors.to_csv(sdir + "error.csv", index=False)
            del errors
        if len(qs):
            qs = pd.DataFrame(qs, columns=["j" + str(i) for i in range(len(qs[0]))])
            qs.to_csv(sdir + "q.csv", index=False)
            del qs

######################################################
