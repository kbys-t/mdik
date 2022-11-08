# coding:utf-8

import os
import time

import numpy as np
import pandas as pd

import pinocchio as pin

from simulator.base import Atlas, Talos
from controller.controller import WalkController


######################################################
Environment = Atlas
PHYSICS = {"numSolverIterations": 50, "numSubSteps": 0} # default
DT = 1.0/(240.0 * 2.0)  # 240Hz is the default of pybullet
# DT = 1.0/(240.0 * 3.0)  # 240Hz is the default of pybullet
TIME_LIMIT = 0.36   # 0.75ms ~= osqp computation for 480 Hz and 0.5 ms for 720 Hz
VERBOSE = False

REALTIMES = [False, True]
METHODS = ["osqp", "mdik"]
N_TRIAL = 100
SAVE_DIR = "./result/"
os.makedirs(SAVE_DIR, exist_ok=True)

######################################################
def summary(results):
    with open(SAVE_DIR + "summary.txt", "w") as f_:
        for realtime in REALTIMES:
            for method in METHODS:
                df = pd.DataFrame(results[(results.method == method) & (results.realtime == realtime)]).describe().transpose()
                print(realtime, method)
                print(realtime, method, file=f_)
                print(df)
                print(df, file=f_)

######################################################
if N_TRIAL:
    np.seterr(all="warn" if VERBOSE else "ignore")
    env = Environment(dt=DT, realtime=False, verbose=VERBOSE, **PHYSICS)
    results = pd.DataFrame(columns=["realtime", "method", "n_trial", "time", "step"])

    for n_trial in range(N_TRIAL):
        for realtime in REALTIMES:
            for method in METHODS:
                print("##########\n {}-th trial: realtime={}, method={}\n##########".format(n_trial+1, realtime, method))
                env.reset(realtime)
                ctrl = WalkController(env.urdf_path, env.end_effectors, env.joint_names, env.grav, env.dt,
                                    env.root_pose, env.joint_pos,
                                    kin_params={"ik": method, "time_limit": TIME_LIMIT},
                                    verbose=VERBOSE)
                delta = env.root_pose[:2].copy()
                t_list = []
                n_step = 0
                is_start = False

                t_ = 0.0
                while t_ < 100.0:
                    env.observe()
                    ts = time.time()
                    qb_ref, qj_ref = ctrl(env.root_pose, env.root_vel, env.joint_pos, env.joint_vel, env.ee_contact)
                    tc = time.time() - ts
                    env.command(qj_ref)
                    if not is_start and t_ > ctrl.gen.T_ini:
                        is_start = True
                        ctrl.start()
                    if is_start and not ctrl.gen.is_start:
                        # time.sleep(ctrl.gen.T_ini)
                        print("end walk!")
                        n_step = ctrl.gen.cnt_footstep
                        if env.realtime:
                            for _ in range(int(1.0 / env.dt)):
                                env.observe()
                        else:
                            time.sleep(1.0)
                            env.observe()
                        delta = np.linalg.norm(delta - env.root_pose[:2])
                        break
                    if ctrl.fail():
                        print("fail walk...")
                        n_step = ctrl.gen.cnt_footstep - 1
                        delta = np.nan
                        break
                    t_list.append(tc)
                    time.sleep(max(env.dt - (time.time() - ts), 0.0))
                    t_ += DT

                result = {"realtime": realtime, "method": method, "n_trial": n_trial, "time": np.mean(t_list), "step": float(n_step), "delta": delta}
                results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
                results.to_csv(SAVE_DIR + "details.csv", index=False)
                summary(results)
    env.close()

results = pd.read_csv(SAVE_DIR + "details.csv")
print(results)
summary(results)
