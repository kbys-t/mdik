# coding:utf-8

import argparse
import numpy as np

######################################################
# init parameters

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="regulation")
parser.add_argument('--name', default="talos")
parser.add_argument('--n_resume', type=int, default=0)
args = parser.parse_args()
task, name, n_resume = args.task, args.name, args.n_resume

if name == "talos":
    id_ee = 23  # 10 dof
    low_rot  = np.array([-0.3, -0.3 - 0.5 * np.pi, -0.3 - 0.5 * np.pi])
    high_rot = np.array([ 0.3,  0.3 - 0.5 * np.pi,  0.3 - 0.5 * np.pi])
    low_trans  = np.array([0.3, -0.2, 0.8])
    high_trans = np.array([0.9,  0.4, 1.4])
elif name == "tiago":
    id_ee = 8   # 8 dof
    low_rot  = np.array([-0.25, -0.25 - 0.5 * np.pi, -0.25 - 0.5 * np.pi])
    high_rot = np.array([ 0.25,  0.25 - 0.5 * np.pi,  0.25 - 0.5 * np.pi])
    low_trans  = np.array([0.3, -0.3, 0.5])
    high_trans = np.array([0.8,  0.2, 1.0])
elif name == "ur5_limited":
    id_ee = 6   # 6 dof
    low_rot  = np.array([-0.2, -0.2 - 0.5 * np.pi, -0.2 - 0.5 * np.pi])
    high_rot = np.array([ 0.2,  0.2 - 0.5 * np.pi,  0.2 - 0.5 * np.pi])
    low_trans  = np.array([0.4, -0.2, 0.2])
    high_trans = np.array([0.8,  0.2, 0.6])

gradients = []
gradients += ["JT"]
gradients += ["LM"]

constraints = []
constraints += ["mirror"]
constraints += ["parameterized"]
constraints += ["projected"]

solvers = []
solvers += ["cvxopt"]
solvers += ["ecos"]
solvers += ["osqp"]
solvers += ["quadprog"]

if "regulation" in task:
    n_trial = 2000
elif "tracking" in task:
    n_trial = 500

dt = 5e-3
# dt_stop = dt
dt_stop = 2.0e-3
tmax = 2.5# i.e. 500 step
if "regulation" in task:
    tperiod = 0.0
elif "tracking" in task:
    tperiod = 5.0
    freq_max = 0.5
is_timelimit = True
frame_skip = 4
trial_skip = 50

eps_err = 1e-10
max_iter = 1000

damp = 1e-3
margin = 1e-2
alpha = 1.0
gain = 2.0 * np.log((1.0 - margin) / margin)

is_update_jacobian = False
is_nesterov = True

weight = np.array([1.0]*3 + [1.0]*3)
weight *= len(weight) / weight.sum()

is_debug = False
is_record = True
