# coding:utf-8

import argparse
import numpy as np
import itertools

######################################################
# init parameters

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="regulation")
parser.add_argument('--name', default="talos")
parser.add_argument('--n_resume', type=int, default=0)
args = parser.parse_args()
task, name, n_resume = args.task, args.name, args.n_resume

# targets
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

# problems
if "regulation" in task:
    n_trial = 2000
elif "tracking" in task:
    n_trial = 500

dt = 5e-3
tmax = 2.5# i.e. 500 step
if "regulation" in task:
    tperiod = 0.0
elif "tracking" in task:
    tperiod = 5.0
    freq_max = 0.5
frame_skip = 4
trial_skip = 50

weight = np.array([1.0]*3 + [1.0]*3)
weight *= len(weight) / weight.sum()

# solvers
# gradients = []
# gradients += ["JT"]
# gradients += ["LM"]
#
# constraints = []
# constraints += ["mirror"]
# constraints += ["parameterized"]
# constraints += ["projected"]
#
# gc_solvers = list([gradient + "-" + constraint for gradient, constraint in itertools.product(gradients, constraints)])

gc_solvers = []
gc_solvers += ["LM-projected"]
gc_solvers += ["JT-mirror"]
gc_solvers += ["JT-mirror-accelerated"]
gc_solvers += ["JT-mirror-accelerated-smooth"]


qp_solvers = []
qp_solvers += ["osqp"]
# qp_solvers += ["qpoases"]

# parameters
damp = 1e-3
margin = 1e-2
step_size = 1.0
gain = 2.0 * np.log((1.0 - margin) / margin)

is_update_jacobian = False
accel = 5.0
gain_step_size = 2.0
smooth_reset = 0.5

# termination conditions
eps_err = 1e-10
max_iter = 1000

is_timelimit = True
# dt_stop = dt
# dt_stop = 2.5e-3
# dt_stop = 1e-3
dt_stop = 5e-4

# save flags
is_debug = False
is_record = True
