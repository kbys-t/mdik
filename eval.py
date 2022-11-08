# coding:utf-8

import os
from tqdm import tqdm
from tqdm.contrib import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

######################################################
def find_dataset(dirs, file_name):
    summary = []
    for dir in dirs:
        if os.path.isdir(dir):
            files = [dir + f_ for f_ in os.listdir(dir) if os.path.isfile(dir + f_) and file_name in f_ and not f_.startswith(".")]
            summary += files + find_dataset([dir + d_ + "/" for d_ in os.listdir(dir) if os.path.isdir(dir + d_)], file_name)
    return summary

######################################################
def get_summary(files, func):
    columns = ["task", "robot", "method", "trial"] + [func.__name__]
    summary = {key: [] for key in columns}
    for file in tqdm(files):
        for key, val in zip(columns[0:-1], file.split("/")[2:-1]):
            summary[key].append(val)
        dat = pd.read_csv(file)
        summary[columns[-1]].append(func(dat))
    return pd.DataFrame(summary)

######################################################
def get_valid(dat, judge, name):
    flag = pd.DataFrame()
    flag["robot"] = dat["robot"]
    flag["trial"] = dat["trial"]
    flag["valid"] = True
    for robot, trial in itertools.product(flag["robot"].unique(), flag["trial"].unique()):
        mask = (flag["robot"].values == robot) * (flag["trial"].values == trial)
        flag.loc[mask, "valid"] = judge(dat.loc[mask, name])
    return flag

######################################################
def save_data(dat, flag, name, hue_order):
    ylabel = dat.columns[-1]
    dat_valid = dat.loc[flag["valid"] == True]
    plt.clf()
    sns.boxenplot(x="robot", y=ylabel, hue="method", hue_order=hue_order, data=dat_valid, showfliers=False)
    plt.legend(bbox_to_anchor=(0.5, 1.175), loc="upper center", frameon=True, ncol=5)
    plt.savefig(name + "_" + ylabel + ".pdf")
    with open(name + "_" + ylabel + ".txt", "w") as f_:
        for robot, method in itertools.product(dat_valid["robot"].unique(), dat_valid["method"].unique()):
            print(robot, method, file=f_)
            df = pd.DataFrame(dat_valid[(dat_valid["robot"] == robot) & (dat_valid["method"] == method)][ylabel])
            print(df.describe().transpose(), file=f_)

######################################################
### configurations of plot
sns.set(context = "paper", style = "white", palette = "Set2", font = "Arial", font_scale = 1.8, rc = {"lines.linewidth": 1.0, "pdf.fonttype": 42})
sns.set_palette("Set2", 8, 1)
colors = sns.color_palette(n_colors=10)
markers = ["o", "s", "d", "*", "+", "x", "v", "^", "<", ">"]
fig = plt.figure(figsize=(8, 6))

# hue_order = None
hue_order = ["OSQP", "LM", "MD", "AMD", "SAMD"]

### for regulation
dir = "./result/regulation/"

# load success
files = sorted(find_dataset([dir], "success"))
def success_rate(dat):
    return dat.values[-1, -1]
summary = get_summary(files, success_rate)

# make valid flag
def judge_success(dat):
    return dat.sum() > 0
flag = get_valid(summary, judge_success, summary.columns[-1])
print("length of valid data: {}".format(len(flag[flag["valid"] == True])))

# record success
save_data(summary, flag, dir[:-1], hue_order)

# mean error
files = sorted(find_dataset([dir], "error"))
def mean_error(dat):
    return np.linalg.norm(dat.values, axis=1).mean()
save_data(get_summary(files, mean_error), flag, dir[:-1], hue_order)

# joint smoothness
files = sorted(find_dataset([dir], "q"))
def joint_smoothness(dat):
    return np.linalg.norm(np.diff(dat.values, axis=0), axis=1).mean()
save_data(get_summary(files, joint_smoothness), flag, dir[:-1], hue_order)

# number of iterations
files = sorted(find_dataset([dir], "time"))
def iteration(dat):
    return dat.loc[1:21, "iter"].values.mean()
save_data(get_summary(files, iteration), flag, dir[:-1], hue_order)

# computational time
def milliseconds(dat):
    return dat.loc[1:21, "time"].values.mean() * 1e+3
save_data(get_summary(files, milliseconds), flag, dir[:-1], hue_order)

### for tracking
dir = "./result/tracking/"

# load mean error
files = sorted(find_dataset([dir], "error"))
def mean_error(dat):
    return np.linalg.norm(dat.values[500:], axis=1).mean()
summary = get_summary(files, mean_error)

# make valid flag
def judge_threshold(dat, threshold=0.1):
    return all(dat < threshold)
flag = get_valid(summary, judge_threshold, summary.columns[-1])
print("length of valid data: {}".format(len(flag[flag["valid"] == True])))

# record mean error
save_data(summary, flag, dir[:-1], hue_order)

# worst error
files = sorted(find_dataset([dir], "error"))
def worst_error(dat):
    return np.linalg.norm(dat.values[500:], axis=1).max()
save_data(get_summary(files, worst_error), flag, dir[:-1], hue_order)

# joint smoothness
files = sorted(find_dataset([dir], "q"))
def joint_smoothness(dat):
    return np.linalg.norm(np.diff(dat.values[500:], axis=0), axis=1).max()
save_data(get_summary(files, joint_smoothness), flag, dir[:-1], hue_order)

# number of iterations
files = sorted(find_dataset([dir], "time"))
def iteration(dat):
    return dat.loc[500:, "iter"].values.mean()
save_data(get_summary(files, iteration), flag, dir[:-1], hue_order)

# computational time
def milliseconds(dat):
    return dat.loc[500:, "time"].values.mean()
save_data(get_summary(files, milliseconds), flag, dir[:-1], hue_order)
