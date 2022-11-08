# coding:utf-8

import os

######################################################
def find_dirs(dirs, depth):
    rtv = []
    if depth:
        for dir in dirs:
            if os.path.isdir(dir):
                ds = [dir + d_ + "/" for d_ in os.listdir(dir) if os.path.isdir(dir + d_)]
                rtv += ds + find_dirs(ds, depth-1)
    return rtv

######################################################
def rename_dirs(dir):
    dir_list = dir.split("/")
    old_name = dir_list[-2]
    new_name = old_name
    if "osqp" in old_name:
        new_name = "OSQP"
    elif "LM" in old_name:
        new_name = "LM"
    elif "mirror" in old_name:
        new_name = "MD"
        if "accelerated" in old_name:
            new_name = "A" + new_name
            if "smooth" in old_name:
                new_name = "S" + new_name
    if old_name != new_name:
        dir_list[-2] = new_name
        print("change: {} \n->\t {}".format(dir, "/".join(dir_list)))
        os.rename(dir, "/".join(dir_list))

######################################################
parent_dir = "./result/"
depth = 3

[rename_dirs(dir) for dir in find_dirs([parent_dir], depth)]
