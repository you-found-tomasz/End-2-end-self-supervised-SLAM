import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from numpy.lib.function_base import diff 
import pandas
import os.path as osp


path = "/media/hamza/DATA/Code/debug_folder/desk_stride0/*/*.csv"
# rgbd_dataset_freiburg2_xyz_15dil_freeze_0

files = glob.glob(path)


def read_results(csv_path):
    df = pandas.read_csv(csv_path)
    res_dict = dict()
    for col in df.columns:
        res_dict[col] = df[col].to_numpy().reshape(-1)
    return res_dict

results = dict()
for file in files:
    results[file] = read_results(file)

def extract_dil(filename):
    name = osp.basename(osp.dirname(filename))
    dil = [i for i in name.split("_") if "dil" in i][0][:-3]
    return dil

def plot_loss(results, rel, loss, measurement):
    pose_error = []
    error_init = []
    error_fin = []
    error_imp = []
    exp_dil = []

    keys = sorted(results.keys())

    for file in keys:
        pose_error.append(results[file][measurement][0])
        init_error = results[file][loss][0]
        final_error = results[file][loss][-1]
        difference = 1e5 * abs(init_error-final_error)
        error_init.append(init_error)
        error_fin.append(final_error)
        error_imp.append(difference)        
        exp_dil.append(extract_dil(file))

        plt.scatter(pose_error[-1],  final_error, alpha=0.5, label=exp_dil[-1], s=difference)
        plt.annotate(exp_dil[-1], (pose_error[-1],  final_error))

    plt.xlabel("Planar Movement (m)")
    plt.ylabel("Abs Difference | L1 loss")
    # plt.legend()
    plt.title("Loss ({}) Behaviour vs Relative Pose Difference".format("improvement" if rel else "Absolute"))
    return True

plot_loss(results = results, rel = False, loss="val_abs_diff", measurement="xy")

plt.show()
print("asd")