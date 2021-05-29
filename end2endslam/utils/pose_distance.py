import numpy as np
import matplotlib.pyplot as plt

path2text = "/media/hamza/DATA/Data/tum/rgbd_dataset_freiburg2_xyz/groundtruth.txt"

file = np.loadtxt(path2text)

trans = file[:, 1:3]
second = trans[1:]
first = trans[:-1]

diff = second - first

plt.plot(np.linalg.norm(diff, axis = 1))
print("asd")