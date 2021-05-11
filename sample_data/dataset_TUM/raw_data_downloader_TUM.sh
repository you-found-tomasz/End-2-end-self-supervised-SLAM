#!/bin/bash

#freiburg1
files=(rgbd_dataset_freiburg1_rpy.tgz
rgbd_dataset_freiburg1_xyz.tgz
)


for i in ${files[@]}; do
                fullname=$i
	echo "Downloading: "$fullname
        wget 'https://vision.in.tum.de/rgbd/dataset/freiburg1/'$fullname
        tar -xvzf $fullname
        rm $fullname
done

#freiburg2
files=(freiburg2_pioneer_slam.tgz
freiburg2_pioneer_slam2.tgz
)

for i in ${files[@]}; do
                fullname=$i
	echo "Downloading: "$fullname
        wget 'https://vision.in.tum.de/rgbd/dataset/freiburg2/'$fullname
        tar -xvzf $fullname
        rm $fullname
done
