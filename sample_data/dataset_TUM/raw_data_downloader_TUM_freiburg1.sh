#!/bin/bash

#freiburg1
files=(rgbd_dataset_freiburg1_desk.tgz
rgbd_dataset_freiburg1_rpy.tgz
)

for i in ${files[@]}; do
                fullname=$i
	echo "Downloading: "$fullname
        wget 'https://vision.in.tum.de/rgbd/dataset/freiburg1/'$fullname
        tar -xvzf $fullname
        rm $fullname
done
