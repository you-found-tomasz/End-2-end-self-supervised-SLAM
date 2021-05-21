#!/bin/bash

#freiburg1
files=(rgbd_dataset_freiburg2_xyz.tgz
)


for i in ${files[@]}; do
                fullname=$i
	echo "Downloading: "$fullname
        wget 'https://vision.in.tum.de/rgbd/dataset/freiburg2/'$fullname
        tar -xvzf $fullname
        rm $fullname
done
