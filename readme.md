# End-2-end-self-supervised-SLAM
Project for ETH 3D vision course

# Cluster

login to cluster
```shell
ssh zaluskat@login.leonhard.ethz.ch
```
load module to have right compiler and python version
```shell
module load gcc/6.3.0 python_gpu/3.8.5
```
clone repository
```shell
git clone https://github.com/aquamin9/End-2-end-self-supervised-SLAM.git
```
create virtual environment
```shell
python -m venv --system-site-packages 3dvision
```
```shell
source 3dvision/bin/activate
```
Install gradslam by navigating to "gradslam/" and executing 
```shell
"pip install ."
```
Execute in the main folder 
```shell
"pip install -e ." 
```

Run on CPU (with the -Is option you see interectively, good to check whether it works)
```shell
bsub -R "rusage[mem=8192]" -Is "python pointfusion_scsfm_brucker.py --dataset tum --dataset_path "../../sample_data/dataset_TUM/" --odometry icp --loss depth_consistency"
```

Run on GPU
```shell
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm_brucker.py --dataset tum --dataset_path "../../sample_data/dataset_TUM/" --odometry icp --loss depth_consistency"
```

Good tutorials for further info:
https://scicomp.ethz.ch/wiki/Workshops



# Downloaders

TUM:
bash script for downloading TUM (to add more download files update the raw_data_downloader_TUM.sh file)
```shell
bash sample_data/dataset_TUM/raw_data_downloader_TUM.sh 
```
Running pointfusion file with the correct relative string to the data
```shell
python pointfusion_scsfm_brucker.py --dataset tum --dataset_path "../../sample_data/dataset_TUM/" --odometry icp --loss "depth_consistency
```

NYU:
downloader doesnt work, need download by hand:
https://onedrive.live.com/?authkey=%21AKVvEAT14TgiFEE&cid=36712431A95E7A25&id=36712431A95E7A25%212551&parId=36712431A95E7A25%21472&action=locate


# Setup

Install gradslam by navigating to "gradslam/" and executing "pip install ."

Execute "pip install -e ." in the main folder


# Folder structure

The main code of our project should be in the "end2endslam" directory
"gradslam" contains a copy of the gradslam repository.
"perception/monodepth2" contains the modified monodepth2 repository



