# End-2-end-self-supervised-SLAM
Project for ETH 3D vision course

# Common Cluster

login to cluster (of course exchange zaluskat for your eth login, use your password for mail)
```shell
ssh zaluskat@login.leonhard.ethz.ch
```

load module to have right compiler and python version
```shell
module load gcc/6.3.0 python_gpu/3.8.5
```

```shell
 cd /cluster/project/infk/courses/3d_vision_21/group_25/End-2-end-self-supervised-SLAM/
```

```shell
source 3dvision/bin/activate
```

inside pointfusion folder

```shell
bsub -Is -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm_brucker2.py --dataset tum --dataset_path "../../sample_data/dataset_TUM/" --odometry icp --loss depth_consistency"
```


# Cluster

login to cluster (of course exchange zaluskat for your eth login, use your password for mail)
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

TUM:
bash script for downloading TUM execute in sample_data/data_TUM folder (to add more download files update the raw_data_downloader_TUM.sh file)
```shell
bash raw_data_downloader_TUM.sh 
```

Run on CPU, need to navigate to the pointfusion folder (with the -Is option you see interectively, good to check whether it works)

```shell
running pointfusion from the pointfusion folder:
bsub -R "rusage[mem=8192]" -Is "python pointfusion_scsfm_brucker.py --dataset tum --dataset_path "../../sample_data/dataset_TUM/" --odometry icp --loss depth_consistency"

running pointfusion from the main folder:
bsub -R "rusage[mem=8192]" -Is "python pointfusion_scsfm_brucker.py --dataset tum --dataset_path "/sample_data/dataset_TUM/" --odometry icp --loss depth_consistency"
```

Run on GPU, need to navigate to the pointfusion folder
```shell
bsub -Is -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm_brucker.py --dataset tum --dataset_path "../../sample_data/dataset_TUM/" --odometry icp --loss depth_consistency"
```

Good tutorials for further info:
https://scicomp.ethz.ch/wiki/Workshops



# Downloaders

TUM:
bash script for downloading TUM, execute in the sample_data/data_TUM folder folder (to add more download files update the raw_data_downloader_TUM.sh file)
```shell
bash raw_data_downloader_TUM.sh 
```
Running pointfusion file with the correct relative string to the data
```shell
python pointfusion_scsfm_brucker.py --dataset tum --dataset_path "../../sample_data/dataset_TUM/" --odometry icp --loss "depth_consistency
```

NYU:
downloader doesnt work, need download by hand:
https://onedrive.live.com/?authkey=%21AKVvEAT14TgiFEE&cid=36712431A95E7A25&id=36712431A95E7A25%212551&parId=36712431A95E7A25%21472&action=locate

# Copying files between cluster and local computer (unix)

All the following examples need to be run on your local computer
1. Upload a file from your workstation to Euler (home directory)
```shell
scp file username@euler.ethz.ch:
```
2. Download a file from Euler to your workstation (current directory)
```shell
scp username@euler.ethz.ch:file .
```
3. Copy a whole directory
```shell
scp -r localdir username@euler.ethz.ch:remotedir
```




# Setup

Install gradslam by navigating to "gradslam/" and executing "pip install ."

Execute "pip install -e ." in the main folder


# Folder structure

The main code of our project should be in the "end2endslam" directory
"gradslam" contains a copy of the gradslam repository.
"perception/monodepth2" contains the modified monodepth2 repository



