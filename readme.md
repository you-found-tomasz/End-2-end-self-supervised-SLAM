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
bash script for downloading TUM execute in sample_data/dataset_TUM folder (to add more download files update the raw_data_downloader_TUM.sh file)
```shell
bash raw_data_downloader_TUM_freiburg2.sh 
```
Run multiple trainings on GPU via example_commands.sh bash file:
```shell
bash example_commands.sh
```
Run on CPU, need to navigate to the end2endslam folder (with the -Is option you see interactively, good to check whether it works)

```shell
bsub -R "rusage[mem=8192]" -Is "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 001_xyz_short_previous_scale1 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"
```

Run on GPU, need to navigate to the end2end folder
```shell
bsub -Is -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 001_xyz_short_previous_scale1 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"
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

# New version with losses from SC-Sfm-Learner

Run command:
```shell
python pointfusion_scsfm.py --dataset tum --dataset_path "/home/matthias/git/End-2-end-self-supervised-SLAM/sample_data/dataset_TUM" --odometry icp --debug_path "/home/matthias/data/3dv_debug/" --model_name test1 
--loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0.1 --loss_gt_factor 0.1 --log_freq 10
```


# Setup

Install gradslam by navigating to "gradslam/" and executing "pip install ."

Execute "pip install -e ." in the main folder


# Folder structure

The main code of our project should be in the "end2endslam" directory
"gradslam" contains a copy of the gradslam repository.
"perception/monodepth2" contains the modified monodepth2 repository



