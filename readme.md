# End-2-end-self-supervised-SLAM
Project for ETH 3D vision course 2021. 
<br/>
Matthias Brucker, Hamza Javed, Davide Plozza, Tomasz Zaluska, Jeremaine Siegenthaler

Reuses code from
https://github.com/gradslam/gradslam and https://github.com/JiawangBian/SC-SfMLearner-Release.

# Reproduce experiments

## Cluster Access (optional)
Login to cluster (use your eth login, use your password for mail)
```shell
ssh <username>@login.leonhard.ethz.ch
```
Load module to have right compiler and Python version
```shell
module load gcc/6.3.0 python_gpu/3.8.5
```
Then continue with "Setup".
Good tutorials for further info:
https://scicomp.ethz.ch/wiki/Workshops

## Setup
Clone repository
```shell
git clone https://github.com/aquamin9/End-2-end-self-supervised-SLAM.git
```
Create virtual environment
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
Execute in the main (End-2-end-self-supervised-SLAM) folder 
```shell
"pip install -e ." 
```

Download TUM sequences:
bash script for downloading TUM, execute in sample_data/dataset_TUM folder 
```shell
bash raw_data_downloader_TUM_freiburg1.sh 
bash raw_data_downloader_TUM_freiburg2.sh 
```
NYU download (not necessary to reproduce results from report): manually (store under sample_data/dataset_NYU)
https://onedrive.live.com/?authkey=%21AKVvEAT14TgiFEE&cid=36712431A95E7A25&id=36712431A95E7A25%212551&parId=36712431A95E7A25%21472&action=locate



## Run Experiments locally
Commands for reproducing the main experiments with GT poses (run from end2endslam folder):
```shell
# xyz
python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name xyzs_010_previous_scale1_dil100 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 1 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous
#slam
python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg2_pioneer_slam --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name slams_010_previous_scale1_dil9 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 9 --max_num_batches 1 --seq_start 369 --seq_end 571 --num_epochs 300 --projection_mode previous
# desk
python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg1_desk --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name desks_010_previous_scale1_dil6 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 6 --max_num_batches 1 --seq_start 353 --seq_end 495 --num_epochs 300 --projection_mode previous
# rpy
python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg1_rpy --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name rpys_010_previous_scale1_dil6 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 6 --max_num_batches 1 --seq_start 310 --seq_end 492 --num_epochs 300 --projection_mode previous
```
To reproduce the results on frame skipping and ablation studies with GT poses, use the commands provided in "commands_frame_skipping", "commands_ablation_loss_weights.sh", and "commands_ablation_optimization.sh".
The arguments to reproduce results with SLAM poses with pointfusion_scsfm.py are provided in end2endslam/slam_poses_experiments.txt. 

## Run Experiments on Cluster
Run multiple trainings on GPU via bash files (in end2endslam folder):
```shell
bash commands_main_results.sh
bash commands_frame_skipping.sh
bash commands_ablation_optimization.sh
bash commands_ablation_loss_weights.sh
```
Example: Single command on CPU, run from end2endslam folder (with the -Is option you see interactively, good to check whether it works)

```shell
bsub -R -Is "rusage[mem=8192]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name xyzs_010_previous_scale1_dil100 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 1 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"
```

Example: Single command on GPU, run from end2endslam folder
```shell
bsub -R -Is "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name xyzs_010_previous_scale1_dil100 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 1 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"
```

## Visualize Results

Generate the plots from the report with  this  Jupyter Notebook: end2endslam/visualizations.ipynb (might need to change filepaths) 


