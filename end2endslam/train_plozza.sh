#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 003_xyz_icp__previous_geom0.1 --loss_photo_factor 1 --loss_geom_factor 0.1 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 1000 --projection_mode previous"

#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 004_xyz_icp_previous_geom0.2 --loss_photo_factor 1 --loss_geom_factor 0.2 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 1000 --projection_mode previous"

#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 005_xyz_icp_previous_geom0.3 --loss_photo_factor 1 --loss_geom_factor 0.3 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 1000 --projection_mode previous"

#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 006_xyz_icp_previous_geom0.2_dil25 --loss_photo_factor 1 --loss_geom_factor 0.3 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 25 --max_num_batches 5 --seq_start 1 --seq_end 2002 --num_epochs 1000 --projection_mode previous"

#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 007_xyz_icp_previous_geom0.2_dil50 --loss_photo_factor 1 --loss_geom_factor 0.3 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 50 --max_num_batches 5 --seq_start 1 --seq_end 2002 --num_epochs 1000 --projection_mode previous"

#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 100_xyz_gt_previous_geom0.5 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 1000 --projection_mode previous"

#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 101_xyz_gt_previous_geom0.2 --loss_photo_factor 1 --loss_geom_factor 0.2 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 1000 --projection_mode previous"

bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 102_xyz_gt_previous_geom0 --loss_photo_factor 1 --loss_geom_factor 0 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 5 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"

bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 009_xyz_icp_previous_geom0.4 --loss_photo_factor 1 --loss_geom_factor 0.4 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 5 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"

bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 010_xyz_gt_previous_geom0.4 --loss_photo_factor 1 --loss_geom_factor 0.4 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 50 --max_num_batches 5 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"
