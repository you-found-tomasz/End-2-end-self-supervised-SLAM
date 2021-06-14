# xyz
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name xyzs_010_previous_scale1_dil100 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 1 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"

#slam
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg2_pioneer_slam --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name slams_010_previous_scale1_dil9 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 9 --max_num_batches 1 --seq_start 369 --seq_end 571 --num_epochs 300 --projection_mode previous"

# desk
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg1_desk --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name desks_010_previous_scale1_dil6 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 6 --max_num_batches 1 --seq_start 353 --seq_end 495 --num_epochs 300 --projection_mode previous"

# rpy
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg1_rpy --seq_length 10 --batch_size 2 --debug_path ../debug_folder --model_name rpys_010_previous_scale1_dil6 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 6 --max_num_batches 1 --seq_start 310 --seq_end 492 --num_epochs 300 --projection_mode previous"

