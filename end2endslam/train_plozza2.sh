#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 003_xyz_icp__previous_geom0.1 --loss_photo_factor 1 --loss_geom_factor 0.1 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 1 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 1000 --projection_mode previous"

bsub -R "rusage[mem=8096, ngpus_excl_p=1]" "python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry icp --train_odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path /cluster/project/infk/courses/3d_vision_21/group_25/debug_plozza --model_name 001_p_xyz_prev_orig --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale 1 --seq_dilation 100 --max_num_batches 5 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous"
