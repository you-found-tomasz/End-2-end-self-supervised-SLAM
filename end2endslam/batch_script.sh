path="/media/hamza/DATA/Data/tum"
debug_path="../../debug_folder"
stride=4
seq_dilation=100

for seq in ${path}/*/ ; do
    sequence=$(basename $seq)
    echo $sequence
    python3 pointfusion_scsfm.py --dataset tum --dataset_path $path --odometry gt --sequences ${sequence} --seq_length 10 --batch_size 5 --debug_path ${debug_path} --model_name ${sequence}_scale4 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 4 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous --seq_stride $stride
done




#python pointfusion_scsfm.py --dataset tum --dataset_path ../sample_data/dataset_TUM --odometry gt --sequences rgbd_dataset_freiburg2_xyz --seq_length 10 --batch_size 5 --debug_path ../debug_folder --model_name 002_xyz_short_previous_scale4 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 30 --max_scale 4 --seq_dilation 100 --max_num_batches 2 --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous
