# path="/media/hamza/DATA/Data/tum"
path="../sample_data/dataset_TUM"
debug_path="../debug_folder/stride0"
stride=0
batch=2
max_batches=1
seq_dilation=100
freeze="n"
scale=1
sequence="rgbd_dataset_freiburg2_xyz"

odometry = "gt"
train_odometry = "gt"

bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 10 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_10dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 75 --max_scale ${scale} --seq_dilation 10 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous --seq_stride ${stride} --freeze ${freeze}
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 10 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_15dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 75 --max_scale ${scale} --seq_dilation 15 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous --seq_stride ${stride} --freeze ${freeze}
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 10 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_25dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 75 --max_scale ${scale} --seq_dilation 25 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous --seq_stride ${stride} --freeze ${freeze}
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 10 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_50dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 75 --max_scale ${scale} --seq_dilation 50 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous --seq_stride ${stride} --freeze ${freeze}
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 10 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_75dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 75 --max_scale ${scale} --seq_dilation 75 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous --seq_stride ${stride} --freeze ${freeze}
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 10 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_100dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 75 --max_scale ${scale} --seq_dilation 100 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs 300 --projection_mode previous --seq_stride ${stride} --freeze ${freeze}
