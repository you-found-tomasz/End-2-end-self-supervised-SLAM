# path="/media/hamza/DATA/Data/tum"
path="/cluster/project/infk/courses/3d_vision_21/group_25/TUM_original"
debug_path="/cluster/project/infk/courses/3d_vision_21/group_25/debug_plozza/slam"
stride=101
batch=2
max_batches=10
seq_dilation=100
freeze="n"
scale=1
sequence="rgbd_dataset_freiburg2_xyz"
odometry="gradicp"
train_odometry="slam"
lr="5e-06"
epochs=301

#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_10dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation 10 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs ${epochs} --projection_mode previous --seq_stride 11 --learning_rate ${lr} --freeze ${freeze}
#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_15dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation 15 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs ${epochs} --projection_mode previous --seq_stride 16 --learning_rate ${lr} --freeze ${freeze}
#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_25dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation 25 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs ${epochs} --projection_mode previous --seq_stride 26 --learning_rate ${lr} --freeze ${freeze}
#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_50dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation 50 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs ${epochs} --projection_mode previous --seq_stride 51 --learning_rate ${lr} --freeze ${freeze}
#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_75dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation 75 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs ${epochs} --projection_mode previous --seq_stride 76 --learning_rate ${lr}--freeze ${freeze}
#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_100dil_freeze_${stride} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation 100 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs ${epochs} --projection_mode previous --seq_stride 101 --learning_rate ${lr} --freeze ${freeze}

#bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_batch10_slam_test --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation 100 --max_num_batches ${max_batches} --seq_start 1 --seq_end 2002 --num_epochs ${epochs} --projection_mode previous --seq_stride 101 --learning_rate ${lr} --freeze ${freeze}


sequence="rgbd_dataset_freiburg2_pioneer_slam"
dilation=19
stride=20
lr="5e-06"
batch=2
max_batches=10
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_batch10_slam_test_${dilation} --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation ${dilation} --max_num_batches ${max_batches} --seq_start 396 --seq_end 771 --num_epochs ${epochs} --projection_mode previous --seq_stride 101 --learning_rate ${lr} --freeze ${freeze}


sequence="rgbd_dataset_freiburg2_pioneer_slam"
dilation=19
stride=20
lr="5e-06"
batch=4
max_batches=5
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_batch10_slam_test_${dilation}_2 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation ${dilation} --max_num_batches ${max_batches} --seq_start 396 --seq_end 771 --num_epochs ${epochs} --projection_mode previous --seq_stride 101 --learning_rate ${lr} --freeze ${freeze}

sequence="rgbd_dataset_freiburg2_pioneer_slam"
dilation=29
stride=30
lr="1e-05"
batch=4
max_batches=5
bsub -R "rusage[mem=8096, ngpus_excl_p=1]" python3 pointfusion_scsfm.py --odometry ${odometry} --train_odometry ${train_odometry} --dataset tum --dataset_path $path --sequences ${sequence} --seq_length 2 --batch_size ${batch} --debug_path ${debug_path} --model_name ${sequence}_batch10_slam_test_${dilation}_3 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0 --log_freq 50 --max_scale ${scale} --seq_dilation ${dilation} --max_num_batches ${max_batches} --seq_start 396 --seq_end 771 --num_epochs ${epochs} --projection_mode previous --seq_stride 101 --learning_rate ${lr} --freeze ${freeze}
