python pointfusion_scsfm.py^
 --dataset tum --dataset_path "../sample_data/dataset_TUM/dataset_TUM_desk"^
 --debug_path "./debug/" --model_name tum_desk_subset_test^
 --odometry gt --seq_length 10 --batch_size 5 --seq_start 396 --seq_end 488 --seq_stride 12 --seq_dilation 3^
 --loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0^
 --log_freq 10 --max_scale 1