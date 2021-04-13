#DISPNET=checkpoints/r18_rectified_nyu/dispnet_model_best.pth.tar

#DATA_ROOT=/media/bjw/Disk/Dataset/nyu_test
#RESULTS_DIR=results/nyu_test/

DATA_ROOT=/home/matthias/data/tum_test_2/color/
RESULTS_DIR=/home/matthias/data/tum_test_2/results/
DISPNET=../models/r18_rectified_nyu/dispnet_model_best.pth.tar

#  test and visualize
python ../quick_test_disp.py --resnet-layers 18 --img-height 480 --img-width 640 \
--pretrained-dispnet $DISPNET --dataset-dir $DATA_ROOT --output-dir $RESULTS_DIR

