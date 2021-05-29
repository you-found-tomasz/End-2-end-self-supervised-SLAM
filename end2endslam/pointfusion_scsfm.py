from argparse import ArgumentParser, RawTextHelpFormatter

import torch
from torch.utils.data import DataLoader

import numpy as np
import csv
import os
import json
import cv2 as cv
#import open3d as o3d # TODO: remove
from gradslam.structures.rgbdimages import RGBDImages
from end2endslam.dataloader.tum import TUM
from end2endslam.dataloader.nyu import NYU
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from end2endslam.pointfusion_scsfm_utils import compute_scaling_coef, slam_step, compute_relative_pose_magnitudes

from end2endslam.scsfmwrapper import SCSfmWrapper
from losses.unified_loss import pred_loss_unified
from losses.gt_loss import compute_errors #validation
from losses.pose_loss import pose_loss_unified

from torch.optim import Adam,SGD
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl
import matplotlib.cm as cm
import imageio

"""r Example run commands:

(leave TUM folder structure unchanged):

runfile('/home/matthias/gitExample Args: 
--dataset tum --dataset_path "/home/matthias/git/End-2-end-self-supervised-SLAM/sample_data/dataset_TUM_desk" 
--debug_path "/home/matthias/data/3dv_debug/" --model_name tum_desk_subset_test
--odometry gt --seq_length 10 --batch_size 5 --seq_start 396 --seq_end 488 --seq_stride 12 --seq_dilation 3
--loss_photo_factor 1 --loss_geom_factor 0.5 --loss_smooth_factor 0.1 --loss_cons_factor 0 --loss_gt_factor 0
--log_freq 1 --max_scale 4

"""
# TODO: Use for Debug
USE_GT_DEPTH = False #also disables training
VISUALIZE_SLAM = False
EVAL_VALIDATION = False # slowing training down a bit, computes validation loss in eval mode

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["tum", "nyu", "nyu-regular"],
    help="Dataset to use. Supported options:\n"
    " tum = Iterative Closest Point\n"
    " nyu = rectified nyu dataset as provided by SfM-Learner"
    " nyu-regular = NOT SUPPORTET YET: regular nyu dataset",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the dataset directory",
)
parser.add_argument(
    "--odometry",
    type=str,
    default="gradicp",
    choices=["gt", "icp", "gradicp"],
    help="Odometry method used by slam. Supported options:\n"
    " gt = Ground Truth odometry\n"
    " icp = Iterative Closest Point\n"
    " gradicp (*default) = Differentiable Iterative Closest Point\n",
)

parser.add_argument(
    "--train_odometry",
    type=str,
    default="gt",
    choices=["gt", "slam"],
    help="Odometry method used by for training. Supported options:\n"
    " gt = Ground Truth odometry (default)\n"
    " slam = Poses from slam, which uses whichever method specified by --odometry\n"
)

parser.add_argument(
    "--sequences",
    type=str,
    default=None,
    help="Path to .txt file containing sequences. \n"
    "Single sequence name, filepath to sequences.txt or None"
)
parser.add_argument(
    "--debug_path",
    type=str,
    default=None,
    help="Path for debug visualizations"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="test1",
    help="Model name"
)
parser.add_argument(
    "--seq_length",
    type=int,
    default=10,
    help="SLAM Sequence Length"
)
parser.add_argument(
    "--seq_dilation",
    type=int,
    default=None,
    help="SLAM Sequence Dilation (n frames to skip)"
)
parser.add_argument(
    "--seq_stride",
    type=int,
    default=None,
    help="SLAM Sequence Stride"
)
parser.add_argument(
    "--seq_start",
    type=int,
    default=None,
    help="SLAM Sequence Start frame"
)
parser.add_argument(
    "--seq_end",
    type=int,
    default=None,
    help="SLAM Sequence End frame"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Batch size"
)
parser.add_argument(
    "--max_num_batches",
    type=int,
    default=10,
    help="Maximum number of batches to be run per epoch"
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=100,
    help="Number of epochs"
)
parser.add_argument(
    "--log_freq",
    type=int,
    default=10,
    help="Frequency for logging"
)
parser.add_argument(
    "--projection_mode",
    type=str,
    default="previous",
    choices=["previous", "first"],
    help="Pairwise or wrt. first image in sequence"
)
parser.add_argument(
    "--loss_photo_factor",
    type=float,
    default=0.3
)
parser.add_argument(
    "--loss_geom_factor",
    type=float,
    default=0.3
)
parser.add_argument(
    "--loss_smooth_factor",
    type=float,
    default=0.0
)
parser.add_argument(
    "--loss_cons_factor",
    type=float,
    default=0.0
)
parser.add_argument(
    "--loss_gt_factor",
    type=float,
    default=1.0
)
parser.add_argument(
    "--loss_pose_rot_factor",
    type=float,
    default=0.0
)
parser.add_argument(
    "--loss_pose_trans_factor",
    type=float,
    default=0.0
)

parser.add_argument(
    "--seed",
    type=int,
    default=123
)
parser.add_argument(
    "--max_scale",
    type=int,
    default=1
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-6
)

parser.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    choices=["adam", "sgd"],
)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)

#original image size (hardcoded for TUM!)
ORIG_HEIGHT = 480
ORIG_WIDTH = 640

#image size used for depth prediction
DEPTH_PRED_HEIGHT = 256 #256 #256
DEPTH_PRED_WIDTH = 320 #320 #320

#image size used for SLAM
SLAM_HEIGHT = 64 #64 #64#128128
SLAM_WIDTH = 80 #80 #80#160

#DEPTH PREDICTION MODEL PARAMETERS
#TODO: implement with args
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE = "models/r18_rectified_nyu/dispnet_model_best.pth.tar"
PRETRAINED_DISPNET_PATH = os.path.join(CURR_DIR, MODEL_FILE)
RESNET_LAYERS = 18




if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize DepthPrediction Network
    depth_net = SCSfmWrapper(
        device=device,
        pretrained=True,
        pretrained_path=PRETRAINED_DISPNET_PATH,
        resnet_layers = RESNET_LAYERS)
    
    # Inizialize optimizer
    if args.optimizer == "adam":
        optim = Adam(depth_net.parameters(), lr = args.learning_rate)
    elif args.optimizer == "sgd":
        optim = SGD(depth_net.parameters(), lr = args.learning_rate,momentum=0,nesterov=False)


    # load dataset
    if args.dataset == "tum":
        #need to have images in 320x256 size as input to sc-sfml net. Thus first we rescale by 1.875, then crop horizontally
        if os.path.exists(args.sequences):
            # path to sequences.txt file
            sequences = args.sequences
        elif args.sequences is not None:
            # pass single sequence as tuple
            sequences = (args.sequences,)
        else:
            sequences = None
        height = DEPTH_PRED_HEIGHT#256 #480/2
        width = int(np.ceil(ORIG_WIDTH*(DEPTH_PRED_HEIGHT/ORIG_HEIGHT))) #342 #ceil(640/2)
        cropped_width = DEPTH_PRED_WIDTH #320 #crop hotizontally (equal margin at both sides)
        dataset = TUM(args.dataset_path, seqlen=args.seq_length, height=height, width=width, cropped_width=cropped_width, sequences=sequences,
                      dilation=args.seq_dilation,stride = args.seq_stride,start = args.seq_start, end = args.seq_end)
    elif args.dataset == "nyu":
        # right now only working with rectified pictures as provided by SfM-github
        dataset = NYU(args.dataset_path, version="rectified", seqlen=args.seq_length, height=DEPTH_PRED_HEIGHT, width=DEPTH_PRED_WIDTH, sequences=None)
    elif args.dataset == "nyu-regular":
        # NOT SUPPORTED YET!!!
        dataset = NYU(args.dataset_path, version="regular", seqlen=args.seq_length, height=DEPTH_PRED_HEIGHT, width=DEPTH_PRED_WIDTH, sequences=None)

    # get data
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False )

    writer = SummaryWriter(comment=args.model_name)
    model_path = os.path.join(args.debug_path, args.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # log args
    args_path = os.path.join(model_path, "args.txt")
    with open(args_path, 'w') as file:
        file.write(json.dumps(vars(args)))

    # Training
    epochs = args.num_epochs
    losses = []
    counter = {"every": 0, "batch": 0, "detailed": 0}
    log_summary = []

    scale_coeff = None
    vmin_vis = None
    vmax_vis = None

    current_iter = 0

    for e_idx in range(epochs):
        # TODO: remove gt depth dependency
        #for batch_idx, (colors, depths, intrinsics, poses, *_) in enumerate(loader):
        for batch_idx, (colors, depths, intrinsics, *rest) in enumerate(loader):
            # Stop training after maximum number of batches
            if batch_idx >= args.max_num_batches:
                continue
            current_iter += 1

            colors = colors.to(device)
            depths = depths.to(device)
            intrinsics = intrinsics.to(device)

            # TODO: make NYU data loader return dummy gt poses
            if args.dataset == 'tum':
                gt_poses = rest[0].to(device)
            else:
                gt_poses = torch.eye(4, device=device).view(1, 4, 4).repeat(args.batch_size, args.seq_length, 1, 1)

            # Hard coded
            batch_loss = {}
            batch_val = {}
            pred_depths = []

            # Initialize SLAM and pointclouds
            slam = PointFusion(odom=args.odometry, dsratio=4, device=device)
            pointclouds = Pointclouds(device=device)
            live_frame = None

            # Scale intrinsics since SLAM works on downsampled images
            intrinsics_slam = intrinsics.clone().detach()
            intrinsics_slam[:, :, 0, :] = intrinsics_slam[:, :, 0, :] * SLAM_WIDTH / DEPTH_PRED_WIDTH
            intrinsics_slam[:, :, 1, :] = intrinsics_slam[:, :, 1, :] * SLAM_HEIGHT / DEPTH_PRED_HEIGHT
            # Intrinsics are already scaled in TUM dataloader!
            intrinsics_depth = intrinsics.clone().detach()

            # Iterate over frames in Sequence
            for pred_index in range(0, args.seq_length):
                depth_net.zero_grad()

                # Logging?
                # if batch_idx % args.log_freq == 0 and pred_index == args.seq_length - 1 and not args.debug_path is None:
                if e_idx % args.log_freq == 0 and batch_idx == 0 and pred_index == args.seq_length - 1 and not args.debug_path is None:
                    log = True
                else:
                    log = False

                # projection mode:
                if args.projection_mode == "previous":
                    ref_index = pred_index - 1
                elif args.projection_mode == "first":
                    ref_index = 0
                else:
                    ref_index = pred_index - 1

                # get input tensors
                input_dict = {"device": device}
                input_dict["rgb"] = (colors[:, pred_index, ::] / 255.0).permute(0, 3, 1, 2)
                input_dict["rgb_ref"] = (colors[:, ref_index, ::] / 255.0).permute(0, 3, 1, 2)

                input_dict["depth"] = depths[:, pred_index, ::].permute(0, 3, 1, 2)
                input_dict["depth_ref"] = depths[:, ref_index, ::].permute(0, 3, 1, 2)
                input_dict["gt_poses"] = gt_poses[:, pred_index, ::]
                input_dict["gt_poses_ref"] = gt_poses[:, ref_index, ::]
                input_dict["intrinsic_slam"] = intrinsics_slam
                input_dict["intrinsic_depth"] = intrinsics_depth

                # scale depth (only in very first iteration)
                if scale_coeff is None or vmax_vis is None or vmin_vis is None:
                    depth_predictions = depth_net(input_dict["rgb"])
                    input_dict["pred_depths"] = depth_predictions
                    scale_coeff, vmin_vis, vmax_vis = compute_scaling_coef(args, input_dict)
                    depth_net.scale_coeff = scale_coeff

                # predict depth
                # TODO: seems inefficient, could also store previous depth prediction
                depth_predictions = depth_net(input_dict["rgb"])
                input_dict["pred_depths_ref"] = depth_net(input_dict["rgb_ref"])
                input_dict["pred_depths"] = depth_predictions #depth_net(input_dict["rgb"])

                #input_dict["pred_depths_ref"][0] = input_dict["pred_depths_ref"][0].detach()
                #input_dict["pred_depths"][0] = input_dict["pred_depths"][0].detach()

                #TODO: use it to test with gt depth
                if USE_GT_DEPTH:
                    print("WARNING: USING GT DEPTH")
                    input_dict["pred_depths_ref"] = list()
                    input_dict["pred_depths_ref"].append(input_dict["depth_ref"])
                    input_dict["pred_depths"] = list()
                    input_dict["pred_depths"].append(input_dict["depth"])

                # Downsample (since depth prediction does not work in (120,160))
                colors_slam = torch.nn.functional.interpolate(input=input_dict["rgb"], size=(SLAM_HEIGHT, SLAM_WIDTH), mode="bicubic")

                # Multi-scale: take largest scale only for SLAM
                pred_depths_slam = torch.nn.functional.interpolate(input=input_dict["pred_depths"][0], size=(SLAM_HEIGHT, SLAM_WIDTH), mode="nearest")

                input_dict["rgb_slam"] = colors_slam
                input_dict["pred_depths_slam"] = pred_depths_slam

                # SLAM to update poses
                slam, pointclouds, live_frame, relative_poses = slam_step(input_dict, slam, pointclouds, live_frame, device, args)

                # compute relative gt and slam poses between reference frame and pose
                
                input_dict["gt_rel_poses"] = torch.matmul(torch.inverse(input_dict["gt_poses_ref"]), input_dict["gt_poses"]).unsqueeze(1)

                # Log difference in poses for analysis (TODO)
                mag_transl, mag_rot = compute_relative_pose_magnitudes(input_dict["gt_rel_poses"].squeeze(1).detach().cpu().numpy())



                if args.projection_mode == "previous":
                    input_dict["slam_rel_poses"] = relative_poses
                elif args.projection_mode == "first":
                    input_dict["slam_rel_poses"] = live_frame.poses

                # choose between using gt poses or slam poses for reprojection
                if args.train_odometry == "slam":
                    input_dict["pose"] = input_dict["slam_rel_poses"] #.detach()
                elif args.train_odometry == "gt":
                    #take relative gt pose between 
                    input_dict["pose"] = torch.matmul(torch.inverse(input_dict["gt_poses_ref"]), input_dict["gt_poses"]).unsqueeze(1)
                else: raise ValueError("invalid --train_odometry argument")

                # TODO: use it to visualize SLAM
                if VISUALIZE_SLAM:
                    # SLAM Vis
                    # o3d.visualization.draw_geometries([pointclouds.open3d(0)])
                    True

                # First frame: SLAM only, for pose, no backpropagation since we don't have poses / reference frame
                slam_grad = args.loss_pose_rot_factor>0 or args.loss_pose_trans_factor #whether gradients are flowing trough slam
                if pred_index == 0 or (slam_grad and pred_index<args.seq_length-1):
                    continue

                # compute loss, backprop, and optimize (depth consistency loss is computed every frame of sequence)
                loss_dict = pred_loss_unified(args, input_dict, slam, pointclouds, live_frame)
                loss = loss_dict["com"]
                if not USE_GT_DEPTH:
                    #loss.backward()
                    #torch.autograd.set_detect_anomaly(True)
                    loss.backward() #retain_graph=True)
                    optim.step()
                print("Epoch: {}, Batch_idx: {}.{} / Loss : {:.4f}".format(e_idx, batch_idx, pred_index, loss))

                # Validation in train mode(TODO)
                # compute various errors by comparing gt to the biggest scale prediction
                # error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
                gt = input_dict["depth"][:,0,:,:]
                pred = input_dict["pred_depths"][0][:,0,:,:]
                validation_errors = compute_errors(pred, gt, args.dataset)
                print("Validation errors:", validation_errors)
                # for tensorboard
                #for backward compatibility...
                loss_dict["val_abs_diff"] = torch.tensor(validation_errors[0])
                loss_dict["val_abs_rel"] = torch.tensor(validation_errors[1])
                val_dict =  {}
                val_dict["val_abs_diff"] = torch.tensor(validation_errors[0])
                val_dict["val_abs_rel"] = torch.tensor(validation_errors[1])
                val_dict["val_sq_rel"] = torch.tensor(validation_errors[2])
                val_dict["val_a1"] = torch.tensor(validation_errors[3])
                val_dict["val_a2"] = torch.tensor(validation_errors[4])
                val_dict["val_a3"] = torch.tensor(validation_errors[5])

                # log pose difference
                val_dict["mag_gt_transl"] = mag_transl
                val_dict["mag_gt_rot"] = mag_rot

                # Validation in eval mode
                if EVAL_VALIDATION:
                    depth_net.disp_net.eval()
                    input_dict["pred_depths_eval"] = depth_net(input_dict["rgb"])
                    gt_eval = input_dict["depth"][:,0,:,:]
                    pred_eval = input_dict["pred_depths_eval"][0][:,0,:,:]
                    validation_errors_eval = compute_errors(pred_eval, gt_eval, args.dataset)
                    print("Validation errors eval:", validation_errors_eval)
                    # for tensorboard
                    loss_dict["val_abs_diff_eval"] = torch.tensor(validation_errors_eval[0])
                    loss_dict["val_abs_rel_eval"] = torch.tensor(validation_errors_eval[1])

                    val_dict["val_abs_diff_eval"] = torch.tensor(validation_errors_eval[0])
                    val_dict["val_abs_rel_eval"] = torch.tensor(validation_errors_eval[1])
                    val_dict["val_sq_rel_eval"] = torch.tensor(validation_errors_eval[2])
                    val_dict["val_a1_eval"] = torch.tensor(validation_errors_eval[3])
                    val_dict["val_a2_eval"] = torch.tensor(validation_errors_eval[4])
                    val_dict["val_a3_eval"] = torch.tensor(validation_errors_eval[5])
                    # set back to train mode
                    depth_net.disp_net.train()


                # Log
                if log:
                    print("Logging depths images to {}".format(model_path))
                    # Color Vis
                    mpl.pyplot.imsave("{}/{}_{}_color.jpg".format(model_path, e_idx, batch_idx),
                                      1.0 * np.vstack(input_dict["rgb"].detach().cpu().permute(0, 2, 3, 1).squeeze().numpy()))
                    # Depth Vis
                    mpl.pyplot.imsave("{}/{}_{}_gt.jpg".format(model_path, e_idx, batch_idx),
                               np.vstack(input_dict["depth"].detach().cpu().squeeze().cpu().numpy()),
                                      vmin=vmin_vis, vmax=vmax_vis)
                    mpl.pyplot.imsave("{}/{}_{}_pred.jpg".format(model_path, e_idx, batch_idx),
                               np.vstack(input_dict["pred_depths"][0].detach().squeeze().cpu().numpy()),
                                      vmin=vmin_vis, vmax=vmax_vis)
                    if EVAL_VALIDATION:
                        mpl.pyplot.imsave("{}/{}_{}_pred_eval.jpg".format(model_path, e_idx, batch_idx),
                                          np.vstack(input_dict["pred_depths_eval"][0].detach().squeeze().cpu().numpy()),
                                          vmin=vmin_vis, vmax=vmax_vis)
                    model_save_path = os.path.join(model_path, "model_epoch_{}".format(e_idx))
                    print("Saving model to {}".format(model_save_path))
                    depth_net.save_model(model_save_path, e_idx, loss_dict)

                # Tensorboard
                eff_batch_size = colors.shape[0]
                for loss_type in loss_dict.keys():
                    writer.add_scalar("Perstep_loss/_{}".format(loss_type), loss_dict[loss_type].item(), counter["every"])
                    if not loss_type in batch_loss.keys():
                        batch_loss[loss_type] = loss_dict[loss_type].item() * 1 / args.seq_length #/ eff_batch_size
                    else:
                        batch_loss[loss_type] += loss_dict[loss_type].item() * 1 / args.seq_length #/ eff_batch_size
                # validation
                for val_type in val_dict.keys():
                    writer.add_scalar("Perstep_validation/_{}".format(val_type), val_dict[val_type].item(), counter["every"])
                    if not val_type in batch_val.keys():
                        batch_val[val_type] = val_dict[val_type].item() * 1 / args.seq_length #/ eff_batch_size
                    else:
                        batch_val[val_type] += val_dict[val_type].item() * 1 / args.seq_length #/ eff_batch_size

                counter["every"] += 1

                pred_depths.append(input_dict["pred_depths"][0].permute(0, 2, 3, 1).unsqueeze(1))

            for loss_type in loss_dict.keys():
                writer.add_scalar("Batchwise_loss/_{}".format(loss_type), batch_loss[loss_type], counter["batch"])
            for val_type in val_dict.keys():
                writer.add_scalar("Batchwise_validation/_{}".format(val_type), batch_val[val_type], counter["batch"])

            # Log training progress
            
            #add validation scores
            batch_loss = {**batch_loss,**batch_val}

            batch_loss["run"] = args.model_name
            batch_loss["epoch"] = e_idx
            batch_loss["batch"] = batch_idx
            batch_loss["iteration"] = current_iter

            log_summary.append(batch_loss)
            #log_summary[-1] = {**log_summary[-1],**batch_val}

            counter["batch"] +=1
            pred_depths = torch.cat(pred_depths, dim= 1)

    # Saving training summary
    keys = log_summary[0].keys()
    with open(os.path.join(model_path, "training_summary.csv"), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(log_summary)

    writer.flush()
    print("Training done, info logged to {}/training_summary.csv".format(model_path))





