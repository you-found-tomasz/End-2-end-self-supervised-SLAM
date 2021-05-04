from argparse import ArgumentParser, RawTextHelpFormatter

import torch
from torch.utils.data import DataLoader

import numpy as np
import os

from gradslam.structures.rgbdimages import RGBDImages

#from gradslam.datasets.tum import TUM
from end2endslam.dataloader.tum import TUM
from end2endslam.dataloader.nyu import NYU
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
#from perception.SC_SfMLearner_Release.scsfmwrapper import SCSfmWrapper

from end2endslam.scsfmwrapper import SCSfmWrapper
from end2endslam.losses import pred_loss_depth_consistency, pred_loss_reproj, pred_loss_depth


from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

"""r Example run commands:

(leave TUM folder structure unchanged):

Debug mode with visulizations saved to given path:
pointfusion_scsfm_brucker.py --dataset tum --dataset_path "/home/matthias/data/tum_test_1" --odometry icp 
--loss "depth_consistency"
--debug_path "/home/matthias/data/"
Plain mode:
pointfusion_scsfm_brucker.py --dataset tum --dataset_path "/home/matthias/data/tum_test_1" --odometry icp 
--loss "depth_consistency"


"""

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
    help="Odometry method to use. Supported options:\n"
    " gt = Ground Truth odometry\n"
    " icp = Iterative Closest Point\n"
    " gradicp (*default) = Differentiable Iterative Closest Point\n",
)
parser.add_argument(
    "--sequences",
    type=str,
    default=None,
    help="Path to .txt file containing sequences. \n"
    " If it doesn't work for TUM, leave option out, it then takes all sequences in dataset_path"
)
parser.add_argument(
    "--debug_path",
    type=str,
    default=None,
    help="Debug, Depth and SLAM Map Visualization"
)
parser.add_argument(
    "--seq_length",
    type=int,
    default=10,
    help="SLAM Sequence Length"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="SLAM Sequence Length"
)
parser.add_argument(
    "--loss",
    type=str,
    default="depth_consistency",
    choices=["gt_depth", "depth_consistency", "reprojection"]
)
args = parser.parse_args()

#original image size (hardcoded for TUM!)
ORIG_HEIGHT = 480
ORIG_WIDTH = 640

#image size used for depth prediction
DEPTH_PRED_HEIGHT = 256
DEPTH_PRED_WIDTH = 320

#image size used for SLAM
SLAM_HEIGHT = 64#128
SLAM_WIDTH = 80#160

#DEPTH PREDICTION MODEL PARAMETERS
#TODO: implement with args
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE = "models/r18_rectified_nyu/dispnet_model_best.pth.tar"
PRETRAINED_DISPNET_PATH = os.path.join(CURR_DIR, MODEL_FILE)
RESNET_LAYERS = 18


def slam_step(input_dict, slam, pointclouds, prev_frame,device):
    """ Perform SLAM step
    """
    # get inputs
    #colors = input_dict["rgb"]
    intrinsics = input_dict["intrinsic"]
    poses = input_dict["pose"]
    #pred_depths = input_dict["pred_depths"]

    colors = input_dict["rgb_slam"]
    pred_depths = input_dict["pred_depths_slam"]

    # Ground truth Depth for comparison (only for debug)
    # if DEBUG_PATH:
    #     gt_depths = input_dict["depth"]
    #     gt_depths = torch.nn.functional.interpolate(input=gt_depths, size=(120, 160), mode="nearest")
    #     gt_depths_u = torch.unsqueeze(gt_depths, 1).permute(0, 1, 3, 4, 2)


    # added artificial sequence length dimension and then don't use it (stupid but necessary)
    # permute since slam does NOT support channels_first = True
    colors_u = torch.unsqueeze(colors, 1).permute(0, 1, 3, 4, 2)
    pred_depths_u = torch.unsqueeze(pred_depths, 1).permute(0, 1, 3, 4, 2)
    poses_u = torch.unsqueeze(poses, 1)

    # SLAM
    rgbdimages = RGBDImages(colors_u, pred_depths_u, intrinsics, poses_u, channels_first=False, device=device)
    live_frame = rgbdimages[:, 0]
    pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)

    return slam, pointclouds, live_frame



if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize DepthPrediction Network
    depth_net = SCSfmWrapper(
        device=device,
        pretrained=True,
        pretrained_path=PRETRAINED_DISPNET_PATH,
        resnet_layers = RESNET_LAYERS)
    optim = Adam(depth_net.parameters(), lr = 1e-6)

    # load dataset
    if args.dataset == "tum":
        #need to have images in 320x256 size as input to sc-sfml net. Thus first we rescale by 1.875, then crop horizontally
        height = DEPTH_PRED_HEIGHT#256 #640/2
        width = int(np.ceil(ORIG_WIDTH*(DEPTH_PRED_HEIGHT/ORIG_HEIGHT))) #342 #ceil(480/2)
        cropped_width = DEPTH_PRED_WIDTH #320 #crop hotizontally (equal margin at both sides)
        dataset = TUM(args.dataset_path, seqlen=args.seq_length, height=height, width=width, cropped_width=cropped_width, sequences=None)
    elif args.dataset == "nyu":
        # right now only working with rectified pictures as provided by SfM-github
        dataset = NYU(args.dataset_path, version="rectified", seqlen=args.seq_length, height=DEPTH_PRED_HEIGHT, width=DEPTH_PRED_WIDTH, sequences=None)
    elif args.dataset == "nyu-regular":
        # NOT SUPPORTET YET!!!
        dataset = NYU(args.dataset_path, version="regular", seqlen=args.seq_length, height=DEPTH_PRED_HEIGHT, width=DEPTH_PRED_WIDTH, sequences=None)

    # get data
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False )

    writer = SummaryWriter()

    # Training
    epochs = 50
    losses = []
    counter = {"every": 0, "batch": 0, "detailed": 0}
    for e_idx in range(epochs):
        # TODO: remove gt depth dependency
        #for batch_idx, (colors, depths, intrinsics, poses, *_) in enumerate(loader):
        for batch_idx, (colors, depths, intrinsics, *_) in enumerate(loader):
            colors = colors.to(device)
            depths = depths.to(device)
            intrinsics = intrinsics.to(device)
            #poses = poses.to(device)

            # Hard coded
            batch_loss = 0
            pred_depths = []

            # Initialize SLAM and pointclouds
            slam = PointFusion(odom=args.odometry, dsratio=4, device=device)
            pointclouds = Pointclouds(device=device)
            live_frame = None

            # Scale intrinsics since SLAM works on downsampled images
            intrinsics[:, :, 0, :] = intrinsics[:, :, 0, :] * SLAM_WIDTH / ORIG_WIDTH
            intrinsics[:, :, 1, :] = intrinsics[:, :, 1, :] * SLAM_HEIGHT / ORIG_HEIGHT

            # Iterate over frames in Sequence
            for pred_index in range(args.seq_length):
                depth_net.zero_grad()

                # get input tensors
                input_dict = {"device": device}
                input_dict["rgb"] = (colors[:, pred_index, ::] / 255.0).permute(0, 3, 1, 2)
                input_dict["rgb_ref"] = (colors[:, pred_index - 1, ::] / 255.0).permute(0, 3, 1, 2)
                # Initial poses are necessary. We use identity here, could also use gt
                poses = torch.eye(4, device=device).view(1, 4, 4).repeat(args.batch_size, 1, 1)
                # correct for last batch size
                if poses.shape[0] != colors.shape[0]: 
                    poses = poses[0:colors.shape[0], :, :]
                input_dict["pose"] = poses
                
                # input_dict["pose"] = torch.matmul(torch.inverse(poses[:, pred_index - 1, ::]), poses[:, pred_index, ::])
                input_dict["depth"] = depths[:, pred_index, ::].permute(0, 3, 1, 2)
                input_dict["intrinsic"] = intrinsics


                # predict depth
                depth_predictions = depth_net(input_dict["rgb"])
                input_dict["pred_depths"] = depth_predictions

                # Downsample (since depth prediction does not work in (120,160))
                colors_slam = torch.nn.functional.interpolate(input=input_dict["rgb"], size=(SLAM_HEIGHT, SLAM_WIDTH), mode="bicubic")
                pred_depths_slam = torch.nn.functional.interpolate(input=input_dict["pred_depths"], size=(SLAM_HEIGHT, SLAM_WIDTH), mode="nearest")
                input_dict["rgb_slam"] = colors_slam
                input_dict["pred_depths_slam"] = pred_depths_slam

                #SLAM
                slam, pointclouds, live_frame = slam_step(input_dict,slam, pointclouds, live_frame, device)

                # predict, backprop, and optimize (depth consistency loss is computed every frame of sequence)
                if args.loss == "depth_consistency":
                    loss = pred_loss_depth_consistency(input_dict, slam, pointclouds, live_frame, args.debug_path,device)
                elif args.loss == "reprojection":
                    depth_predictions, loss = pred_loss_reproj(depth_net,input_dict)
                elif args.loss == "gt_depth":
                    depth_predictions, loss = pred_loss_depth(depth_net,input_dict)

                batch_loss += loss.item() * 0.1
                loss.backward()
                optim.step()
                print("Batch_idx: {} / Seq: {} / Loss ({}): {}".format(batch_idx, pred_index, args.loss, loss))

                # Tensorboard
                writer.add_scalar("Depth/Perstep_loss", loss.item(), counter["every"])
                counter["every"] +=1

                pred_depths.append(depth_predictions.permute(0, 2, 3, 1).unsqueeze(1))

            writer.add_scalar("Depth/Batchwise_loss", batch_loss, counter["batch"] )
            losses.append(batch_loss)
            counter["batch"] +=1

            pred_depths = torch.cat(pred_depths, dim= 1)


