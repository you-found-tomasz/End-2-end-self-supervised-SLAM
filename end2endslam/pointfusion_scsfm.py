from argparse import ArgumentParser, RawTextHelpFormatter

import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import cv2 as cv
# import open3d as o3d # TODO: remove
from gradslam.structures.rgbdimages import RGBDImages
from end2endslam.dataloader.tum import TUM
from end2endslam.dataloader.nyu import NYU
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds

from end2endslam.scsfmwrapper import SCSfmWrapper
from losses.unified_loss import pred_loss_unified

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl
import matplotlib.cm as cm
import imageio

"""r Example run commands:

(leave TUM folder structure unchanged):

runfile('/home/matthias/git/End-2-end-self-supervised-SLAM/end2endslam/pointfusion_scsfm.py', 
args=['--dataset', 'tum', '--dataset_path', '/home/matthias/git/End-2-end-self-supervised-SLAM/sample_data/dataset_TUM', 
'--odometry', 'icp', '--seq_length', '10', '--batch_size', '8', '--debug_path', '/home/matthias/data/3dv_debug/', 
'--loss_photo_factor', '1', '--loss_geom_factor', '0.5', '--loss_smooth_factor', '0.1', '--loss_cons_factor', '0.0', 
'--loss_gt_factor', '0.0', '--log_freq', '1'], wdir='/home/matthias/git/End-2-end-self-supervised-SLAM/end2endslam')


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
    "--batch_size",
    type=int,
    default=8,
    help="SLAM Sequence Length"
)
parser.add_argument(
    "--log_freq",
    type=int,
    default=10,
    help="Frequency for logging"
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
    "--seed",
    type=int,
    default=123
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
    intrinsics = input_dict["intrinsic"]
    poses = input_dict["pose"]
    colors = input_dict["rgb_slam"]
    pred_depths = input_dict["pred_depths_slam"]

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
    model_path = os.path.join(args.debug_path, args.model_name)

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
            batch_loss = {}
            pred_depths = []

            # Initialize SLAM and pointclouds
            slam = PointFusion(odom=args.odometry, dsratio=4, device=device)
            pointclouds = Pointclouds(device=device)
            live_frame = None

            # Scale intrinsics since SLAM works on downsampled images
            intrinsics[:, :, 0, :] = intrinsics[:, :, 0, :] * SLAM_WIDTH / ORIG_WIDTH
            intrinsics[:, :, 1, :] = intrinsics[:, :, 1, :] * SLAM_HEIGHT / ORIG_HEIGHT

            # Iterate over frames in Sequence
            for pred_index in range(0, args.seq_length):
                depth_net.zero_grad()

                # Logging?
                if batch_idx % args.log_freq == 0 and pred_index == args.seq_length - 1 and not args.debug_path is None:
                    log = True
                else:
                    log = False

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
                input_dict["depth_ref"] = depths[:, pred_index - 1, ::].permute(0, 3, 1, 2)
                input_dict["intrinsic"] = intrinsics

                # predict depth
                # TODO: seems inefficient, could also store previous depth prediction
                depth_predictions = depth_net(input_dict["rgb"])
                input_dict["pred_depths_ref"] = depth_net(input_dict["rgb_ref"])
                input_dict["pred_depths"] = depth_predictions

                #TODO: use it to test with gt
                if False:
                    input_dict["pred_depths_ref"] = input_dict["depth_ref"]
                    input_dict["pred_depths"] = input_dict["depth"]

                # Downsample (since depth prediction does not work in (120,160))
                colors_slam = torch.nn.functional.interpolate(input=input_dict["rgb"], size=(SLAM_HEIGHT, SLAM_WIDTH), mode="bicubic")
                pred_depths_slam = torch.nn.functional.interpolate(input=input_dict["pred_depths"], size=(SLAM_HEIGHT, SLAM_WIDTH), mode="nearest")
                input_dict["rgb_slam"] = colors_slam
                input_dict["pred_depths_slam"] = pred_depths_slam

                # SLAM to update poses
                slam, pointclouds, live_frame = slam_step(input_dict, slam, pointclouds, live_frame, device)
                input_dict["pose"] = live_frame.poses.detach()
                # TODO: use it to visualize SLAM
                if False:
                    # SLAM Vis
                    o3d.visualization.draw_geometries([pointclouds.open3d(0)])

                # First frame: SLAM only, for pose, no backpropagation since we don't have poses / reference frame
                if pred_index == 0:
                    continue

                # predict, backprop, and optimize (depth consistency loss is computed every frame of sequence)
                loss_dict = pred_loss_unified(args, input_dict, slam, pointclouds, live_frame)
                loss = loss_dict["com"]
                loss.backward()
                optim.step()
                print("Epoch: {}, Batch_idx: {}.{} / Loss : {:.4f}".format(e_idx, batch_idx, pred_index, loss))

                # Log
                if log:
                    print("Logging depths images to {}".format(model_path))
                    # Depth Vis 1
                    scaled_depth_predictions = input_dict["pred_depths"]
                    cv.imwrite("{}/gt.jpg".format(model_path),
                               40 * np.vstack(input_dict["depth"].detach().cpu().permute(0, 2, 3, 1).cpu().numpy()))
                    cv.imwrite("{}/pred.jpg".format(model_path),
                               40 * np.vstack(scaled_depth_predictions.detach().cpu().permute(0, 2, 3, 1).numpy()))
                    # Depth Vis 2
                    # GT
                    vis_gt_depth = input_dict["depth"].detach().cpu().numpy()
                    vmax = np.percentile(vis_gt_depth, 95)
                    vmin = vis_gt_depth.min()
                    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    gt_depth_np = (mapper.to_rgba(vis_gt_depth[0, 0, :, :])[:, :, :3] * 255).astype(np.uint8)
                    imageio.imwrite(os.path.join(model_path, "debug_depth_gt.png"), gt_depth_np)
                    # Prediction
                    vis_pred_depth = input_dict["pred_depths"].detach().cpu().numpy()
                    vmax = np.percentile(vis_pred_depth, 95)
                    vmin = vis_pred_depth.min()
                    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    pred_depth_np = (mapper.to_rgba(vis_pred_depth[0, 0, :, :])[:, :, :3] * 255).astype(np.uint8)
                    imageio.imwrite(os.path.join(model_path, "debug_depth_pred.png"), pred_depth_np)

                # Tensorboard
                for loss_type in loss_dict.keys():
                    writer.add_scalar("Perstep_loss/_{}".format(loss_type), loss_dict[loss_type].item(), counter["every"])
                    if not loss_type in batch_loss.keys():
                        batch_loss[loss_type] = loss_dict[loss_type].item() * 0.1
                    else:
                        batch_loss[loss_type] += loss_dict[loss_type].item() * 0.1

                counter["every"] += 1

                pred_depths.append(depth_predictions.permute(0, 2, 3, 1).unsqueeze(1))

            for loss_type in loss_dict.keys():
                writer.add_scalar("Batchwise_loss/_{}".format(loss_type), batch_loss[loss_type], counter["batch"])

            counter["batch"] +=1
            pred_depths = torch.cat(pred_depths, dim= 1)


