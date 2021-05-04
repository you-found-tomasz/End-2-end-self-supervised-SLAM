from argparse import ArgumentParser, RawTextHelpFormatter

import torch
from torch.utils.data import DataLoader

from gradslam.datasets.tum import TUM
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
#from perception.SC_SfMLearner_Release.scsfmwrapper import SCSfmWrapper
from end2endslam.pointfusion.scsfmwrapper2 import SCSfmWrapper
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
import numpy as np
import os

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
    choices=["tum"],
    help="Dataset to use. Supported options:\n"
    " tum = Iterative Closest Point\n",
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
    default="runs",
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
    "--log_freq",
    type=int,
    default=10,
    help="Frequency for logging"
)
parser.add_argument(
    "--loss_cons_factor",
    type=float,
    default=0.0
)
parser.add_argument(
    "--loss_reproj_factor",
    type=float,
    default=0.0
)
parser.add_argument(
    "--loss_gt_factor",
    type=float,
    default=1.0
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True
)
parser.add_argument(
    "--width",
    type=int,
    default=160
)
parser.add_argument(
    "--height",
    type=int,
    default=120
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

if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize DepthPrediction Network
    DepthPredictor = SCSfmWrapper(device)
    optim = SGD(DepthPredictor.disp_net.decoder.parameters(), lr = 1e-4)

    # load dataset
    if args.dataset == "tum":
        # doesn't work with 120 / 160, probably has something to do with upsampling / downsampling in ResNet
        dataset = TUM(args.dataset_path, seqlen=args.seq_length, height=480, width=640, sequences=None)
        # dataset = TUM(args.dataset_path, seqlen=args.seq_length, height=480, width=640, sequences=None)

    # get data
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    model_path = os.path.join(args.debug_path, args.model_name)

    writer = SummaryWriter(model_path)

    # Training
    epochs = 50
    losses = []
    counter = {"every": 0, "batch": 0, "detailed": 0}
    for e_idx in range(epochs):
        # TODO: remove gt depth dependency
        for batch_idx, (colors, depths, intrinsics, poses, *_) in enumerate(loader):
            colors = colors.to(device)
            depths = depths.to(device)
            intrinsics = intrinsics.to(device)
            poses = poses.to(device)

            # Hard coded
            batch_loss = {}
            pred_depths = []

            # Initialize SLAM and pointclouds
            slam = PointFusion(odom=args.odometry, dsratio=4, device=device)
            pointclouds = Pointclouds(device=device)
            prev_frame = None

            # Scale intrinsics since SLAM works on downsampled images
            intrinsics[:, :, 0, :] = intrinsics[:, :, 0, :] * args.width / 640
            intrinsics[:, :, 1, :] = intrinsics[:, :, 1, :] * args.height / 480

            # Iterate over frames in Sequence
            for pred_index in range(1, args.seq_length):
                DepthPredictor.zero_grad()

                # get input tensors
                input_dict = {"device": device}
                input_dict["rgb"] = (colors[:, pred_index, ::] / 255.0).permute(0, 3, 1, 2)
                input_dict["rgb_ref"] = (colors[:, pred_index - 1, ::] / 255.0).permute(0, 3, 1, 2)
                # Initial poses are necessary. We use identity here, could also use gt
                #input_dict["pose"] = torch.eye(4, device=device).view(1, 4, 4).repeat(args.batch_size, 1, 1)
                input_dict["pose"] = torch.matmul(torch.inverse(poses[:, pred_index - 1, ::]), poses[:, pred_index, ::])
                input_dict["depth"] = depths[:, pred_index, ::].permute(0, 3, 1, 2)
                input_dict["intrinsic"] = intrinsics

                if batch_idx % args.log_freq == 0 and pred_index == args.seq_length - 1 and not args.debug_path is None:
                    log = True
                else:
                    log = False

                # predict, backprop, and optimize (depth consistency loss is computed every frame of sequence)
                depth_predictions, loss_dict, slam, pointclouds, prev_frame = DepthPredictor.pred_loss_unified(args,
                        input_dict, slam, pointclouds, prev_frame, model_path, log)

                if log:
                    scaled_depth_predictions = depth_predictions
                    cv.imwrite("{}/gt.jpg".format(model_path), 40 * np.vstack(input_dict["depth"].detach().cpu().permute(0, 2, 3, 1).cpu().numpy()))
                    cv.imwrite("{}/pred.jpg".format(model_path), 40 * np.vstack(scaled_depth_predictions.detach().cpu().permute(0, 2, 3, 1).numpy()))

                loss = loss_dict["com"]
                loss.backward()
                optim.step()
                print("Epoch: {}, Batch_idx: {}.{} / Loss : {:.4f}".format(e_idx, batch_idx, pred_index, loss))

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